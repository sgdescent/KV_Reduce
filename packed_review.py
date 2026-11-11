"""Qwen2-only streaming packed-cache prototype for review systems measurements.

Existing packed affine quantizer and Triton kernel, with no full-cache rebuild
on decode. Key groups are committed after rollback; speculative keys stay BF16.
This is a measured prototype, not an optimized production implementation.
"""
import argparse, gc, hashlib, json, time, types
from pathlib import Path
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from packed_kv_cache import pack_kivi_kv, pack_per_token_affine, pack_per_channel_grouped_affine, tensor_bytes
from triton_packed_kv_attention import packed_kv_decode_attention
from kv_utils import load_causal_lm, load_tokenizer, iter_token_blocks, as_legacy_cache
from benchmark_spec_kv_quantization import cached_prefill, cached_step, crop_cache_to_length, greedy_target_generate_with_margins

class PackedLayer(DynamicLayer):
    def __init__(self,k,v,residual):
        super().__init__(); self.k,self.v=pack_kivi_kv(k,v,k_bits=4,v_bits=4,group_size=32,residual_length=residual)
        self.residual_length=residual; self.is_initialized=True
    def get_seq_length(self): return self.k.shape[-2]
    def get_mask_sizes(self,cache_position): return self.get_seq_length()+cache_position.shape[0],0
    def append(self,k,v):
        b,h,n,d=self.k.shape; extra=k.shape[-2]
        self.k.residual=torch.cat((self.k.residual,k),dim=-2).contiguous()
        self.k.shape=(b,h,n+extra,d)
        added=pack_per_token_affine(v,4)
        self.v.payload=torch.cat((self.v.payload.reshape(b,h,-1),added.payload.reshape(b,h,-1)),dim=-1).flatten()
        self.v.scale=torch.cat((self.v.scale,added.scale),dim=-2)
        self.v.offset=torch.cat((self.v.offset,added.offset),dim=-2)
        self.v.shape=(b,h,n+extra,d)
    def commit(self):
        old=self.k
        added=pack_per_channel_grouped_affine(old.residual,4,group_size=32,residual_length=self.residual_length)
        if added.prefix_length:
            b,h,_,_=old.shape
            old.payload=torch.cat((old.payload.reshape(b,h,-1),added.payload.reshape(b,h,-1)),dim=-1).flatten()
            old.scale=torch.cat((old.scale,added.scale),dim=-3)
            old.offset=torch.cat((old.offset,added.offset),dim=-3)
            old.prefix_length+=added.prefix_length
            old.residual=added.residual
    def crop(self,length):
        b,h,n,d=self.k.shape
        assert self.k.prefix_length <= length <= n, (self.k.prefix_length,length,n)
        self.k.residual=self.k.residual[...,:length-self.k.prefix_length,:].contiguous()
        self.k.shape=(b,h,length,d)
        self.v.payload=self.v.payload.reshape(b,h,n,d//2)[...,:length,:].contiguous().flatten()
        self.v.scale=self.v.scale[...,:length,:].contiguous(); self.v.offset=self.v.offset[...,:length,:].contiguous()
        self.v.shape=(b,h,length,d)
    def breakdown(self):
        return {'k_payload':tensor_bytes(self.k.payload),'v_payload':tensor_bytes(self.v.payload),
                'k_scales':tensor_bytes(self.k.scale),'v_scales':tensor_bytes(self.v.scale),
                'k_offsets':tensor_bytes(self.k.offset),'v_offsets':tensor_bytes(self.v.offset),
                'key_residual':tensor_bytes(self.k.residual)}

class PackedCache(DynamicCache):
    def __init__(self,cache,residual):
        super().__init__()
        self.layers=[PackedLayer(k,v,residual) for k,v in as_legacy_cache(cache)]
    def commit(self):
        for layer in self.layers: layer.commit()
    def breakdown(self):
        return {k:sum(layer.breakdown()[k] for layer in self.layers) for k in self.layers[0].breakdown()}

def install(model):
    for layer in model.model.layers:
        attn=layer.self_attn; original=attn.forward
        def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,cache_position=None,_original=original,**kwargs):
            if not isinstance(past_key_values,PackedCache):
                return _original(hidden_states,position_embeddings,attention_mask,past_key_values,cache_position,**kwargs)
            assert hidden_states.shape[1]==1
            shape=hidden_states.shape[:-1]; view=(*shape,-1,self.head_dim)
            q=self.q_proj(hidden_states).view(view).transpose(1,2)
            k=self.k_proj(hidden_states).view(view).transpose(1,2)
            v=self.v_proj(hidden_states).view(view).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
            packed=past_key_values.layers[self.layer_idx]; packed.append(k,v)
            out=packed_kv_decode_attention(q.contiguous(),packed.k,packed.v,split_tokens=512)
            return self.o_proj(out.transpose(1,2).reshape(*shape,-1).contiguous()),None
        attn.forward=types.MethodType(forward,attn)

def unit_test():
    torch.manual_seed(123)
    for residual in (0,128):
        k=torch.randn(1,2,1024,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        layer=PackedLayer(k,v,residual); original_prefix=layer.k.payload.clone()
        for _ in range(4):layer.append(torch.randn_like(k[:,:,:1]),torch.randn_like(v[:,:,:1]))
        layer.crop(1026); layer.commit()
        assert layer.get_seq_length()==1026
        assert torch.equal(original_prefix,layer.k.payload)
        q=torch.randn(1,12,1,128,device='cuda',dtype=torch.bfloat16)
        actual=packed_kv_decode_attention(q,layer.k,layer.v,split_tokens=512)
        expected=torch.nn.functional.scaled_dot_product_attention(q,layer.k.unpack().repeat_interleave(6,1),layer.v.unpack().repeat_interleave(6,1))
        torch.testing.assert_close(actual,expected,atol=.005,rtol=.05)
        # Exercise a key-group boundary, then roll back inside the uncommitted tail.
        for _ in range(34):layer.append(torch.randn_like(k[:,:,:1]),torch.randn_like(v[:,:,:1]))
        layer.crop(1057);layer.commit();assert layer.k.prefix_length==((1057-residual)//32)*32
    print('Packed append/crop/commit and attention checks passed',flush=True)

@torch.inference_mode()
def decode(big,small,prompt,mode,residual,steps,count,verification='batched'):
    target=cached_prefill(big,prompt,'cuda:0');tc=target.pop('cache');tl=target['logits'];length=prompt.shape[1]
    draft=cached_prefill(small,prompt,'cuda:0');dc=draft.pop('cache');dl=draft['logits']
    if mode=='packed': dc=PackedCache(dc,residual)
    generated=[];rounds=[];calls=1;draftcalls=1
    while len(generated)<count:
        start=length;proposal=[]
        for _ in range(min(steps,count-len(generated))):
            token=int(dl[..., :VOCAB].argmax(-1));proposal.append(token)
            result=cached_step(model=small,input_ids=torch.tensor([[token]]),cache=dc,cache_len=start+len(proposal)-1,device='cuda:0')
            dc=result['cache'];dl=result['logits'][:,-1];draftcalls+=1
        accepted=0
        if verification=='batched':
            verified=cached_step(model=big,input_ids=torch.tensor([proposal]),cache=tc,cache_len=start,device='cuda:0');calls+=1;tc=verified['cache'];vl=verified['logits']
            for i,token in enumerate(proposal):
                logits=tl if i==0 else vl[:,i-1]
                if int(logits[..., :VOCAB].argmax(-1))!=token:break
                accepted+=1
            correction=tl if accepted==0 else vl[:,accepted-1]
        else:
            for token in proposal:
                if int(tl[..., :VOCAB].argmax(-1))!=token:break
                verified=cached_step(model=big,input_ids=torch.tensor([[token]]),cache=tc,cache_len=start+accepted,device='cuda:0');calls+=1
                tc=verified['cache'];tl=verified['logits'][:,-1];accepted+=1
            correction=tl
        rounds.append(dict(position=len(generated),proposed=len(proposal),accepted_prefix=accepted,rejection_position=accepted if accepted<len(proposal) else -1))
        generated.extend(proposal[:accepted]);length=start+accepted
        if len(generated)>=count:break
        token=int(correction[..., :VOCAB].argmax(-1));generated.append(token)
        tc=crop_cache_to_length(tc,length);dc=crop_cache_to_length(dc,length)
        result=cached_step(model=big,input_ids=torch.tensor([[token]]),cache=tc,cache_len=length,device='cuda:0');tc=result['cache'];tl=result['logits'][:,-1];calls+=1
        result=cached_step(model=small,input_ids=torch.tensor([[token]]),cache=dc,cache_len=length,device='cuda:0');dc=result['cache'];dl=result['logits'][:,-1];draftcalls+=1;length+=1
        if mode=='packed':dc.commit()
    breakdown=dc.breakdown() if mode=='packed' else {'bf16_payload':sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in as_legacy_cache(dc))}
    return dict(tokens=generated[:count],rounds=rounds,target_calls=calls,draft_calls=draftcalls,cache_bytes=breakdown,
                absolute_acceptance=sum(r['accepted_prefix'] for r in rounds)/sum(r['proposed'] for r in rounds),
                accepted_per_round=sum(r['accepted_prefix'] for r in rounds)/len(rounds),target_calls_per_output_token=calls/len(generated))

def measured(fn):
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
    result=fn();torch.cuda.synchronize()
    return result,dict(seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

def main():
    global VOCAB
    p=argparse.ArgumentParser();p.add_argument('--context',type=int,default=1024);p.add_argument('--prompts',type=int,default=24);p.add_argument('--tokens',type=int,default=256);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:
        a.prompts, a.tokens = 1, 32
    if min(a.context, a.prompts, a.tokens) < 1:
        p.error('--context, --prompts and --tokens must be positive')
    if not torch.cuda.is_available():
        p.error('This prototype requires a CUDA GPU and Triton')
    print('WARNING: batched verifier output mismatches remain unresolved, including BF16. '
          'These timings do not establish exact-output serving performance.', flush=True)
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);unit_test()
    tokenizer=load_tokenizer('Qwen/Qwen2.5-3B')
    prompts=[x[None,:] for x in iter_token_blocks(tokenizer=tokenizer,seq_len=a.context,max_blocks=a.prompts+1,dataset_name='HuggingFaceFW/fineweb-edu',dataset_config='sample-10BT',split='train',streaming=True,shuffle=True,seed=73,skip_blocks=200)]
    assert len(prompts)==a.prompts+1
    (a.out/'prompts.json').write_text(json.dumps([x.tolist() for x in prompts]))
    big=load_causal_lm('Qwen/Qwen2.5-3B',device='cuda:0',dtype_name='bf16',attn_implementation='sdpa')
    vocab=tokenizer.vocab_size; VOCAB=vocab; references=[]
    with (a.out/'target_only.jsonl').open('w') as f:
        for i,prompt in enumerate(prompts):
            result,timing=measured(lambda:greedy_target_generate_with_margins(big_model=big,prompt_ids=prompt,max_new_tokens=a.tokens,big_device='cuda:0',shared_vocab_size=vocab))
            references.append(result['tokens']);f.write(json.dumps(dict(prompt=i,warmup=i==0,**result,**timing))+'\n');f.flush()
    small=load_causal_lm('Qwen/Qwen2.5-1.5B',device='cuda:0',dtype_name='bf16',attn_implementation='sdpa');install(small)
    with (a.out/'speculative.jsonl').open('w') as f:
        for i,prompt in enumerate(prompts):
            # Alternate order to reduce systematic warm-cache effects.
            configs=[('bf16',128),('packed',0),('packed',128)]
            if i%2:configs.reverse()
            for mode,residual in configs:
                result,timing=measured(lambda:decode(big,small,prompt,mode,residual,4,a.tokens))
                match=result['tokens']==references[i]
                row=dict(prompt=i,warmup=i==0,mode=mode,residual=residual,verification='batched',matches_target=match,**result,**timing)
                f.write(json.dumps(row)+'\n');f.flush();print(json.dumps({k:v for k,v in row.items() if k not in ('tokens','rounds')}),flush=True)
                if not match:
                    audit,_=measured(lambda:decode(big,small,prompt,mode,residual,4,a.tokens,verification='sequential'))
                    f.write(json.dumps(dict(prompt=i,mode=mode,residual=residual,verification='sequential_audit',matches_target=audit['tokens']==references[i],**audit))+'\n');f.flush()
                    assert audit['tokens']==references[i], 'Sequential rollback/correctness audit failed'
    (a.out/'complete.json').write_text(json.dumps(dict(context=a.context,prompts=a.prompts,tokens=a.tokens,torch=torch.__version__,gpu=torch.cuda.get_device_name(),note='Greedy Qwen2 prototype; packed affine BF16 metadata; batch 1; all timings include prefill. Warmup prompt excluded from analysis.'),indent=2))

if __name__=='__main__':main()
