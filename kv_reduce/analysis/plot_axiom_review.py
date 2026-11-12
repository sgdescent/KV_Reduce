"""Recreate AXIOM review figures from the verified raw archive; CPU only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from kv_reduce.analysis.analyze_axiom_review import analyze, compare
from kv_reduce.analysis.raw_axiom_archive import BUNDLE, rebuild_metrics

ORDER = ['1024_r0_s0', '1024_r128_s0', '1024_r0_s1', '1024_r128_s1', '16384_r0_s0', '16384_r128_s0']
LABELS = ['1K · r0 · seed 0', '1K · r128 · seed 0', '1K · r0 · seed 1', '1K · r128 · seed 1', '16K · r0 · seed 0', '16K · r128 · seed 0']
NAVY, TEAL, GOLD, GRAY = '#244765', '#168577', '#BD8034', '#A0ADB8'


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#BCC6CD', 'axes.labelcolor': '#253C4B',
        'text.color': '#253C4B', 'xtick.color': '#455B69', 'ytick.color': '#455B69',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white',
    })


def save(fig, out, name, rows):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'{name}.{ext}', dpi=220, bbox_inches='tight')
        if ext == 'svg':
            svg = out/f'{name}.svg'
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                           encoding='utf-8', newline='\n')
    with (out/f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)


def forest(out, name, title, xlabel, labels, estimates, intervals, note):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.subplots_adjust(left=.25, right=.96, top=.84, bottom=.23)
    x = np.asarray(estimates); bounds = np.asarray(intervals)
    y = np.arange(len(labels))
    ax.axvline(0, color=GRAY, linewidth=1, linestyle='--')
    ax.errorbar(x, y, xerr=[x-bounds[:,0],bounds[:,1]-x], fmt='o', color=TEAL, capsize=4, markersize=6)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle(title, x=.25, y=.96, ha='left', fontsize=14, fontweight='bold')
    fig.text(.25,.04,note,fontsize=9,va='bottom')
    rows=[{'setting':label,'estimate':estimate,'ci95_low':interval[0],'ci95_high':interval[1]} for label,estimate,interval in zip(labels,estimates,intervals)]
    save(fig,out,name,rows)


def generate(report, out):
    out.mkdir(parents=True, exist_ok=True)
    style()
    quality=[report['quality']['axiom_quality_ctx'+s]['k4v4']['delta_pp'] for s in ORDER]
    forest(out,'acceptance_delta','K4V4 acceptance over 256-token continuations',
           'Acceptance change vs. BF16 (percentage points)', LABELS,
           [x['mean'] for x in quality],[x['ci95'] for x in quality],
           'Qwen2.5-3B / 1.5B · 24 paired prompts per setting\n95% paired prompt-bootstrap intervals; zero-crossing intervals do not establish equivalence.')
    residual=list(report['paired_residual128_minus0_k4v4'].values())
    forest(out,'residual_delta','Effect of retaining 128 BF16 key tokens',
           'K4V4 acceptance: residual 128 − residual 0 (percentage points)',
           ['1K · seed 0','1K · seed 1','16K · seed 0'],
           [x['delta_pp'] for x in residual],[x['ci95'] for x in residual],
           'Qwen2.5-3B / 1.5B · 24 paired prompts per setting\n95% paired prompt-bootstrap intervals; no established advantage for residual 128.')
    prefix=[report['prefix']['axiom_quality_ctx'+s]['k4v4'] for s in ORDER]
    forest(out,'accepted_prefix_delta','Accepted draft prefixes remain close to BF16',
           'Accepted prefix change vs. BF16 (tokens per round)', LABELS,
           [x['prefix_delta'] for x in prefix],[x['prefix_delta_ci95'] for x in prefix],
           'Equal-weight prompt means · 95% paired prompt-bootstrap intervals\nCorrection/bonus tokens excluded; final proposal rounds can be shorter than four tokens.')

    fig, axes=plt.subplots(1,3,figsize=(14.5,5.1))
    fig.subplots_adjust(left=.06,right=.99,bottom=.24,top=.78,wspace=.34)
    ctxs=['1024','16384']; configs=[('Target only',None,GRAY),('BF16 spec.',"('bf16', 128)",NAVY),('Packed r0',"('packed', 0)",TEAL),('Packed r128',"('packed', 128)",GOLD)]
    rows=[]
    for i,ctx in enumerate(ctxs):
        g=report['packed'][ctx]
        for j,(label,key,color) in enumerate(configs):
            if key:
                v=g[key];latency=v['mean_s'];peak=v['peak_GiB'];saving=v['draft_cache_saving_pct']['mean'];matches=v['matches']
            else:
                latency=g['target_only_mean_s'];peak=g['target_only_peak_GiB'];saving=None;matches='reference'
            rows.append({'context':int(ctx),'configuration':label,'mean_latency_s':latency,'peak_allocated_GiB':peak,'draft_cache_saved_pct':saving,'batched_exact_sequences':matches,'scored_prompts':24 if ctx=='1024' else 12})
            for ax,value in [(axes[1],latency),(axes[2],peak)]:
                ax.bar(i+(j-1.5)*.19,value,width=.18,color=color,label=label if i==0 else None)
            if key and label.startswith('Packed'):
                delta=-.16 if j==2 else .16
                axes[0].bar(i+delta,saving,width=.29,color=color,label=label if i==0 else None)
                axes[0].text(i+delta,saving+1,f'{saving:.1f}%',ha='center',fontsize=9)
    for ax,title,ylabel in zip(axes,['Persistent draft cache','Whole-request latency','Peak GPU allocation'],['Draft bytes saved vs. BF16 (%)','Mean seconds (including prefill)','Mean allocated GiB']):
        ax.set_title(title,loc='left');ax.set_ylabel(ylabel)
        ax.set_xticks([0,1],['1K · n=24','16K · n=12']);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[0].set_ylim(0,85)
    handles, labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.9),ncol=4,frameon=False)
    fig.suptitle('Packed draft-cache storage and prototype cost',x=.06,y=.99,ha='left',fontsize=15,fontweight='bold')
    fig.text(.06,.055,'Qwen2.5-3B / 1.5B · batch 1 · 256 generated tokens · warmups excluded\nBatched output mismatches remain unresolved: these are prototype costs, not validated exact-output serving speedups.',fontsize=9)
    save(fig,out,'storage_runtime',rows)

    fig,ax=plt.subplots(figsize=(10,4.8));fig.subplots_adjust(left=.22,right=.98,top=.77,bottom=.2)
    colors=[TEAL,NAVY,'#5785A1',GOLD,'#D6B782'];categories=['All proposed accepted','Reject 1st','Reject 2nd','Reject 3rd','Reject 4th']
    values=np.array([x['mean_prompt_rejection_distribution'] for x in prefix])*100
    left=np.zeros(6)
    for col,(label,color) in enumerate(zip(categories,colors)):
        ax.barh(np.arange(6),values[:,col],left=left,color=color,label=label);left+=values[:,col]
    ax.set_yticks(np.arange(6),LABELS);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Mean per-prompt share of rounds (%)')
    fig.suptitle('Where K4V4 draft proposals are rejected',x=.22,y=.98,ha='left',fontsize=14,fontweight='bold')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.57,.91),ncol=3,frameon=False,fontsize=9)
    fig.text(.22,.035,'Descriptive frequencies; rounds within a prompt are correlated.\n“All proposed accepted” includes truncated final rounds.',fontsize=9)
    rows=[dict(setting=label,**{c:float(v) for c,v in zip(categories,row)}) for label,row in zip(LABELS,values)]
    save(fig,out,'rejection_positions',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=BUNDLE)
    p.add_argument('--out',type=Path,default=Path('outputs/axiom-review-plots'))
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='axiom-plot-metrics-') as temp:
        metrics=rebuild_metrics(Path(temp)/'metrics',args.bundle)
        report=analyze(metrics)
        reference=json.loads((args.bundle/'validated-summary.json').read_text())
        compare({k:report[k] for k in reference},reference)
        compare(report['prefix'],json.loads((args.bundle/'prefix-summary.json').read_text())['results'])
    generate(report,args.out)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Verified raw outputs and reproduced frozen summaries. Wrote five figures (PDF, SVG, PNG), five CSV tables and summary.json to {args.out}')


if __name__=='__main__':
    main()
