# Technical Specification: Absorptive KV-Shared Speculative Decoding (AKV-SD)

## 1. Executive Summary
Traditional Speculative Decoding accelerates Large Language Model (LLM) inference by using a smaller Draft model to generate candidate tokens, which a larger Target model then verifies in parallel. However, this requires maintaining two separate Key-Value (KV) caches, leading to high memory overhead. 

**Absorptive KV-Shared Speculative Decoding (AKV-SD)** is a zero-training, plug-and-play architecture that eliminates the Draft model's KV cache. It enables the Draft model to directly read the Target model's KV cache during the speculative generation phase. Instead of applying runtime Multi-Layer Perceptrons (MLPs) to map the caches—which introduces latency—AKV-SD utilizes closed-form multivariate linear regression to learn optimal mappings offline. These mappings are mathematically "absorbed" directly into the Draft model's Query ($W_Q$) and Output ($W_O$) projection matrices.

## 2. High-Level Architecture
During inference, the Draft model operates entirely statelessly regarding the KV cache. 
1. The Target model generates and stores Keys ($K_{target}$) and Values ($V_{target}$) in its cache.
2. The Draft model generates its hidden state for the current token.
3. Instead of using its own $W_Q$ and $W_O$ weights, the Draft model uses swapped weights ($W_{Q\_new}$ and $W_{O\_new}$).
4. $W_{Q\_new}$ projects the Draft's residual stream directly into the Target's Key space, generating "Target-native" Queries.
5. The attention dot product is performed using these Queries and $K_{target}$.
6. The resulting attention weights are applied to $V_{target}$.
7. $W_{O\_new}$ acts as a bottleneck projector, taking the high-dimensional Target Values and compressing them directly back into the Draft model's lower-dimensional residual stream.

## 3. Mathematical Formulation

### 3.1. Key Mapping (Query Absorption)
Let the Draft hidden state be $h_{draft}$. We learn a linear transform $W_{K\_transform}$ such that $K_{draft} \approx K_{target} W_{K\_transform}$.
Instead of mapping the Keys at runtime, we apply the transposition to the Draft's Query projection.

$$Q_{draft\_native} = h_{draft} W_{Q\_original}$$
$$Q_{mapped} = Q_{draft\_native} W_{K\_transform}^T$$

**Absorption:** We pre-compute $W_{Q\_new} = W_{Q\_original} W_{K\_transform}^T$.
During inference: $Q_{mapped} = h_{draft} W_{Q\_new}$.

### 3.2. Value Mapping (Output Absorption)
We avoid mapping $V_{target}$ to $V_{draft}$ directly. Instead, we map the attention-weighted output.
Let $\tilde{H}_{target} = A_{draft} \cdot V_{target}$ be the Target Values weighted by the Draft's attention scores.
Let $Y_{draft} = (A_{draft} \cdot V_{draft}) \cdot W_{O\_original}$ be the true Draft attention block output.

We seek $W_{O\_new}$ such that:
$$\tilde{H}_{target} W_{O\_new} \approx Y_{draft}$$

**Absorption:** $W_{O\_new}$ is solved offline via Ridge Regression and permanently replaces $W_{O\_original}$ in the Draft model.

## 4. Low-Level Implementation

### Phase 1: Calibration & Weight Solve (Offline)
This step occurs once before the inference server starts. It requires a small calibration dataset (e.g., 500 sequences from WikiText).

1.  **Forward Pass:** Run the calibration data through both the frozen Target and Draft models.
2.  **Harvest Tensors:** For each designated layer pair (Target Layer $N$, Draft Layer $M$), intercept and save:
    * $K_{target}$, $K_{draft}$
    * $V_{target}$
    * $A_{draft}$ (Draft Attention Softmax output)
    * $Y_{draft}$ (Draft final $W_O$ output)
3.  **Solve $W_{Q\_new}$:**
    * Use Ordinary Least Squares (OLS) to solve $K_{target} W_{K\_transform} \approx K_{draft}$.
    * Compute $W_{Q\_new} = W_{Q\_original} W_{K\_transform}^T$.
4.  **Solve $W_{O\_new}$:**
    * Compute $\tilde{H}_{target} = A_{draft} \cdot V_{target}$.
    * Use Ridge Regression to solve $\tilde{H}_{target} W_{O\_new} \approx Y_{draft}$.
    * *Note: Ridge Regression is preferred over OLS here to prevent over-indexing on semantic outliers in the calibration set.*

### Phase 2: Architectural Dilation
To account for dimensionality mismatch (e.g., Target has $d=4096, h=32$; Draft has $d=2048, h=16$), the Draft model's architecture is "dilated" at the attention layer.

* **Original Draft Shapes:**
    * `W_Q`: `[2048, 16 * 128]` 
    * `W_O`: `[16 * 128, 2048]`
* **New Draft Shapes (Dilated):**
    * `W_Q_new`: `[2048, 32 * 128]` $\leftarrow$ Draft residual expands to create 32 queries.
    * `W_O_new`: `[32 * 128, 2048]` $\leftarrow$ 32 Target Value heads compress back to Draft residual.

### Phase 3: Runtime Inference Loop
The inference engine is modified to pass the Target cache by reference to the Draft model.

```python
def draft_model_forward(hidden_states, target_kv_cache, layer_map):
    for draft_layer_idx, layer in enumerate(draft_model.layers):
        target_layer_idx = layer_map[draft_layer_idx]
        
        # 1. Fetch Target Cache for corresponding layer
        K_target, V_target = target_kv_cache[target_layer_idx]
        
        # 2. Generate Dilated Queries
        # hidden_states: [B, Seq, d_draft] -> Q_dilated: [B, Seq, target_heads, head_dim]
        Q_dilated = layer.q_proj_new(hidden_states)
        
        # 3. Apply RoPE (Crucial)
        # Apply the TARGET model's RoPE frequencies to Q_dilated, because 
        # K_target in the cache has already been rotated by the Target's RoPE.
        Q_dilated = apply_target_rope(Q_dilated, position_ids)
        
        # 4. Standard Attention with Target Cache
        attn_weights = softmax((Q_dilated @ K_target.T) / sqrt(head_dim))
        
        # 5. Apply to Target Values
        # H_target shape: [B, Seq, target_heads * head_dim]
        H_target = attn_weights @ V_target 
        
        # 6. Bottleneck Output Projection
        # H_target: [B, Seq, d_target] -> hidden_states: [B, Seq, d_draft]
        attn_output = layer.o_proj_new(H_target)
        
        # 7. Residual connection & MLP
        hidden_states = hidden_states + attn_output
        hidden_states = layer.mlp(layer.ln(hidden_states)) + hidden_states
        
    return hidden_states
```

## 5. Critical Constraints & Optimizations

1.  **RoPE Synchronization:** The positional embeddings are the most sensitive failure point. If the Target model applies RoPE to its Keys before caching, the Draft model *must* apply the Target model's specific RoPE base frequencies to $Q_{dilated}$ before computing the dot product. Do not use the Draft model's native RoPE frequencies.
2.  **Layer Topology Mapping:** Do not assume a strict linear mapping (e.g., Draft Layer 1 $\rightarrow$ Target Layer 2). Before solving the weights, perform a cosine similarity search across all Target caches to find which Target layer's $V_{target}$ requires the least aggressive $W_O$ transformation to yield the Draft's $Y_{draft}$. 
3.  **Numerical Stability:** When solving for $W_{O\_new}$, apply `RMSNorm` scaling to $\tilde{H}_{target}$ if the variance of the Target Values drastically exceeds the variance expected by the Draft's residual stream.