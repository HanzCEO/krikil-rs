use crate::math::{
    apply_rope, dot, gelu, gemm_transb, gemv, rms_norm, rms_norm_inplace, silu, softmax_inplace,
    softplus,
};
use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

pub const HIDDEN_SIZE: usize = 608;
pub const VOCAB_SIZE: usize = 2048;
pub const INTERMEDIATE_SIZE: usize = 2432;
pub const NUM_LAYERS: usize = 8;
pub const NUM_ATTN_HEADS: usize = 8;
pub const ATTN_HEAD_DIM: usize = 76; // 608 / 8
pub const ROPE_THETA: f32 = 10000.0;
pub const RMS_NORM_EPS: f32 = 1e-6;

// Mamba-2 constants
pub const MAMBA_D_STATE: usize = 128;
pub const MAMBA_D_CONV: usize = 4;
pub const MAMBA_EXPAND: usize = 2;
pub const MAMBA_HEAD_DIM: usize = 64;
pub const MAMBA_D_INNER: usize = HIDDEN_SIZE * MAMBA_EXPAND; // 1216
pub const MAMBA_NUM_HEADS: usize = MAMBA_D_INNER / MAMBA_HEAD_DIM; // 19
pub const MAMBA_D_IN_PROJ: usize = 2 * MAMBA_D_INNER + 2 * MAMBA_D_STATE + MAMBA_NUM_HEADS; // 2707
pub const MAMBA_D_XBC: usize = MAMBA_D_INNER + 2 * MAMBA_D_STATE; // 1472
pub const MAMBA_NORM_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Deserialize)]
pub struct PebbleConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_block_pattern")]
    pub block_pattern: String,
}

fn default_vocab_size() -> usize {
    VOCAB_SIZE
}
fn default_hidden_size() -> usize {
    HIDDEN_SIZE
}
fn default_num_layers() -> usize {
    NUM_LAYERS
}
fn default_num_heads() -> usize {
    NUM_ATTN_HEADS
}
fn default_intermediate_size() -> usize {
    INTERMEDIATE_SIZE
}
fn default_max_pos() -> usize {
    2048
}
fn default_block_pattern() -> String {
    "mmma|mmma".to_string()
}

impl Default for PebbleConfig {
    fn default() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            hidden_size: HIDDEN_SIZE,
            num_hidden_layers: NUM_LAYERS,
            num_attention_heads: NUM_ATTN_HEADS,
            intermediate_size: INTERMEDIATE_SIZE,
            max_position_embeddings: 2048,
            block_pattern: "mmma|mmma".to_string(),
        }
    }
}

pub struct MambaBlockWeights {
    pub ln_w: Vec<f32>,       // [608]
    pub in_proj_w: Vec<f32>,  // [2707, 608]
    pub conv1d_w: Vec<f32>,   // [1472, 4]
    pub conv1d_b: Vec<f32>,   // [1472]
    pub dt_bias: Vec<f32>,    // [19]
    pub a_neg_exp: Vec<f32>,  // precomputed: -exp(A_log) [19]
    pub d: Vec<f32>,          // [19]
    pub norm_w: Vec<f32>,     // [1216]
    pub out_proj_w: Vec<f32>, // [608, 1216]
}

pub struct AttentionBlockWeights {
    pub ln1_w: Vec<f32>,  // [608]
    pub wqkv_w: Vec<f32>, // [1824, 608]
    pub wo_w: Vec<f32>,   // [608, 608]
    pub ln2_w: Vec<f32>,  // [608]
    pub fc1_w: Vec<f32>,  // [2432, 608]
    pub fc2_w: Vec<f32>,  // [608, 2432]
}

pub enum BlockWeights {
    Mamba(MambaBlockWeights),
    Attention(AttentionBlockWeights),
}

pub struct PebbleModel {
    pub config: PebbleConfig,
    pub wte: Vec<f32>, // [2048, 608]
    pub blocks: Vec<BlockWeights>,
    pub lnf_w: Vec<f32>, // [608]
}

#[derive(Clone)]
pub struct MambaLayerState {
    pub conv_state: Vec<f32>, // [1472, 4]
    pub ssm_state: Vec<f32>,  // [19, 64, 128]
}

impl MambaLayerState {
    pub fn new() -> Self {
        Self {
            conv_state: vec![0.0f32; MAMBA_D_XBC * MAMBA_D_CONV],
            ssm_state: vec![0.0f32; MAMBA_NUM_HEADS * MAMBA_HEAD_DIM * MAMBA_D_STATE],
        }
    }

    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.ssm_state.fill(0.0);
    }
}

#[derive(Clone)]
pub struct AttentionLayerState {
    pub k_cache: Vec<f32>, // [max_ctx, NUM_ATTN_HEADS * ATTN_HEAD_DIM]
    pub v_cache: Vec<f32>, // [max_ctx, NUM_ATTN_HEADS * ATTN_HEAD_DIM]
    pub current_len: usize,
}

impl AttentionLayerState {
    pub fn new(max_ctx: usize) -> Self {
        let dim = NUM_ATTN_HEADS * ATTN_HEAD_DIM;
        Self {
            k_cache: vec![0.0f32; max_ctx * dim],
            v_cache: vec![0.0f32; max_ctx * dim],
            current_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }
}

pub enum LayerState {
    Mamba(MambaLayerState),
    Attention(AttentionLayerState),
}

pub struct PebbleBuffers {
    pub h_buf: Vec<f32>,
    pub normed_buf: Vec<f32>,
    pub zxbcdt_buf: Vec<f32>,
    pub mamba_y_buf: Vec<f32>,
    pub mamba_out_buf: Vec<f32>,
    pub qkv_buf: Vec<f32>,
    pub attn_out_buf: Vec<f32>,
    pub mlp_h_buf: Vec<f32>,
    pub mlp_out_buf: Vec<f32>,
    pub logits_buf: Vec<f32>,
}

impl PebbleBuffers {
    pub fn new() -> Self {
        Self {
            h_buf: vec![0.0f32; HIDDEN_SIZE],
            normed_buf: vec![0.0f32; HIDDEN_SIZE],
            zxbcdt_buf: vec![0.0f32; MAMBA_D_IN_PROJ],
            mamba_y_buf: vec![0.0f32; MAMBA_D_INNER],
            mamba_out_buf: vec![0.0f32; HIDDEN_SIZE],
            qkv_buf: vec![0.0f32; 3 * HIDDEN_SIZE],
            attn_out_buf: vec![0.0f32; HIDDEN_SIZE],
            mlp_h_buf: vec![0.0f32; INTERMEDIATE_SIZE],
            mlp_out_buf: vec![0.0f32; HIDDEN_SIZE],
            logits_buf: vec![0.0f32; VOCAB_SIZE],
        }
    }
}

pub struct PebbleState {
    pub layers: Vec<LayerState>,
    pub buffers: PebbleBuffers,
    pub context_limit: usize,
    pub current_pos: usize,
}

impl PebbleState {
    pub fn new(config: &PebbleConfig, max_ctx: usize) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            if i % 4 < 3 {
                layers.push(LayerState::Mamba(MambaLayerState::new()));
            } else {
                layers.push(LayerState::Attention(AttentionLayerState::new(max_ctx)));
            }
        }
        Self {
            layers,
            buffers: PebbleBuffers::new(),
            context_limit: max_ctx,
            current_pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current_pos = 0;
        for layer in &mut self.layers {
            match layer {
                LayerState::Mamba(s) => s.reset(),
                LayerState::Attention(s) => s.reset(),
            }
        }
    }
}

impl PebbleModel {
    pub fn load_safetensors(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open model file: {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("Failed to parse safetensors from {}", path.display()))?;

        let get_tensor = |name: &str| -> Result<Vec<f32>> {
            let tensor = safetensors
                .tensor(name)
                .with_context(|| format!("Tensor '{}' not found in model", name))?;
            let data = tensor.data();
            let num_floats = data.len() / 4;
            let mut result = vec![0.0f32; num_floats];
            // Read floats in little-endian
            for i in 0..num_floats {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().unwrap();
                result[i] = f32::from_le_bytes(bytes);
            }
            Ok(result)
        };

        let wte = get_tensor("wte.weight")?;
        let lnf_w = get_tensor("lnf.weight")?;

        let mut blocks = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            if i % 4 < 3 {
                // Mamba block
                let ln_w = get_tensor(&format!("blocks.{}.ln.weight", i))?;
                let in_proj_w = get_tensor(&format!("blocks.{}.mixer.in_proj.weight", i))?;
                let conv1d_w = get_tensor(&format!("blocks.{}.mixer.conv1d.weight", i))?;
                let conv1d_b = get_tensor(&format!("blocks.{}.mixer.conv1d.bias", i))?;
                let dt_bias = get_tensor(&format!("blocks.{}.mixer.dt_bias", i))?;
                let a_log = get_tensor(&format!("blocks.{}.mixer.A_log", i))?;
                let d = get_tensor(&format!("blocks.{}.mixer.D", i))?;
                let norm_w = get_tensor(&format!("blocks.{}.mixer.norm.weight", i))?;
                let out_proj_w = get_tensor(&format!("blocks.{}.mixer.out_proj.weight", i))?;

                // Precompute A = -exp(A_log)
                let a_neg_exp = a_log.iter().map(|&v| -v.exp()).collect();

                blocks.push(BlockWeights::Mamba(MambaBlockWeights {
                    ln_w,
                    in_proj_w,
                    conv1d_w,
                    conv1d_b,
                    dt_bias,
                    a_neg_exp,
                    d,
                    norm_w,
                    out_proj_w,
                }));
            } else {
                // Attention block
                let ln1_w = get_tensor(&format!("blocks.{}.ln1.weight", i))?;
                let wqkv_w = get_tensor(&format!("blocks.{}.wqkv.weight", i))?;
                let wo_w = get_tensor(&format!("blocks.{}.wo.weight", i))?;
                let ln2_w = get_tensor(&format!("blocks.{}.ln2.weight", i))?;
                let fc1_w = get_tensor(&format!("blocks.{}.fc1.weight", i))?;
                let fc2_w = get_tensor(&format!("blocks.{}.fc2.weight", i))?;

                blocks.push(BlockWeights::Attention(AttentionBlockWeights {
                    ln1_w,
                    wqkv_w,
                    wo_w,
                    ln2_w,
                    fc1_w,
                    fc2_w,
                }));
            }
        }

        Ok(Self {
            config: PebbleConfig::default(),
            wte,
            blocks,
            lnf_w,
        })
    }

    /// Prefill sequence of tokens and return the logits of the last token.
    pub fn forward_prefill(&self, tokens: &[u32], state: &mut PebbleState) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            bail!("Cannot prefill empty sequence");
        }
        let t = tokens.len();
        if state.current_pos + t > state.context_limit {
            bail!(
                "Sequence length {} exceeds context limit {}",
                state.current_pos + t,
                state.context_limit
            );
        }

        // Initialize x: [T, HIDDEN_SIZE] from embedding table
        let mut x = vec![0.0f32; t * HIDDEN_SIZE];
        for (i, &tok_id) in tokens.iter().enumerate() {
            let tok_idx = tok_id as usize;
            if tok_idx >= VOCAB_SIZE {
                bail!("Token ID {} exceeds vocab size {}", tok_idx, VOCAB_SIZE);
            }
            let emb = &self.wte[tok_idx * HIDDEN_SIZE..(tok_idx + 1) * HIDDEN_SIZE];
            x[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE].copy_from_slice(emb);
        }

        // Pass through each block
        for (block_idx, block) in self.blocks.iter().enumerate() {
            match block {
                BlockWeights::Mamba(m) => {
                    let LayerState::Mamba(m_state) = &mut state.layers[block_idx] else {
                        panic!("Layer state mismatch");
                    };
                    self.prefill_mamba(m, m_state, &mut x, t);
                }
                BlockWeights::Attention(a) => {
                    let LayerState::Attention(a_state) = &mut state.layers[block_idx] else {
                        panic!("Layer state mismatch");
                    };
                    self.prefill_attention(a, a_state, &mut x, t, state.current_pos);
                }
            }
        }

        state.current_pos += t;

        // Final RMSNorm and LM head on the last token x[t - 1]
        let last_x = &x[(t - 1) * HIDDEN_SIZE..t * HIDDEN_SIZE];
        let mut normed = vec![0.0f32; HIDDEN_SIZE];
        rms_norm(last_x, &self.lnf_w, RMS_NORM_EPS, &mut normed);

        let mut logits = vec![0.0f32; VOCAB_SIZE];
        gemv(&self.wte, &normed, &mut logits, VOCAB_SIZE, HIDDEN_SIZE);
        Ok(logits)
    }

    fn prefill_mamba(
        &self,
        m: &MambaBlockWeights,
        s: &mut MambaLayerState,
        x: &mut [f32],
        t: usize,
    ) {
        // Step 1: RMSNorm on each token vector of x
        let mut h = vec![0.0f32; t * HIDDEN_SIZE];
        for i in 0..t {
            let src = &x[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE];
            let dst = &mut h[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE];
            rms_norm(src, &m.ln_w, RMS_NORM_EPS, dst);
        }

        // Step 2: in_proj: [T, 608] x [2707, 608]^T => [T, 2707]
        let mut zxbcdt = vec![0.0f32; t * MAMBA_D_IN_PROJ];
        gemm_transb(&h, &m.in_proj_w, &mut zxbcdt, t, HIDDEN_SIZE, MAMBA_D_IN_PROJ);

        // Step 3: Sequential causal conv1d and SSM scan across time steps
        let mut y_seq = vec![0.0f32; t * MAMBA_D_INNER];

        for step in 0..t {
            let row = &zxbcdt[step * MAMBA_D_IN_PROJ..(step + 1) * MAMBA_D_IN_PROJ];
            let z = &row[..MAMBA_D_INNER];
            let xbc = &row[MAMBA_D_INNER..MAMBA_D_INNER + MAMBA_D_XBC];
            let dt = &row[MAMBA_D_INNER + MAMBA_D_XBC..];

            // Conv step: roll conv_state left by 1 and insert xbc
            let mut conv_out = vec![0.0f32; MAMBA_D_XBC];
            for c in 0..MAMBA_D_XBC {
                let cs = &mut s.conv_state[c * MAMBA_D_CONV..(c + 1) * MAMBA_D_CONV];
                cs[0] = cs[1];
                cs[1] = cs[2];
                cs[2] = cs[3];
                cs[3] = xbc[c];

                let cw = &m.conv1d_w[c * MAMBA_D_CONV..(c + 1) * MAMBA_D_CONV];
                let val = cs[0] * cw[0] + cs[1] * cw[1] + cs[2] * cw[2] + cs[3] * cw[3] + m.conv1d_b[c];
                conv_out[c] = silu(val);
            }

            let x_ssm = &conv_out[..MAMBA_D_INNER];
            let b_ssm = &conv_out[MAMBA_D_INNER..MAMBA_D_INNER + MAMBA_D_STATE];
            let c_ssm = &conv_out[MAMBA_D_INNER + MAMBA_D_STATE..];

            // SSM recurrent scan
            let mut y_step = vec![0.0f32; MAMBA_D_INNER];

            for head in 0..MAMBA_NUM_HEADS {
                let dt_val = softplus(dt[head] + m.dt_bias[head]);
                let da = (dt_val * m.a_neg_exp[head]).exp();

                let x_head = &x_ssm[head * MAMBA_HEAD_DIM..(head + 1) * MAMBA_HEAD_DIM];
                let y_head = &mut y_step[head * MAMBA_HEAD_DIM..(head + 1) * MAMBA_HEAD_DIM];
                let d_val = m.d[head];

                let state_head = &mut s.ssm_state
                    [head * MAMBA_HEAD_DIM * MAMBA_D_STATE..(head + 1) * MAMBA_HEAD_DIM * MAMBA_D_STATE];

                for p in 0..MAMBA_HEAD_DIM {
                    let x_p = x_head[p];
                    let x_dt = x_p * dt_val;
                    let st_row = &mut state_head[p * MAMBA_D_STATE..(p + 1) * MAMBA_D_STATE];

                    let mut sum_y = 0.0f32;
                    for n in 0..MAMBA_D_STATE {
                        let new_s = st_row[n] * da + b_ssm[n] * x_dt;
                        st_row[n] = new_s;
                        sum_y += new_s * c_ssm[n];
                    }
                    y_head[p] = sum_y + d_val * x_p;
                }
            }

            // Gating: y = y * silu(z)
            for i in 0..MAMBA_D_INNER {
                y_step[i] *= silu(z[i]);
            }

            // RMSNorm on y with eps=1e-5
            rms_norm_inplace(&mut y_step, &m.norm_w, MAMBA_NORM_EPS);

            y_seq[step * MAMBA_D_INNER..(step + 1) * MAMBA_D_INNER].copy_from_slice(&y_step);
        }

        // Step 4: out_proj: [T, 1216] x [608, 1216]^T => [T, 608]
        let mut out = vec![0.0f32; t * HIDDEN_SIZE];
        gemm_transb(&y_seq, &m.out_proj_w, &mut out, t, MAMBA_D_INNER, HIDDEN_SIZE);

        // Step 5: Residual add x = x + out
        for i in 0..t * HIDDEN_SIZE {
            x[i] += out[i];
        }
    }

    fn prefill_attention(
        &self,
        a: &AttentionBlockWeights,
        s: &mut AttentionLayerState,
        x: &mut [f32],
        t: usize,
        start_pos: usize,
    ) {
        // Step 1: RMSNorm ln1
        let mut h = vec![0.0f32; t * HIDDEN_SIZE];
        for i in 0..t {
            let src = &x[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE];
            let dst = &mut h[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE];
            rms_norm(src, &a.ln1_w, RMS_NORM_EPS, dst);
        }

        // Step 2: wqkv: [T, 608] x [1824, 608]^T => [T, 1824]
        let mut qkv = vec![0.0f32; t * 3 * HIDDEN_SIZE];
        gemm_transb(&h, &a.wqkv_w, &mut qkv, t, HIDDEN_SIZE, 3 * HIDDEN_SIZE);

        // Step 3: RoPE and KV Cache storage
        let mut q_seq = vec![0.0f32; t * HIDDEN_SIZE];

        for step in 0..t {
            let row = &qkv[step * 3 * HIDDEN_SIZE..(step + 1) * 3 * HIDDEN_SIZE];
            let mut q_cur = row[..HIDDEN_SIZE].to_vec();
            let mut k_cur = row[HIDDEN_SIZE..2 * HIDDEN_SIZE].to_vec();
            let v_cur = &row[2 * HIDDEN_SIZE..3 * HIDDEN_SIZE];

            let pos = start_pos + step;
            apply_rope(
                &mut q_cur,
                &mut k_cur,
                pos,
                NUM_ATTN_HEADS,
                ATTN_HEAD_DIM,
                ROPE_THETA,
            );

            // Store in KV cache
            let cache_offset = (s.current_len + step) * HIDDEN_SIZE;
            s.k_cache[cache_offset..cache_offset + HIDDEN_SIZE].copy_from_slice(&k_cur);
            s.v_cache[cache_offset..cache_offset + HIDDEN_SIZE].copy_from_slice(v_cur);

            q_seq[step * HIDDEN_SIZE..(step + 1) * HIDDEN_SIZE].copy_from_slice(&q_cur);
        }

        let total_cached = s.current_len + t;

        // Step 4: Causal self-attention
        let scale = 1.0 / (ATTN_HEAD_DIM as f32).sqrt();
        let mut attn_out = vec![0.0f32; t * HIDDEN_SIZE];

        for step in 0..t {
            let cur_pos = s.current_len + step;
            let q_step = &q_seq[step * HIDDEN_SIZE..(step + 1) * HIDDEN_SIZE];
            let out_step = &mut attn_out[step * HIDDEN_SIZE..(step + 1) * HIDDEN_SIZE];

            for h in 0..NUM_ATTN_HEADS {
                let q_h = &q_step[h * ATTN_HEAD_DIM..(h + 1) * ATTN_HEAD_DIM];

                // Compute attention scores with cached keys for j = 0..=cur_pos
                let num_keys = cur_pos + 1;
                let mut scores = vec![0.0f32; num_keys];
                for j in 0..num_keys {
                    let k_j = &s.k_cache[j * HIDDEN_SIZE + h * ATTN_HEAD_DIM
                        ..j * HIDDEN_SIZE + (h + 1) * ATTN_HEAD_DIM];
                    scores[j] = dot(q_h, k_j) * scale;
                }

                softmax_inplace(&mut scores);

                // Weighted sum of values
                let out_h = &mut out_step[h * ATTN_HEAD_DIM..(h + 1) * ATTN_HEAD_DIM];
                for (j, &w) in scores.iter().enumerate() {
                    let v_j = &s.v_cache[j * HIDDEN_SIZE + h * ATTN_HEAD_DIM
                        ..j * HIDDEN_SIZE + (h + 1) * ATTN_HEAD_DIM];
                    for d in 0..ATTN_HEAD_DIM {
                        out_h[d] += w * v_j[d];
                    }
                }
            }
        }

        s.current_len = total_cached;

        // Step 5: wo: [T, 608] x [608, 608]^T => [T, 608]
        let mut wo_out = vec![0.0f32; t * HIDDEN_SIZE];
        gemm_transb(&attn_out, &a.wo_w, &mut wo_out, t, HIDDEN_SIZE, HIDDEN_SIZE);

        // Residual 1
        for i in 0..t * HIDDEN_SIZE {
            x[i] += wo_out[i];
        }

        // Step 6: MLP: ln2 -> fc1 -> gelu -> fc2 -> residual
        let mut ln2_out = vec![0.0f32; t * HIDDEN_SIZE];
        for i in 0..t {
            let src = &x[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE];
            let dst = &mut ln2_out[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE];
            rms_norm(src, &a.ln2_w, RMS_NORM_EPS, dst);
        }

        let mut fc1_out = vec![0.0f32; t * INTERMEDIATE_SIZE];
        gemm_transb(&ln2_out, &a.fc1_w, &mut fc1_out, t, HIDDEN_SIZE, INTERMEDIATE_SIZE);

        for val in fc1_out.iter_mut() {
            *val = gelu(*val);
        }

        let mut fc2_out = vec![0.0f32; t * HIDDEN_SIZE];
        gemm_transb(&fc1_out, &a.fc2_w, &mut fc2_out, t, INTERMEDIATE_SIZE, HIDDEN_SIZE);

        // Residual 2
        for i in 0..t * HIDDEN_SIZE {
            x[i] += fc2_out[i];
        }
    }

    /// Single autoregressive decoding step for a new token.
    pub fn forward_step(&self, token: u32, state: &mut PebbleState) -> Result<Vec<f32>> {
        let tok_idx = token as usize;
        if tok_idx >= VOCAB_SIZE {
            bail!("Token ID {} exceeds vocab size {}", tok_idx, VOCAB_SIZE);
        }
        if state.current_pos >= state.context_limit {
            bail!("Context limit {} reached", state.context_limit);
        }

        let pos = state.current_pos;

        // Initialize h from embedding table
        state.buffers.h_buf.copy_from_slice(&self.wte[tok_idx * HIDDEN_SIZE..(tok_idx + 1) * HIDDEN_SIZE]);

        for (block_idx, block) in self.blocks.iter().enumerate() {
            match block {
                BlockWeights::Mamba(m) => {
                    let LayerState::Mamba(m_state) = &mut state.layers[block_idx] else {
                        panic!("Layer state mismatch");
                    };
                    self.step_mamba(m, m_state, &mut state.buffers);
                }
                BlockWeights::Attention(a) => {
                    let LayerState::Attention(a_state) = &mut state.layers[block_idx] else {
                        panic!("Layer state mismatch");
                    };
                    self.step_attention(a, a_state, &mut state.buffers, pos);
                }
            }
        }

        state.current_pos += 1;

        // Final RMSNorm and LM head
        rms_norm(&state.buffers.h_buf, &self.lnf_w, RMS_NORM_EPS, &mut state.buffers.normed_buf);
        gemv(&self.wte, &state.buffers.normed_buf, &mut state.buffers.logits_buf, VOCAB_SIZE, HIDDEN_SIZE);

        Ok(state.buffers.logits_buf.clone())
    }

    fn step_mamba(
        &self,
        m: &MambaBlockWeights,
        s: &mut MambaLayerState,
        buf: &mut PebbleBuffers,
    ) {
        // 1. RMSNorm
        rms_norm(&buf.h_buf, &m.ln_w, RMS_NORM_EPS, &mut buf.normed_buf);

        // 2. in_proj GEMV: [2707, 608] x [608] => [2707]
        gemv(&m.in_proj_w, &buf.normed_buf, &mut buf.zxbcdt_buf, MAMBA_D_IN_PROJ, HIDDEN_SIZE);

        let z = &buf.zxbcdt_buf[..MAMBA_D_INNER];
        let xbc = &buf.zxbcdt_buf[MAMBA_D_INNER..MAMBA_D_INNER + MAMBA_D_XBC];
        let dt = &buf.zxbcdt_buf[MAMBA_D_INNER + MAMBA_D_XBC..];

        // 3. Conv step: roll conv_state and dot with weights
        let mut conv_out = vec![0.0f32; MAMBA_D_XBC];
        for c in 0..MAMBA_D_XBC {
            let cs = &mut s.conv_state[c * MAMBA_D_CONV..(c + 1) * MAMBA_D_CONV];
            cs[0] = cs[1];
            cs[1] = cs[2];
            cs[2] = cs[3];
            cs[3] = xbc[c];

            let cw = &m.conv1d_w[c * MAMBA_D_CONV..(c + 1) * MAMBA_D_CONV];
            let val = cs[0] * cw[0] + cs[1] * cw[1] + cs[2] * cw[2] + cs[3] * cw[3] + m.conv1d_b[c];
            conv_out[c] = silu(val);
        }

        let x_ssm = &conv_out[..MAMBA_D_INNER];
        let b_ssm = &conv_out[MAMBA_D_INNER..MAMBA_D_INNER + MAMBA_D_STATE];
        let c_ssm = &conv_out[MAMBA_D_INNER + MAMBA_D_STATE..];

        // 4. SSM step
        for head in 0..MAMBA_NUM_HEADS {
            let dt_val = softplus(dt[head] + m.dt_bias[head]);
            let da = (dt_val * m.a_neg_exp[head]).exp();

            let x_head = &x_ssm[head * MAMBA_HEAD_DIM..(head + 1) * MAMBA_HEAD_DIM];
            let y_head = &mut buf.mamba_y_buf[head * MAMBA_HEAD_DIM..(head + 1) * MAMBA_HEAD_DIM];
            let d_val = m.d[head];

            let state_head = &mut s.ssm_state
                [head * MAMBA_HEAD_DIM * MAMBA_D_STATE..(head + 1) * MAMBA_HEAD_DIM * MAMBA_D_STATE];

            for p in 0..MAMBA_HEAD_DIM {
                let x_p = x_head[p];
                let x_dt = x_p * dt_val;
                let st_row = &mut state_head[p * MAMBA_D_STATE..(p + 1) * MAMBA_D_STATE];

                let mut sum_y = 0.0f32;
                for n in 0..MAMBA_D_STATE {
                    let new_s = st_row[n] * da + b_ssm[n] * x_dt;
                    st_row[n] = new_s;
                    sum_y += new_s * c_ssm[n];
                }
                y_head[p] = sum_y + d_val * x_p;
            }
        }

        // Gating: y = y * silu(z)
        for i in 0..MAMBA_D_INNER {
            buf.mamba_y_buf[i] *= silu(z[i]);
        }

        // RMSNorm on y with eps=1e-5
        rms_norm_inplace(&mut buf.mamba_y_buf, &m.norm_w, MAMBA_NORM_EPS);

        // out_proj: [608, 1216] x [1216] => [608]
        gemv(&m.out_proj_w, &buf.mamba_y_buf, &mut buf.mamba_out_buf, HIDDEN_SIZE, MAMBA_D_INNER);

        // Residual
        for i in 0..HIDDEN_SIZE {
            buf.h_buf[i] += buf.mamba_out_buf[i];
        }
    }

    fn step_attention(
        &self,
        a: &AttentionBlockWeights,
        s: &mut AttentionLayerState,
        buf: &mut PebbleBuffers,
        pos: usize,
    ) {
        // 1. RMSNorm ln1
        rms_norm(&buf.h_buf, &a.ln1_w, RMS_NORM_EPS, &mut buf.normed_buf);

        // 2. wqkv GEMV: [1824, 608] x [608] => [1824]
        gemv(&a.wqkv_w, &buf.normed_buf, &mut buf.qkv_buf, 3 * HIDDEN_SIZE, HIDDEN_SIZE);

        let mut q_cur = buf.qkv_buf[..HIDDEN_SIZE].to_vec();
        let mut k_cur = buf.qkv_buf[HIDDEN_SIZE..2 * HIDDEN_SIZE].to_vec();
        let v_cur = &buf.qkv_buf[2 * HIDDEN_SIZE..3 * HIDDEN_SIZE];

        apply_rope(
            &mut q_cur,
            &mut k_cur,
            pos,
            NUM_ATTN_HEADS,
            ATTN_HEAD_DIM,
            ROPE_THETA,
        );

        let cache_offset = s.current_len * HIDDEN_SIZE;
        s.k_cache[cache_offset..cache_offset + HIDDEN_SIZE].copy_from_slice(&k_cur);
        s.v_cache[cache_offset..cache_offset + HIDDEN_SIZE].copy_from_slice(v_cur);
        s.current_len += 1;

        // 3. Self-attention over all cached keys
        let scale = 1.0 / (ATTN_HEAD_DIM as f32).sqrt();
        buf.attn_out_buf.fill(0.0);

        for h in 0..NUM_ATTN_HEADS {
            let q_h = &q_cur[h * ATTN_HEAD_DIM..(h + 1) * ATTN_HEAD_DIM];

            let num_keys = s.current_len;
            let mut scores = vec![0.0f32; num_keys];
            for j in 0..num_keys {
                let k_j = &s.k_cache[j * HIDDEN_SIZE + h * ATTN_HEAD_DIM
                    ..j * HIDDEN_SIZE + (h + 1) * ATTN_HEAD_DIM];
                scores[j] = dot(q_h, k_j) * scale;
            }

            softmax_inplace(&mut scores);

            let out_h = &mut buf.attn_out_buf[h * ATTN_HEAD_DIM..(h + 1) * ATTN_HEAD_DIM];
            for (j, &w) in scores.iter().enumerate() {
                let v_j = &s.v_cache[j * HIDDEN_SIZE + h * ATTN_HEAD_DIM
                    ..j * HIDDEN_SIZE + (h + 1) * ATTN_HEAD_DIM];
                for d in 0..ATTN_HEAD_DIM {
                    out_h[d] += w * v_j[d];
                }
            }
        }

        // 4. wo GEMV: [608, 608] x [608] => [608]
        let mut wo_out = vec![0.0f32; HIDDEN_SIZE];
        gemv(&a.wo_w, &buf.attn_out_buf, &mut wo_out, HIDDEN_SIZE, HIDDEN_SIZE);

        for i in 0..HIDDEN_SIZE {
            buf.h_buf[i] += wo_out[i];
        }

        // 5. MLP
        rms_norm(&buf.h_buf, &a.ln2_w, RMS_NORM_EPS, &mut buf.normed_buf);
        gemv(&a.fc1_w, &buf.normed_buf, &mut buf.mlp_h_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE);

        for val in buf.mlp_h_buf.iter_mut() {
            *val = gelu(*val);
        }

        gemv(&a.fc2_w, &buf.mlp_h_buf, &mut buf.mlp_out_buf, HIDDEN_SIZE, INTERMEDIATE_SIZE);

        for i in 0..HIDDEN_SIZE {
            buf.h_buf[i] += buf.mlp_out_buf[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_pebble_forward_matches_pytorch_reference() {
        let model_path = Path::new("model.safetensors");
        let ref_path = Path::new("test_reference_logits.json");
        if !model_path.exists() || !ref_path.exists() {
            eprintln!("model.safetensors or test_reference_logits.json not found, skipping");
            return;
        }

        let ref_data: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(ref_path).unwrap()).unwrap();
        let tokens: Vec<u32> = ref_data["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected_logits: Vec<f32> = ref_data["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let model = PebbleModel::load_safetensors(model_path).expect("Failed to load model");
        let mut state = PebbleState::new(&model.config, 2048);

        let rust_logits = model
            .forward_prefill(&tokens, &mut state)
            .expect("Prefill failed");

        assert_eq!(rust_logits.len(), expected_logits.len());

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        for (i, (&r, &e)) in rust_logits.iter().zip(expected_logits.iter()).enumerate() {
            let diff = (r - e).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }

        println!(
            "Max logit difference between Rust and PyTorch: {} at index {} (Rust: {}, PyTorch: {})",
            max_diff, max_idx, rust_logits[max_idx], expected_logits[max_idx]
        );

        // Check top token matches
        let rust_top = rust_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let exp_top = expected_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(rust_top, exp_top, "Top predicted token must match");
        assert!(
            max_diff < 1e-4,
            "Max difference {} exceeds tolerance 1e-4",
            max_diff
        );
    }

    #[test]
    fn test_pebble_step_matches_pytorch_reference() {
        let model_path = Path::new("model.safetensors");
        let ref_path = Path::new("test_reference_step_logits.json");
        if !model_path.exists() || !ref_path.exists() {
            eprintln!("model.safetensors or test_reference_step_logits.json not found, skipping");
            return;
        }

        let ref_data: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(ref_path).unwrap()).unwrap();
        let prompt_tokens = vec![
            397, 264, 26, 1270, 311, 263, 1530, 1360, 284, 420, 82, 549, 31, 842, 373, 419, 26,
            221,
        ];
        let step_token: u32 = ref_data["step_token"].as_u64().unwrap() as u32;
        let expected_logits: Vec<f32> = ref_data["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let model = PebbleModel::load_safetensors(model_path).expect("Failed to load model");
        let mut state = PebbleState::new(&model.config, 2048);

        // 1. Prefill prompt
        let _ = model
            .forward_prefill(&prompt_tokens, &mut state)
            .expect("Prefill failed");

        // 2. Step forward on step_token
        let step_logits = model
            .forward_step(step_token, &mut state)
            .expect("Step failed");

        assert_eq!(step_logits.len(), expected_logits.len());

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        for (i, (&r, &e)) in step_logits.iter().zip(expected_logits.iter()).enumerate() {
            let diff = (r - e).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }

        println!(
            "Max step logit difference: {} at index {} (Rust: {}, PyTorch: {})",
            max_diff, max_idx, step_logits[max_idx], expected_logits[max_idx]
        );

        let rust_top = step_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let exp_top = expected_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert_eq!(rust_top, exp_top, "Top predicted token for step must match");
        assert!(
            max_diff < 1e-4,
            "Max step difference {} exceeds tolerance 1e-4",
            max_diff
        );
    }
}
