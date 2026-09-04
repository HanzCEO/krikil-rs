use rayon::prelude::*;

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    // Exact erf-based GELU matching PyTorch default: 0.5 * x * (1 + erf(x / sqrt(2)))
    0.5 * x * (1.0 + libm_erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

// Highly accurate approximation of erf matching standard libm
fn libm_erf(x: f32) -> f32 {
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;
    let p = 0.3275911f32;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();

    sign * y
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());
    let dim = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / (dim as f32);
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * scale * weight[i];
    }
}

pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len());
    let dim = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / (dim as f32);
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..dim {
        x[i] = x[i] * scale * weight[i];
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2_fma(a, b) };
        }
    }

    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();
    let mut pa = a.as_ptr();
    let mut pb = b.as_ptr();

    for _ in 0..chunks {
        let va = _mm256_loadu_ps(pa);
        let vb = _mm256_loadu_ps(pb);
        acc = _mm256_fmadd_ps(va, vb, acc);
        pa = pa.add(8);
        pb = pb.add(8);
    }

    // Horizontal sum of acc
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(hi128, lo128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    let mut res = _mm_cvtss_f32(sum32);

    for i in 0..remainder {
        res += *pa.add(i) * *pb.add(i);
    }

    res
}

/// Matrix-vector multiplication: out = W * x
/// W shape: [rows, cols] (row-major)
/// x shape: [cols]
/// out shape: [rows]
pub fn gemv(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);

    if rows >= 256 {
        // Parallelize across rows for large matrices
        out.par_chunks_mut(64)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_r = chunk_idx * 64;
                for (i, val) in out_chunk.iter_mut().enumerate() {
                    let r = start_r + i;
                    let row_w = &w[r * cols..(r + 1) * cols];
                    *val = dot(row_w, x);
                }
            });
    } else {
        for r in 0..rows {
            let row_w = &w[r * cols..(r + 1) * cols];
            out[r] = dot(row_w, x);
        }
    }
}

/// Matrix-matrix multiplication: C = A * B
/// A shape: [M, K]
/// B shape: [K, N]
/// C shape: [M, N]
pub fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// Computes C = A * B^T
/// A shape: [M, K]
/// B shape: [N, K]
/// C shape: [M, N]
pub fn gemm_transb(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);

    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_val = x[0];
    for &v in x.iter().skip(1) {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, nh: usize, hd: usize, theta: f32) {
    assert_eq!(q.len(), nh * hd);
    assert_eq!(k.len(), nh * hd);
    let half = hd / 2;

    for h in 0..nh {
        let q_head = &mut q[h * hd..(h + 1) * hd];
        let k_head = &mut k[h * hd..(h + 1) * hd];

        for i in 0..half {
            let invf = 1.0 / theta.powf((i as f32 * 2.0) / (hd as f32));
            let ang = (pos as f32) * invf;
            let cos = ang.cos();
            let sin = ang.sin();

            let q0 = q_head[i];
            let q1 = q_head[i + half];
            q_head[i] = q0 * cos - q1 * sin;
            q_head[i + half] = q0 * sin + q1 * cos;

            let k0 = k_head[i];
            let k1 = k_head[i + half];
            k_head[i] = k0 * cos - k1 * sin;
            k_head[i + half] = k0 * sin + k1 * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![2.0, 1.0, 0.0, -1.0, 3.0, 2.0, 1.0, 0.0, -2.0];
        let d = dot(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((d - expected).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        rms_norm(&x, &weight, 1e-6, &mut out);

        let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let rstd = 1.0 / (mean_sq + 1e-6f32).sqrt();
        assert!((out[0] - 1.0 * rstd).abs() < 1e-5);
        assert!((out[3] - 4.0 * rstd).abs() < 1e-5);
    }

    #[test]
    fn test_gelu() {
        // PyTorch F.gelu(torch.tensor([0.0, 1.0, -1.0]))
        // 0.0 -> 0.0
        // 1.0 -> 0.8413447
        // -1.0 -> -0.1586553
        assert!((gelu(0.0) - 0.0).abs() < 1e-5);
        assert!((gelu(1.0) - 0.8413447).abs() < 1e-4);
        assert!((gelu(-1.0) - (-0.1586553)).abs() < 1e-4);
    }
}
