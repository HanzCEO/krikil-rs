use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub eos_token_id: u32,
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 50,
            repetition_penalty: 1.2,
            eos_token_id: 0,
            seed: None,
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Sample the next token from raw logits given previously generated / context tokens.
    pub fn sample(&mut self, logits: &[f32], context_tokens: &[u32]) -> u32 {
        assert!(!logits.is_empty());
        let vocab_size = logits.len();

        // Clone logits for modification
        let mut modified_logits = logits.to_vec();

        // 1. Repetition penalty (must be applied before greedy or temperature)
        if (self.config.repetition_penalty - 1.0).abs() > 1e-4 {
            let mut seen_tokens = HashSet::new();
            for &token in context_tokens {
                seen_tokens.insert(token as usize);
            }
            let penalty = self.config.repetition_penalty;
            for &token in &seen_tokens {
                if token < vocab_size {
                    let logit = modified_logits[token];
                    if logit > 0.0 {
                        modified_logits[token] = logit / penalty;
                    } else {
                        modified_logits[token] = logit * penalty;
                    }
                }
            }
        }

        // 2. If temperature is effectively 0, perform greedy selection
        if self.config.temperature <= 1e-6 {
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in modified_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            return best_idx as u32;
        }

        // 3. Temperature scaling
        let inv_temp = 1.0 / self.config.temperature;
        for val in modified_logits.iter_mut() {
            *val *= inv_temp;
        }

        // 4. Collect candidate (token_id, logit) pairs
        let mut candidates: Vec<(u32, f32)> = modified_logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, l))
            .collect();

        // Sort candidates in descending order of logit
        candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 5. Top-K filtering
        if self.config.top_k > 0 && self.config.top_k < candidates.len() {
            candidates.truncate(self.config.top_k);
        }

        // 6. Compute softmax probabilities over remaining candidates
        let max_logit = candidates[0].1;
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&(id, l)| {
                let p = (l - max_logit).exp();
                sum_exp += p;
                (id, p)
            })
            .collect();

        let inv_sum = 1.0 / sum_exp;
        for (_, p) in probs.iter_mut() {
            *p *= inv_sum;
        }

        // 7. Top-P (nucleus) filtering
        if self.config.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cut_idx = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= self.config.top_p {
                    cut_idx = (i + 1).min(probs.len());
                    break;
                }
            }
            probs.truncate(cut_idx);

            // Renormalize
            let new_sum: f32 = probs.iter().map(|&(_, p)| p).sum();
            let inv_new_sum = 1.0 / new_sum;
            for (_, p) in probs.iter_mut() {
                *p *= inv_new_sum;
            }
        }

        // 8. Sample from probability distribution
        let r: f32 = self.rng.gen();
        let mut acc = 0.0f32;
        for &(id, p) in &probs {
            acc += p;
            if r <= acc {
                return id;
            }
        }

        // Fallback to highest probability candidate
        probs[0].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let token = sampler.sample(&logits, &[]);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 3.0,
            ..Default::default()
        });
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let token = sampler.sample(&logits, &[1u32]);
        eprintln!("Repetition penalty chosen token: {}", token);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sampling_deterministic_with_seed() {
        let config = SamplerConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            repetition_penalty: 1.0,
            eos_token_id: 0,
            seed: Some(42),
        };
        let mut s1 = Sampler::new(config.clone());
        let mut s2 = Sampler::new(config);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = s1.sample(&logits, &[]);
        let t2 = s2.sample(&logits, &[]);
        assert_eq!(t1, t2);
    }
}
