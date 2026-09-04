use crate::model::{PebbleModel, PebbleState};
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::PebbleTokenizer;
use anyhow::Result;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub sampler_config: SamplerConfig,
    pub stream_stdout: bool,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            sampler_config: SamplerConfig::default(),
            stream_stdout: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub generated_text: String,
    pub generated_tokens: Vec<u32>,
    pub prefill_duration: std::time::Duration,
    pub decode_duration: std::time::Duration,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub tokens_per_second: f64,
}

pub struct GenerationEngine<'a> {
    pub model: &'a PebbleModel,
    pub tokenizer: &'a PebbleTokenizer,
}

impl<'a> GenerationEngine<'a> {
    pub fn new(model: &'a PebbleModel, tokenizer: &'a PebbleTokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        state: &mut PebbleState,
        options: &GenerationOptions,
    ) -> Result<GenerationResult> {
        self.generate_with_callback(prompt_tokens, state, options, |_| {})
    }

    pub fn generate_with_callback<F>(
        &self,
        prompt_tokens: &[u32],
        state: &mut PebbleState,
        options: &GenerationOptions,
        mut on_token: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(&str),
    {
        state.reset();

        let prompt_len = prompt_tokens.len();
        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated_tokens = Vec::new();
        let mut generated_text = String::new();

        let mut sampler = Sampler::new(options.sampler_config.clone());
        let eos_token_id = options.sampler_config.eos_token_id;

        // 1. Prefill
        let prefill_start = Instant::now();
        let prefill_logits = self.model.forward_prefill(prompt_tokens, state)?;
        let prefill_duration = prefill_start.elapsed();

        // Sample first token from prefill logits
        let mut next_token = sampler.sample(&prefill_logits, &all_tokens);

        let decode_start = Instant::now();
        let mut token_count = 0;

        while token_count < options.max_new_tokens {
            if next_token == eos_token_id {
                break;
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);
            token_count += 1;

            let token_str = self
                .tokenizer
                .decode_token(next_token)
                .unwrap_or_else(|_| String::new());

            if options.stream_stdout {
                print!("{}", token_str);
                let _ = io::stdout().flush();
            }
            on_token(&token_str);
            generated_text.push_str(&token_str);

            if state.current_pos >= state.context_limit {
                break;
            }

            // Next step
            let step_logits = self.model.forward_step(next_token, state)?;
            next_token = sampler.sample(&step_logits, &all_tokens);
        }

        let decode_duration = decode_start.elapsed();
        let decode_secs = decode_duration.as_secs_f64();
        let tokens_per_second = if decode_secs > 0.0 {
            token_count as f64 / decode_secs
        } else {
            0.0
        };

        Ok(GenerationResult {
            generated_text,
            generated_tokens,
            prefill_duration,
            decode_duration,
            prompt_token_count: prompt_len,
            generated_token_count: token_count,
            tokens_per_second,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_generation_engine_e2e() {
        let model_path = Path::new("model.safetensors");
        let tokenizer_path = Path::new("tokenizer.json");
        if !model_path.exists() || !tokenizer_path.exists() {
            eprintln!("model.safetensors or tokenizer.json not found, skipping");
            return;
        }

        let model = PebbleModel::load_safetensors(model_path).unwrap();
        let tokenizer = PebbleTokenizer::from_file(tokenizer_path).unwrap();
        let mut state = PebbleState::new(&model.config, 2048);

        let engine = GenerationEngine::new(&model, &tokenizer);

        let prompt = "user: What is the capital of France? assistant: ";
        let prompt_tokens = tokenizer.encode(prompt).unwrap();

        let options = GenerationOptions {
            max_new_tokens: 15,
            sampler_config: SamplerConfig {
                temperature: 0.7,
                top_p: 0.95,
                top_k: 50,
                repetition_penalty: 1.2,
                eos_token_id: 0,
                seed: Some(12345),
            },
            stream_stdout: false,
        };

        let result = engine
            .generate(&prompt_tokens, &mut state, &options)
            .expect("Generation failed");

        println!("Generated text in test: {:?}", result.generated_text);
        println!(
            "Prefill: {:?}, Decode: {:?}, Speed: {:.2} tok/s",
            result.prefill_duration, result.decode_duration, result.tokens_per_second
        );

        assert!(!result.generated_tokens.is_empty());
        assert!(result.tokens_per_second > 0.0);
    }
}
