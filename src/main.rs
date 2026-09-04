use anyhow::{Context, Result};
use clap::Parser;
use krikil::generate::{GenerationEngine, GenerationOptions};
use krikil::model::{PebbleModel, PebbleState};
use krikil::sampler::SamplerConfig;
use krikil::template::ChatTemplateEngine;
use krikil::tokenizer::PebbleTokenizer;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "krikil")]
#[command(author = "hanz")]
#[command(version = "0.1.0")]
#[command(about = "High-performance CPU inference engine for basically-ai/Pebble-25M-Chat", long_about = None)]
struct Args {
    /// Path to model safetensors weights
    #[arg(short = 'm', long = "model", default_value = "model.safetensors")]
    model: PathBuf,

    /// Path to tokenizer.json file (defaults to sibling of model or ./tokenizer.json)
    #[arg(long = "tokenizer")]
    tokenizer: Option<PathBuf>,

    /// Path to Jinja chat template file (optional, defaults to built-in template)
    #[arg(long = "template")]
    template: Option<PathBuf>,

    /// Maximum context window length
    #[arg(long = "context", default_value_t = 2048)]
    context: usize,

    /// Maximum new tokens to generate
    #[arg(long = "max-tokens", default_value_t = 128)]
    max_tokens: usize,

    /// Sampling temperature (0.0 for greedy decoding)
    #[arg(long = "temp", default_value_t = 0.7)]
    temperature: f32,

    /// Nucleus sampling probability threshold
    #[arg(long = "top-p", default_value_t = 0.95)]
    top_p: f32,

    /// Top-K filtering count
    #[arg(long = "top-k", default_value_t = 50)]
    top_k: usize,

    /// Repetition penalty factor
    #[arg(long = "repetition-penalty", default_value_t = 1.2)]
    repetition_penalty: f32,

    /// Random seed for deterministic generation
    #[arg(long = "seed")]
    seed: Option<u64>,

    /// Suppress timing and statistics printout at the end
    #[arg(long = "quiet", default_value_t = false)]
    quiet: bool,

    /// User prompt to generate a response for
    prompt: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Resolve template engine
    let template_engine = match &args.template {
        Some(path) => ChatTemplateEngine::from_file(path)?,
        None => ChatTemplateEngine::default_template(),
    };

    // 2. Resolve tokenizer
    let tokenizer_path = PebbleTokenizer::resolve_path(&args.model, args.tokenizer.as_deref())?;
    let tokenizer = PebbleTokenizer::from_file(&tokenizer_path)
        .with_context(|| format!("Failed to load tokenizer from {}", tokenizer_path.display()))?;

    // 3. Load model weights
    let model = PebbleModel::load_safetensors(&args.model)
        .with_context(|| format!("Failed to load model from {}", args.model.display()))?;

    // 4. Render prompt using chat template
    let formatted_prompt = template_engine
        .render_user_prompt(&args.prompt)
        .context("Failed to format prompt with chat template")?;

    let prompt_tokens = tokenizer
        .encode(&formatted_prompt)
        .context("Failed to tokenize formatted prompt")?;

    if prompt_tokens.is_empty() {
        anyhow::bail!("Prompt token sequence is empty");
    }

    if prompt_tokens.len() >= args.context {
        anyhow::bail!(
            "Prompt length ({} tokens) exceeds context limit ({})",
            prompt_tokens.len(),
            args.context
        );
    }

    // 5. Initialize model state
    let mut state = PebbleState::new(&model.config, args.context);

    // 6. Set up generation options
    let options = GenerationOptions {
        max_new_tokens: args.max_tokens,
        sampler_config: SamplerConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
            eos_token_id: tokenizer.eos_token_id(),
            seed: args.seed,
        },
        stream_stdout: true,
    };

    let engine = GenerationEngine::new(&model, &tokenizer);

    // 7. Stream generation
    let result = engine.generate(&prompt_tokens, &mut state, &options)?;

    // Trailing newline after stream output
    println!();

    if !args.quiet {
        eprintln!(
            "\n[Prompt: {} tokens, Generated: {} tokens | Prefill: {:?}, Decode: {:?}, Speed: {:.2} tok/s]",
            result.prompt_token_count,
            result.generated_token_count,
            result.prefill_duration,
            result.decode_duration,
            result.tokens_per_second
        );
    }

    Ok(())
}
