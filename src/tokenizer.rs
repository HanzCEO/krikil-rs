use anyhow::Result;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub const DEFAULT_EOS_TOKEN_ID: u32 = 0;

#[derive(Clone)]
pub struct PebbleTokenizer {
    inner: Tokenizer,
    eos_token_id: u32,
}

impl PebbleTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let inner = Tokenizer::from_file(p)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", p.display(), e))?;
        
        let eos_token_id = inner
            .token_to_id("<|eos|>")
            .unwrap_or(DEFAULT_EOS_TOKEN_ID);

        Ok(Self { inner, eos_token_id })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))
    }

    pub fn decode_token(&self, token: u32) -> Result<String> {
        self.inner
            .decode(&[token], false)
            .map_err(|e| anyhow::anyhow!("Failed to decode token {}: {}", token, e))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn resolve_path(
        model_path: &Path,
        explicit_tokenizer: Option<&Path>,
    ) -> Result<PathBuf> {
        if let Some(explicit) = explicit_tokenizer {
            if explicit.exists() {
                return Ok(explicit.to_path_buf());
            }
            anyhow::bail!("Explicit tokenizer path not found: {}", explicit.display());
        }

        let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
        let sibling = model_dir.join("tokenizer.json");
        if sibling.exists() {
            return Ok(sibling);
        }

        let cwd_candidate = Path::new("tokenizer.json");
        if cwd_candidate.exists() {
            return Ok(cwd_candidate.to_path_buf());
        }

        anyhow::bail!(
            "Could not find tokenizer.json. Looked at {} and {}. Use --tokenizer to specify its path.",
            sibling.display(),
            cwd_candidate.display()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encoding_and_decoding() {
        let path = Path::new("tokenizer.json");
        if !path.exists() {
            eprintln!("tokenizer.json not found in working directory, skipping test");
            return;
        }

        let tok = PebbleTokenizer::from_file(path).expect("Failed to load tokenizer");
        assert_eq!(tok.eos_token_id(), 0);

        let input = "user: What is the capital of France? assistant: ";
        let tokens = tok.encode(input).expect("Encoding failed");
        let expected = vec![
            397, 264, 26, 1270, 311, 263, 1530, 1360, 284, 420, 82, 549, 31, 842, 373, 419, 26,
            221,
        ];
        assert_eq!(tokens, expected);

        let decoded = tok.decode(&tokens).expect("Decoding failed");
        assert_eq!(decoded, input);
    }
}
