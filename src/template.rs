use anyhow::{Context, Result};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub const DEFAULT_CHAT_TEMPLATE: &str = r#"{% for message in messages %}{% if message['role'] == 'user' %}user: {% elif message['role'] == 'assistant' %}assistant: {% else %}{{ message['role'] }}: {% endif %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for part in message['content'] %}{{ part['text'] }}{% endfor %}{% endif %} {% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"#;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

pub struct ChatTemplateEngine {
    template_source: String,
}

impl ChatTemplateEngine {
    pub fn new(template_source: String) -> Self {
        Self { template_source }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let content = fs::read_to_string(p)
            .with_context(|| format!("Failed to read chat template from {}", p.display()))?;
        Ok(Self::new(content))
    }

    pub fn default_template() -> Self {
        Self::new(DEFAULT_CHAT_TEMPLATE.to_string())
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        let mut env = Environment::new();
        env.add_template("chat", &self.template_source)
            .context("Failed to parse Jinja chat template")?;
        let tmpl = env.get_template("chat")?;
        let ctx = context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
        };
        let rendered = tmpl
            .render(ctx)
            .context("Failed to render Jinja chat template")?;
        Ok(rendered)
    }

    pub fn render_user_prompt(&self, prompt: &str) -> Result<String> {
        let messages = vec![ChatMessage::user(prompt)];
        self.render(&messages, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_template_rendering() {
        let engine = ChatTemplateEngine::default_template();
        let rendered = engine.render_user_prompt("What is the capital of France?").unwrap();
        assert_eq!(rendered, "user: What is the capital of France? assistant: ");
    }

    #[test]
    fn test_multi_turn_template_rendering() {
        let engine = ChatTemplateEngine::default_template();
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ];
        let rendered = engine.render(&messages, true).unwrap();
        assert_eq!(
            rendered,
            "user: Hello assistant: Hi there! user: How are you? assistant: "
        );
    }
}
