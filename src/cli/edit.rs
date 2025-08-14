use anyhow::Result;
use std::path::PathBuf;
use crate::config::Config;

pub async fn handle_edit(
    file: PathBuf, // The file to edit
    _lang: Option<String>,
    query: String, // The instruction for the AI
    _output: Option<PathBuf>,
    config: &Config,
) -> Result<()> {
    // This function is now a clean wrapper around the robust, centralized
    // editing logic in `function_calling.rs`. This ensures both the CLI
    // and the chat agent use the same high-performance implementation.
    let file_path_str = file.to_str().unwrap_or_default().to_string();
    let function_call = crate::function_calling::FunctionCall {
        name: "edit_code".to_string(),
        arguments: std::collections::HashMap::from([
            ("file_path".to_string(), serde_json::Value::String(file_path_str)),
            ("query".to_string(), serde_json::Value::String(query)),
        ]),
    };

    // Use the main, robust editing logic. The conversation history is empty for direct CLI calls.
    match crate::function_calling::execute_function_call(&function_call, config, "").await {
        Ok(summary) => {
            crate::emerald_println!("{}", summary);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

// Removed broken apply_diff function - now using direct file replacement approach