use anyhow::Result;
use std::path::PathBuf;
use std::process;
use crate::config::Config;
use crate::file_io;
use crate::parser;
use crate::gemini;
use crate::permissions;

pub async fn handle_compact(
    action: crate::CompactAction,
    config: &Config,
) -> Result<()> {
    // Compact mode suppresses all decorative output
    // Only outputs essential data or error codes

    match action {
        crate::CompactAction::Edit { file, query } => {
            match compact_edit(file, query, config).await {
                Ok(result) => {
                    println!("{}", result);
                    process::exit(0);
                }
                Err(_) => {
                    process::exit(1);
                }
            }
        }
        crate::CompactAction::Parse { file } => {
            match compact_parse(file).await {
                Ok(result) => {
                    println!("{}", result);
                    process::exit(0);
                }
                Err(_) => {
                    process::exit(1);
                }
            }
        }
        crate::CompactAction::Check { file } => {
            match compact_check(file).await {
                Ok(allowed) => {
                    process::exit(if allowed { 0 } else { 1 });
                }
                Err(_) => {
                    process::exit(2);
                }
            }
        }
    }
}

async fn compact_edit(file: PathBuf, query: String, config: &Config) -> Result<String> {
    // Check permissions silently
    if !permissions::verify_path_access(&file).await? {
        return Err(anyhow::anyhow!("Access denied"));
    }

    // Read file
    let original_content = if file.exists() {
        file_io::read_file(&file).await?
    } else {
        String::new()
    };

    // Detect language
    let language = file_io::detect_language(&file);

    // Get AI assistance
    let edited_content = gemini::edit_code(&original_content, &query, &language, config).await?;

    // Write back
    file_io::write_file(&file, &edited_content).await?;

    // Return the edited content
    Ok(edited_content)
}

async fn compact_parse(file: PathBuf) -> Result<String> {
    // Check permissions silently
    if !permissions::verify_path_access(&file).await? {
        return Err(anyhow::anyhow!("Access denied"));
    }

    // Read file
    let content = file_io::read_file(&file).await?;

    // Detect language
    let language = file_io::detect_language(&file);

    // Parse code - for compact mode, always return JSON
    let parse_result = parser::parse_code(&content, &language, "json").await?;

    // Return JSON representation
    Ok(serde_json::to_string(&parse_result)?)
}

async fn compact_check(file: PathBuf) -> Result<bool> {
    // Just check if the path is accessible
    permissions::verify_path_access(&file).await
}