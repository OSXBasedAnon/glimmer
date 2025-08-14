use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::fs;
use serde_json;
use crate::config;
use crate::permissions::PermissionManager;
use crate::{emerald_println, warn_println};

pub async fn handle_export(
    export_type: String,
    output: PathBuf,
    format: String,
) -> Result<()> {
    emerald_println!("Exporting {} as {} to: {}", export_type, format, output.display());

    match export_type.as_str() {
        "conversation" => export_conversation(output, format).await?,
        "config" => export_config(output, format).await?,
        "permissions" => export_permissions(output, format).await?,
        _ => return Err(anyhow::anyhow!("Unknown export type: {}", export_type)),
    }

    emerald_println!("Export completed successfully");
    Ok(())
}

async fn export_conversation(output: PathBuf, format: String) -> Result<()> {
    // Look for recent conversation files in common locations
    let mut conversation_files = Vec::new();
    
    // Check current directory
    if let Ok(entries) = fs::read_dir(".").await {
        let mut entries = entries;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                if filename.to_string_lossy().contains("conversation") && 
                   path.extension().map_or(false, |ext| ext == "json") {
                    conversation_files.push(path);
                }
            }
        }
    }

    if conversation_files.is_empty() {
        warn_println!("No conversation files found in current directory");
        return Ok(());
    }

    // Export the most recent conversation file
    let latest_file = conversation_files.into_iter()
        .max_by_key(|path| {
            path.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        })
        .context("Failed to find latest conversation")?;

    let content = fs::read_to_string(&latest_file).await?;
    
    match format.as_str() {
        "json" => {
            fs::write(&output, content).await?;
        }
        "markdown" => {
            let conversation: serde_json::Value = serde_json::from_str(&content)?;
            let markdown = convert_conversation_to_markdown(&conversation)?;
            fs::write(&output, markdown).await?;
        }
        "txt" => {
            let conversation: serde_json::Value = serde_json::from_str(&content)?;
            let text = convert_conversation_to_text(&conversation)?;
            fs::write(&output, text).await?;
        }
        _ => return Err(anyhow::anyhow!("Unsupported format: {}", format)),
    }

    Ok(())
}

async fn export_config(output: PathBuf, format: String) -> Result<()> {
    let config = config::load_config(None).await?;
    
    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&config)?;
            fs::write(&output, json).await?;
        }
        "toml" => {
            let toml_content = toml::to_string_pretty(&config)?;
            fs::write(&output, toml_content).await?;
        }
        _ => return Err(anyhow::anyhow!("Config export only supports json and toml formats")),
    }

    Ok(())
}

async fn export_permissions(output: PathBuf, format: String) -> Result<()> {
    let manager = PermissionManager::new().await?;
    let allowed_paths = manager.list_allowed_paths();
    
    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&allowed_paths)?;
            fs::write(&output, json).await?;
        }
        "txt" => {
            let mut content = String::new();
            content.push_str("Glimmer Allowed Paths\n");
            content.push_str("=====================\n\n");
            
            for path in allowed_paths {
                content.push_str(&format!("{}\n", path.display()));
            }
            
            fs::write(&output, content).await?;
        }
        "markdown" => {
            let mut content = String::new();
            content.push_str("# Glimmer Allowed Paths\n\n");
            
            for path in allowed_paths {
                content.push_str(&format!("- `{}`\n", path.display()));
            }
            
            fs::write(&output, content).await?;
        }
        _ => return Err(anyhow::anyhow!("Unsupported format: {}", format)),
    }

    Ok(())
}

fn convert_conversation_to_markdown(conversation: &serde_json::Value) -> Result<String> {
    let mut markdown = String::new();
    markdown.push_str("# Conversation Export\n\n");
    
    if let Some(messages) = conversation.get("messages").and_then(|m| m.as_array()) {
        for message in messages {
            if let (Some(role), Some(content)) = (
                message.get("role").and_then(|r| r.as_str()),
                message.get("content").and_then(|c| c.as_str())
            ) {
                match role {
                    "user" => {
                        markdown.push_str("## User\n\n");
                        markdown.push_str(content);
                        markdown.push_str("\n\n");
                    }
                    "assistant" => {
                        markdown.push_str("## Assistant\n\n");
                        markdown.push_str(content);
                        markdown.push_str("\n\n");
                    }
                    _ => {}
                }
            }
        }
    }
    
    Ok(markdown)
}

fn convert_conversation_to_text(conversation: &serde_json::Value) -> Result<String> {
    let mut text = String::new();
    text.push_str("Conversation Export\n");
    text.push_str("===================\n\n");
    
    if let Some(messages) = conversation.get("messages").and_then(|m| m.as_array()) {
        for message in messages {
            if let (Some(role), Some(content)) = (
                message.get("role").and_then(|r| r.as_str()),
                message.get("content").and_then(|c| c.as_str())
            ) {
                text.push_str(&format!("{}: {}\n\n", 
                    if role == "user" { "You" } else { "Assistant" },
                    content
                ));
            }
        }
    }
    
    Ok(text)
}