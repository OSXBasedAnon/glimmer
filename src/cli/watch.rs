use anyhow::{Context, Result};
use std::path::PathBuf;
use notify::{Watcher, RecommendedWatcher, RecursiveMode, Event, EventKind};
use crate::{emerald_println, warn_println};
use std::sync::mpsc::channel;

pub async fn handle_watch(
    path: PathBuf,
    pattern: String,
    command: Option<String>,
) -> Result<()> {
    emerald_println!("Watching path: {} with pattern: {}", path.display(), pattern);
    
    if let Some(cmd) = &command {
        emerald_println!("Will execute command on changes: {}", cmd);
    }

    // Create a channel to receive the events
    let (tx, rx) = channel();

    // Create a watcher object
    let mut watcher = RecommendedWatcher::new(
        move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    if let Err(e) = tx.send(event) {
                        eprintln!("Failed to send event: {}", e);
                    }
                },
                Err(e) => eprintln!("Watch error: {}", e),
            }
        },
        notify::Config::default(),
    ).context("Failed to create file watcher")?;

    // Add a path to be watched
    watcher.watch(&path, RecursiveMode::Recursive)
        .with_context(|| format!("Failed to watch path: {}", path.display()))?;

    emerald_println!("Watching for changes... Press Ctrl+C to stop");

    loop {
        match rx.recv() {
            Ok(event) => {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                        for event_path in event.paths {
                            if matches_pattern(&event_path, &pattern) {
                                emerald_println!("File changed: {}", event_path.display());
                                
                                if let Some(cmd) = &command {
                                    execute_command(cmd).await?;
                                }
                            }
                        }
                    },
                    _ => {},
                }
            },
            Err(e) => {
                warn_println!("Watch error: {:?}", e);
                break;
            }
        }
    }

    Ok(())
}

fn matches_pattern(path: &PathBuf, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(filename) = path.file_name() {
        if let Some(filename_str) = filename.to_str() {
            return filename_str.contains(pattern) || 
                   glob_match(pattern, filename_str);
        }
    }
    
    false
}

fn glob_match(pattern: &str, text: &str) -> bool {
    // Simple glob matching - in a real implementation you'd use a proper glob library
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            let prefix = parts[0];
            let suffix = parts[1];
            return text.starts_with(prefix) && text.ends_with(suffix);
        }
    }
    
    pattern == text
}

async fn execute_command(command: &str) -> Result<()> {
    emerald_println!("Executing: {}", command);
    
    let output = tokio::process::Command::new("cmd")
        .args(["/C", command])
        .output()
        .await
        .context("Failed to execute command")?;

    if output.status.success() {
        if !output.stdout.is_empty() {
            println!("{}", String::from_utf8_lossy(&output.stdout));
        }
    } else {
        warn_println!("Command failed with status: {}", output.status);
        if !output.stderr.is_empty() {
            warn_println!("Error: {}", String::from_utf8_lossy(&output.stderr));
        }
    }

    Ok(())
}