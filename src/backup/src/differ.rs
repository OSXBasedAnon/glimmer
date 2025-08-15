use anyhow::Result;
use imara_diff::{
    Algorithm, Diff, InternedInput,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Change {
    pub change_type: String,
    pub old_line: Option<usize>,
    pub new_line: Option<usize>,
    pub content: String,
}

pub fn create_diff(old: &str, new: &str, old_name: &str, new_name: &str) -> Result<String> {
    let input = InternedInput::new(old, new);
    let diff = Diff::compute(Algorithm::Histogram, &input);
    
    // Get the original lines for reference
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    
    // Create a simple unified diff format manually
    let mut output = String::new();
    output.push_str(&format!("--- {}\n", old_name));
    output.push_str(&format!("+++ {}\n", new_name));
    
    for hunk in diff.hunks() {
        output.push_str(&format!("@@ -{},{} +{},{} @@\n", 
            hunk.before.start + 1, hunk.before.len(),
            hunk.after.start + 1, hunk.after.len()));
        
        // Process removed lines
        for idx in hunk.before.clone() {
            if diff.is_removed(idx) {
                if let Some(line) = old_lines.get(idx as usize) {
                    output.push_str(&format!("-{}\n", line));
                }
            }
        }
        
        // Process added lines
        for idx in hunk.after.clone() {
            if diff.is_added(idx) {
                if let Some(line) = new_lines.get(idx as usize) {
                    output.push_str(&format!("+{}\n", line));
                }
            }
        }
        
        // Process unchanged lines in context
        for idx in hunk.before.clone() {
            if !diff.is_removed(idx) {
                if let Some(line) = old_lines.get(idx as usize) {
                    output.push_str(&format!(" {}\n", line));
                }
            }
        }
    }
    
    Ok(output)
}

pub fn create_split_diff(old: &str, new: &str, old_name: &str, new_name: &str) -> Result<String> {
    let mut result = String::new();

    result.push_str(&format!("Left: {} | Right: {}\n", old_name, new_name));
    result.push_str(&"-".repeat(80));
    result.push('\n');

    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    let max_lines = old_lines.len().max(new_lines.len());

    for i in 0..max_lines {
        let old_line = old_lines.get(i).unwrap_or(&"");
        let new_line = new_lines.get(i).unwrap_or(&"");
        
        result.push_str(&format!("{:40} | {}\n", 
            format!("{:3}: {}", i + 1, old_line),
            format!("{:3}: {}", i + 1, new_line)
        ));
    }

    Ok(result)
}

/// Re-implementation of get_changes using imara-diff to restore functionality.
pub fn get_changes(old: &str, new: &str) -> Result<Vec<Change>> {
    let input = InternedInput::new(old, new);
    let diff = Diff::compute(Algorithm::Histogram, &input);
    
    // Get the original lines for reference
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    
    let mut changes = Vec::new();
    
    for hunk in diff.hunks() {
        // Process removed lines
        for idx in hunk.before.clone() {
            if diff.is_removed(idx) {
                if let Some(line) = old_lines.get(idx as usize) {
                    changes.push(Change {
                        change_type: "delete".to_string(),
                        old_line: Some((idx + 1) as usize),
                        new_line: None,
                        content: line.to_string(),
                    });
                }
            }
        }
        
        // Process added lines
        for idx in hunk.after.clone() {
            if diff.is_added(idx) {
                if let Some(line) = new_lines.get(idx as usize) {
                    changes.push(Change {
                        change_type: "insert".to_string(),
                        old_line: None,
                        new_line: Some((idx + 1) as usize),
                        content: line.to_string(),
                    });
                }
            }
        }
    }
    
    Ok(changes)
}

pub fn colorize_diff(diff: &str) -> String {
    use crate::cli::colors::*;
    
    let mut result = String::new();
    
    for line in diff.lines() {
        if line.starts_with('+') && !line.starts_with("+++") {
            if line.trim_start_matches('+').trim().contains("return Err") {
                // Light green background for error returns in additions
                result.push_str(&format!("\x1b[102m\x1b[30m{}\x1b[0m\n", line));
            } else {
                // Dark green background with white text for normal additions
                result.push_str(&format!("\x1b[42m\x1b[37m{}\x1b[0m\n", line));
            }
        } else if line.starts_with('-') && !line.starts_with("---") {
            if line.trim_start_matches('-').trim().contains("return Err") {
                // Light red background for error returns in deletions
                result.push_str(&format!("\x1b[101m\x1b[30m{}\x1b[0m\n", line));
            } else {
                // Dark red background with white text for normal deletions
                result.push_str(&format!("\x1b[41m\x1b[37m{}\x1b[0m\n", line));
            }
        } else if line.starts_with("@@") {
            result.push_str(&format!("{}{}{}\n", YELLOW_WARN, line, RESET));
        } else {
            result.push_str(&format!("{}\n", line));
        }
    }
    
    result
}