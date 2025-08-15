// Modern chat interface with box-style input prompts
use crate::cli::colors::{EMERALD_BRIGHT, EMERALD_LIGHT, GRAY_DIM, WHITE_BRIGHT, RESET};
use std::io::{self, Write};

pub struct ChatUI {
    box_width: usize,
    show_timestamps: bool,
}

impl Default for ChatUI {
    fn default() -> Self {
        Self {
            box_width: 80,
            show_timestamps: true,
        }
    }
}

impl ChatUI {
    pub fn new(box_width: usize, show_timestamps: bool) -> Self {
        Self {
            box_width,
            show_timestamps,
        }
    }

    /// Display a modern chat input prompt with box styling
    pub fn show_input_prompt(&self) -> Result<String, io::Error> {
        let term_width = terminal_width();
        let effective_width = if term_width > 0 && term_width < self.box_width {
            term_width
        } else {
            self.box_width
        };

        // Create box border
        let horizontal_line = "─".repeat(effective_width.saturating_sub(4));
        
        // Top border
        println!("{}{} ┌{}┐{}", EMERALD_BRIGHT, "│", horizontal_line, RESET);
        
        // Input label line
        let label = " Message ";
        let padding_left = (effective_width.saturating_sub(label.len() + 4)) / 2;
        let padding_right = effective_width.saturating_sub(label.len() + 4 + padding_left);
        
        print!("{}{} │{}", EMERALD_BRIGHT, "│", RESET);
        print!("{}{}", " ".repeat(padding_left), EMERALD_LIGHT);
        print!("{}", label);
        print!("{}{}", RESET, " ".repeat(padding_right));
        println!("{}{} │{}", EMERALD_BRIGHT, "│", RESET);
        
        // Separator line
        println!("{}{} ├{}┤{}", EMERALD_BRIGHT, "│", "─".repeat(effective_width.saturating_sub(4)), RESET);
        
        // Input area - show multiline input box
        print!("{}{} │{} ", EMERALD_BRIGHT, "│", RESET);
        io::stdout().flush()?;
        
        // Read the input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed_input = input.trim().to_string();
        
        // Bottom border
        println!("{}{} └{}┘{}", EMERALD_BRIGHT, "│", horizontal_line, RESET);
        println!(); // Extra spacing
        
        Ok(trimmed_input)
    }

    /// Display assistant response in a styled box
    pub fn display_response(&self, response: &str, thinking: Option<&str>) {
        let term_width = terminal_width();
        let effective_width = if term_width > 0 && term_width < self.box_width {
            term_width
        } else {
            self.box_width
        };

        // Create box border
        let horizontal_line = "═".repeat(effective_width.saturating_sub(4));
        
        // Top border with AI label
        println!("{} ╔{}╗", EMERALD_LIGHT, horizontal_line);
        
        let label = " ✨ Glimmer AI ";
        let padding_left = (effective_width.saturating_sub(label.len() + 4)) / 2;
        let padding_right = effective_width.saturating_sub(label.len() + 4 + padding_left);
        
        print!("{} ║{}", EMERALD_LIGHT, " ".repeat(padding_left));
        print!("{}{}{}", EMERALD_BRIGHT, label, RESET);
        print!("{}", " ".repeat(padding_right));
        println!("{} ║", EMERALD_LIGHT);
        
        // Separator
        println!("{} ╠{}╣", EMERALD_LIGHT, "═".repeat(effective_width.saturating_sub(4)));
        
        // Display thinking process if available
        if let Some(thinking_content) = thinking {
            print!("{} ║{} ", EMERALD_LIGHT, GRAY_DIM);
            print!("💭 Thinking: ");
            
            // Wrap thinking text
            let thinking_lines = wrap_text(thinking_content, effective_width.saturating_sub(6));
            for (i, line) in thinking_lines.iter().enumerate() {
                if i > 0 {
                    print!("{} ║{} ", EMERALD_LIGHT, " ".repeat(12));
                }
                println!("{}{}", line, RESET);
            }
            
            // Separator for thinking
            println!("{} ╠{}╣", EMERALD_LIGHT, "─".repeat(effective_width.saturating_sub(4)));
        }
        
        // Display main response
        let response_lines = wrap_text(response, effective_width.saturating_sub(6));
        for line in response_lines {
            print!("{} ║{} ", EMERALD_LIGHT, WHITE_BRIGHT);
            println!("{}{}", line, RESET);
        }
        
        // Bottom border
        println!("{} ╚{}╝", EMERALD_LIGHT, horizontal_line);
        println!(); // Extra spacing
    }

    /// Display user message in a simpler style
    pub fn display_user_message(&self, message: &str) {
        let term_width = terminal_width();
        let effective_width = if term_width > 0 && term_width < self.box_width {
            term_width
        } else {
            self.box_width
        };

        // Simple right-aligned user message box
        let horizontal_line = "─".repeat(effective_width.saturating_sub(10));
        let indent = " ".repeat(6);
        
        println!("{}    ┌{}┐", indent, horizontal_line);
        
        let message_lines = wrap_text(message, effective_width.saturating_sub(16));
        for line in message_lines {
            println!("{}    │ {}{}{} │", indent, GRAY_DIM, line, RESET);
        }
        
        println!("{}    └{}┘", indent, horizontal_line);
        println!();
    }

    /// Display a system message or notification
    pub fn display_system_message(&self, message: &str, message_type: SystemMessageType) {
        let (icon, color) = match message_type {
            SystemMessageType::Info => ("ℹ", EMERALD_BRIGHT),
            SystemMessageType::Warning => ("⚠", crate::cli::colors::YELLOW_WARN),
            SystemMessageType::Error => ("❌", crate::cli::colors::RED_ERROR),
            SystemMessageType::Success => ("✅", EMERALD_BRIGHT),
        };

        println!("{}{}  {} {}{}", color, icon, message, RESET, "");
        println!();
    }

    /// Show a loading/thinking animation in box style
    pub fn show_thinking_box(&self, context: &str) {
        let term_width = terminal_width();
        let effective_width = if term_width > 0 && term_width < self.box_width {
            term_width
        } else {
            self.box_width
        };

        let horizontal_line = "─".repeat(effective_width.saturating_sub(4));
        
        println!("{} ┌{}┐", EMERALD_LIGHT, horizontal_line);
        print!("{} │ {}💭 Thinking", EMERALD_LIGHT, GRAY_DIM);
        if !context.is_empty() {
            print!(": {}", context);
        }
        println!("{}{}│", " ".repeat(effective_width.saturating_sub(15 + context.len())), EMERALD_LIGHT);
        println!("{} └{}┘", EMERALD_LIGHT, horizontal_line);
    }
}

#[derive(Debug, Clone)]
pub enum SystemMessageType {
    Info,
    Warning, 
    Error,
    Success,
}

/// Get terminal width, fallback to 80 if not available
fn terminal_width() -> usize {
    if let Some((width, _)) = term_size::dimensions() {
        width.min(120) // Cap at reasonable max width
    } else {
        80 // Fallback width
    }
}

/// Wrap text to fit within specified width
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    
    for paragraph in text.split('\n') {
        if paragraph.trim().is_empty() {
            lines.push(String::new());
            continue;
        }
        
        let words: Vec<&str> = paragraph.split_whitespace().collect();
        let mut current_line = String::new();
        
        for word in words {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else if current_line.len() + word.len() + 1 <= max_width {
                current_line.push(' ');
                current_line.push_str(word);
            } else {
                lines.push(current_line);
                current_line = word.to_string();
            }
        }
        
        if !current_line.is_empty() {
            lines.push(current_line);
        }
    }
    
    if lines.is_empty() {
        lines.push(String::new());
    }
    
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wrap_text() {
        let text = "This is a long line that should be wrapped at the specified width";
        let wrapped = wrap_text(text, 20);
        
        assert!(!wrapped.is_empty());
        for line in &wrapped {
            assert!(line.len() <= 20);
        }
    }
    
    #[test]
    fn test_wrap_text_with_newlines() {
        let text = "First line\n\nSecond line after empty line";
        let wrapped = wrap_text(text, 50);
        
        assert_eq!(wrapped.len(), 3);
        assert_eq!(wrapped[1], "");
    }
}