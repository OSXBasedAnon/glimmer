use crate::cli::colors::{PURPLE_BOLD, GRAY_DIM, RESET};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::time::sleep;

pub struct ThinkingIndicator {
    is_running: Arc<AtomicBool>,
    is_interrupted: Arc<AtomicBool>,
    handle: Option<tokio::task::JoinHandle<()>>,
    start_time: std::time::Instant,
}

impl ThinkingIndicator {
    pub fn new() -> Self {
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            is_interrupted: Arc::new(AtomicBool::new(false)),
            handle: None,
            start_time: std::time::Instant::now(),
        }
    }
    
    pub fn is_interrupted(&self) -> bool {
        self.is_interrupted.load(Ordering::Relaxed)
    }

    pub fn start(&mut self, context: String) {
        if self.is_running.load(Ordering::Relaxed) {
            return; // Already running
        }

        self.start_time = std::time::Instant::now(); // Reset timer
        self.is_running.store(true, Ordering::Relaxed);
        self.is_interrupted.store(false, Ordering::Relaxed);
        
        let is_running = Arc::clone(&self.is_running);
        let start_time = self.start_time;

        let handle = tokio::spawn(async move {
            let thinking_words = [
                "Buffering", "Marinating", "Consulting", "Toasting", "Searching",
                "Hamster", "Bartholomew", "Downloading", "Recalibrating", "Polishing",
                "Fritz", "Ripening", "Unboxing", "Hibernating", "Snorkelling",
                "Unlocking", "Squirrels", "Brainstorming", "Pondering", "Calculating"
            ];

            let mut current_word_index = 0;
            let mut dots = String::new();
            // Use a simple index-based approach to avoid Send issues
            let word_index = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as usize % thinking_words.len();
            let mut word = thinking_words[word_index];

            // This animation is for non-TUI contexts. The main chat loop handles its own spinner.
            // The check for is_running is sufficient. The main event loop will handle interrupts.
            while is_running.load(Ordering::Relaxed) {
                // Clear the line and move cursor to beginning
                print!("\r\x1b[K");
                
                // Add dots animation
                dots = match dots.len() {
                    0 => ".".to_string(),
                    1 => "..".to_string(),
                    2 => "...".to_string(),
                    _ => "".to_string(),
                };

                // Calculate elapsed time
                let elapsed = start_time.elapsed();
                let seconds = elapsed.as_secs();
                let time_str = if seconds < 60 {
                    format!("{}s", seconds)
                } else {
                    format!("{}m{}s", seconds / 60, seconds % 60)
                };

                // Print the thinking indicator with ⌘ symbol before buffering message
                print!("{}⌘{} {}{}{} {}{} - {} - ESC to interrupt{}{}", 
                    PURPLE_BOLD, RESET,
                    PURPLE_BOLD, word, RESET,
                    GRAY_DIM, context, time_str, dots, RESET
                );

                std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());

                // Change word occasionally
                current_word_index += 1;
                if current_word_index % 8 == 0 {
                    let new_word_index = (word_index + current_word_index / 8) % thinking_words.len();
                    word = thinking_words[new_word_index];
                }

                sleep(Duration::from_millis(500)).await;
            }

            // The main loop will handle displaying any interruption messages.
            print!("\r\x1b[K");
            std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
        });

        self.handle = Some(handle);
    }

    pub async fn stop(&mut self) {
        self.is_running.store(false, Ordering::Relaxed);
        
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }

        // Make sure line is cleared (unless interrupted)
        if !self.is_interrupted.load(Ordering::Relaxed) {
            print!("\r\x1b[K");
            std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
        }
    }

    pub fn update_context(&self, _new_context: String) {
        // For now, we'll implement this in a future version
        // Could use a shared context variable that the animation reads from
    }
}

impl Drop for ThinkingIndicator {
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::Relaxed);
        // Clear the line (unless interrupted)
        if !self.is_interrupted.load(Ordering::Relaxed) {
            print!("\r\x1b[K");
            std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
        }
    }
}

// Convenience functions for different contexts
pub async fn show_thinking_with_context(context: &str) -> ThinkingIndicator {
    let mut indicator = ThinkingIndicator::new();
    indicator.start(format!("({})", context));
    indicator
}

pub async fn show_file_search(query: &str) -> ThinkingIndicator {
    show_thinking_with_context(&format!("searching for files matching '{}'", query)).await
}

pub async fn show_ai_processing(task: &str) -> ThinkingIndicator {
    show_thinking_with_context(&format!("asking AI to {}", task)).await
}

pub async fn show_ai_processing_with_tokens(task: &str, token_count: u32) -> ThinkingIndicator {
    show_thinking_with_context(&format!("asking AI to {} ({} tokens)", task, token_count)).await
}

pub async fn show_permission_check(path: &str) -> ThinkingIndicator {
    show_thinking_with_context(&format!("checking permissions for {}", path)).await
}

pub async fn show_fuzzy_matching(input: &str) -> ThinkingIndicator {
    show_thinking_with_context(&format!("interpreting '{}'", input)).await
}