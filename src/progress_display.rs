use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use crate::cli::colors::*;

/// Advanced progress display system inspired by Cargo and Claude Code
pub struct ProgressDisplay {
    is_active: Arc<AtomicBool>,
    current_operation: Arc<std::sync::Mutex<String>>,
}

impl ProgressDisplay {
    pub fn new() -> Self {
        Self {
            is_active: Arc::new(AtomicBool::new(false)),
            current_operation: Arc::new(std::sync::Mutex::new(String::new())),
        }
    }

    /// Start a progress indicator for long-running operations
    pub async fn start_operation(&self, operation: &str) -> Result<ProgressHandle> {
        self.is_active.store(true, Ordering::Relaxed);
        
        if let Ok(mut current_op) = self.current_operation.lock() {
            *current_op = operation.to_string();
        }

        let is_active = Arc::clone(&self.is_active);
        let current_operation = Arc::clone(&self.current_operation);
        
        // Spawn the animation task
        let animation_task = tokio::spawn(async move {
            let spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut spinner_idx = 0;
            
            while is_active.load(Ordering::Relaxed) {
                if let Ok(operation) = current_operation.lock() {
                    if !operation.is_empty() {
                        // Skip progress animation in ratatui mode to avoid UI interference
                    }
                }
                
                spinner_idx = (spinner_idx + 1) % spinners.len();
                sleep(Duration::from_millis(100)).await;
            }
            
            // Skip clearing in ratatui mode
        });

        Ok(ProgressHandle {
            is_active: Arc::clone(&self.is_active),
            current_operation: Arc::clone(&self.current_operation),
            _animation_task: animation_task,
        })
    }

    /// Update the current operation text
    pub fn update_operation(&self, operation: &str) {
        if let Ok(mut current_op) = self.current_operation.lock() {
            *current_op = operation.to_string();
        }
    }

    /// Start a compilation progress display with cycling operations
    pub async fn start_compilation_progress(&self, operations: Vec<String>) -> Result<ProgressHandle> {
        self.is_active.store(true, Ordering::Relaxed);

        let is_active = Arc::clone(&self.is_active);
        let current_operation = Arc::clone(&self.current_operation);
        
        // Spawn the cycling compilation animation
        let animation_task = tokio::spawn(async move {
            let spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut spinner_idx = 0;
            let mut operation_idx = 0;
            let mut cycles_on_current_op = 0;
            
            while is_active.load(Ordering::Relaxed) {
                let current_op = if !operations.is_empty() {
                    operations[operation_idx % operations.len()].clone()
                } else {
                    "Compiling...".to_string()
                };
                
                // Update the shared operation
                if let Ok(mut op) = current_operation.lock() {
                    *op = current_op.clone();
                }
                
                // Skip progress display and flush in ratatui mode
                
                spinner_idx = (spinner_idx + 1) % spinners.len();
                cycles_on_current_op += 1;
                
                // Switch operation every 20 cycles (2 seconds at 100ms intervals)
                if cycles_on_current_op >= 20 {
                    operation_idx += 1;
                    cycles_on_current_op = 0;
                }
                
                sleep(Duration::from_millis(100)).await;
            }
            
            // Skip clearing in ratatui mode
        });

        Ok(ProgressHandle {
            is_active: Arc::clone(&self.is_active),
            current_operation: Arc::clone(&self.current_operation),
            _animation_task: animation_task,
        })
    }

    /// Display a simple loading bar for file operations
    pub async fn show_file_progress(&self, file_name: &str, progress: f32) -> Result<()> {
        let bar_width = 40;
        let filled = (progress * bar_width as f32) as usize;
        let empty = bar_width - filled;
        
        // Skip file progress display and flush in ratatui mode
        
        if progress >= 1.0 {
            // Skip println in ratatui mode: println!(); // New line when complete
        }
        
        Ok(())
    }

    /// Show a multi-line compilation status (like cargo does for many crates)
    pub async fn show_compilation_status(&self, status_lines: &[String]) {
        // Clear previous lines
        for _ in 0..status_lines.len() {
            // Skip print in ratatui mode: print!("\x1B[F\x1B[K"); // Move up and clear line
        }
        
        for (i, line) in status_lines.iter().enumerate() {
            let prefix = if i == status_lines.len() - 1 {
                format!("{}{}", BLUE_BRIGHT, "⠋") // Active spinner for current line
            } else {
                format!("{}{}", EMERALD_BRIGHT, "✓") // Check mark for completed
            };
            
            // Skip compilation status display in ratatui mode to avoid UI interference
        }
        
        // Move cursor back up to the active line
        if !status_lines.is_empty() {
            for _ in 0..status_lines.len() - 1 {
                // Skip print in ratatui mode: print!("\x1B[F");
            }
        }
    }
}

pub struct ProgressHandle {
    is_active: Arc<AtomicBool>,
    current_operation: Arc<std::sync::Mutex<String>>,
    _animation_task: tokio::task::JoinHandle<()>,
}

impl ProgressHandle {
    pub fn update_text(&self, text: &str) {
        if let Ok(mut op) = self.current_operation.lock() {
            *op = text.to_string();
        }
    }

    pub fn finish(self) {
        self.is_active.store(false, Ordering::Relaxed);
        // Task will clean up the display automatically
    }

    pub fn finish_with_message(self, message: &str) {
        self.is_active.store(false, Ordering::Relaxed);
        
        // Wait a moment for cleanup, then show completion message
        std::thread::sleep(Duration::from_millis(150));
        // Skip success message in ratatui mode to avoid UI interference
    }
}

impl Drop for ProgressHandle {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
}

/// Create compilation progress messages similar to cargo
pub fn generate_compilation_steps() -> Vec<String> {
    vec![
        "Checking dependencies...".to_string(),
        "Compiling proc-macro2 v1.0.89".to_string(),
        "Compiling unicode-ident v1.0.14".to_string(),
        "Compiling serde v1.0.219".to_string(),
        "Compiling tokio v1.47.1".to_string(),
        "Compiling reqwest v0.12.22".to_string(),
        "Compiling anyhow v1.0.98".to_string(),
        "Compiling regex v1.10.5".to_string(),
        "Compiling clap v4.5.4".to_string(),
        "Compiling rodio v0.19.0".to_string(),
        "Compiling glimmer v0.1.0".to_string(),
        "Finished dev profile".to_string(),
    ]
}

/// Utility function to wrap long operations with progress display
pub async fn with_progress<F, T>(operation_name: &str, future: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let progress = ProgressDisplay::new();
    let handle = progress.start_operation(operation_name).await?;
    
    let result = future.await;
    
    match &result {
        Ok(_) => handle.finish_with_message(&format!("✓ {}", operation_name)),
        Err(_) => {
            handle.finish();
            // Skip error message in ratatui mode to avoid UI interference
        }
    }
    
    result
}