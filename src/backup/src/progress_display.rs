use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use std::sync::Mutex;
use std::collections::VecDeque;
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
    pub async fn show_file_progress(&self, _file_name: &str, progress: f32) -> Result<()> {
        let bar_width = 40;
        let filled = (progress * bar_width as f32) as usize;
        let _empty = bar_width - filled;
        
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
        
        for (i, _line) in status_lines.iter().enumerate() {
            let _prefix = if i == status_lines.len() - 1 {
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

    pub fn finish_with_message(self, _message: &str) {
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

/// Perfect Claude Code / Gemini CLI style real-time UI for chat integration
pub struct RealtimeUI {
    operations: Arc<Mutex<VecDeque<Operation>>>,
    output_lines: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
struct Operation {
    id: String,
    operation_type: OperationType,
    status: OperationStatus,
    token_info: Option<String>,
    result: Option<String>,
}

#[derive(Clone)]
enum OperationType {
    Analyzing,
    Editing,
    Reading,
    Listing,
    TaskComplete,
}

#[derive(Clone, PartialEq)]
enum OperationStatus {
    InProgress,
    Completed,
}

impl RealtimeUI {
    pub fn new() -> Self {
        Self {
            operations: Arc::new(Mutex::new(VecDeque::new())),
            output_lines: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start an operation with real-time display
    pub fn start_operation(&self, operation_id: &str, op_type: OperationType, token_info: Option<&str>) {
        let operation = Operation {
            id: operation_id.to_string(),
            operation_type: op_type.clone(),
            status: OperationStatus::InProgress,
            token_info: token_info.map(|s| s.to_string()),
            result: None,
        };

        // Display immediately
        let (color, name) = match op_type {
            OperationType::Analyzing => (YELLOW_ANALYZE, "Analyzing code"),
            OperationType::Editing => (ORANGE_EDIT, "Editing code"),
            OperationType::Reading => (WHITE_BRIGHT, "Read"),
            OperationType::Listing => (WHITE_BRIGHT, "List"),
            OperationType::TaskComplete => (GREEN_COMPLETE, "Task completed"),
        };

        // Send directly to chat via reasoning steps 
        let line = if let Some(tokens) = &operation.token_info {
            format!("{}●{} {} ({})", color, RESET, name, tokens)
        } else {
            format!("{}●{} {}", color, RESET, name)
        };
        
        // Add to reasoning steps for immediate chat display
        crate::thinking_display::PersistentStatusBar::add_reasoning_step(&line);

        // Store operation
        if let Ok(mut ops) = self.operations.lock() {
            ops.push_back(operation);
            // Keep only last 10 operations for memory
            if ops.len() > 10 {
                ops.pop_front();
            }
        }
    }

    /// Complete an operation
    pub fn complete_operation(&self, operation_id: &str, result: &str) {
        // Update the operation
        if let Ok(mut ops) = self.operations.lock() {
            if let Some(op) = ops.iter_mut().find(|o| o.id == operation_id) {
                op.status = OperationStatus::Completed;
                op.result = Some(result.to_string());
            }
        }

        // Send completion directly to chat
        let completion_line = format!("  ⎿ {}", result);
        crate::thinking_display::PersistentStatusBar::add_reasoning_step(&completion_line);
    }

    /// Show task completion (final green bullet)
    pub fn show_task_completion(&self, summary: &str) {
        let task_line = format!("{}● Task completed{}", GREEN_COMPLETE, RESET);
        let summary_line = format!("  ⎿ {}", summary);
        
        // Send directly to chat
        crate::thinking_display::PersistentStatusBar::add_reasoning_step(&task_line);
        crate::thinking_display::PersistentStatusBar::add_reasoning_step(&summary_line);
    }
    
    /// Get new output lines for chat integration
    pub fn get_new_lines(&self) -> Vec<String> {
        if let Ok(mut lines) = self.output_lines.lock() {
            let new_lines = lines.clone();
            lines.clear();
            new_lines
        } else {
            Vec::new()
        }
    }

    /// Clear all operations
    pub fn clear(&self) {
        if let Ok(mut ops) = self.operations.lock() {
            ops.clear();
        }
    }
}

/// Global UI instance
static mut REALTIME_UI: Option<RealtimeUI> = None;
static mut INITIALIZED: bool = false;

pub fn init_realtime_ui() {
    unsafe {
        if !INITIALIZED {
            REALTIME_UI = Some(RealtimeUI::new());
            INITIALIZED = true;
        }
    }
}

fn get_ui() -> &'static RealtimeUI {
    unsafe {
        if !INITIALIZED {
            init_realtime_ui();
        }
        REALTIME_UI.as_ref().unwrap()
    }
}

/// Public API functions that match your exact requirements

/// Start analyzing with token count display
pub fn start_analyzing(token_count: u32) -> String {
    let operation_id = format!("analyze_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
    let token_info = format!("{}→0", token_count);
    get_ui().start_operation(&operation_id, OperationType::Analyzing, Some(&token_info));
    operation_id
}

/// Start editing with token count display  
pub fn start_editing(token_count: u32) -> String {
    let operation_id = format!("edit_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
    let token_info = format!("{}→0", token_count);
    get_ui().start_operation(&operation_id, OperationType::Editing, Some(&token_info));
    operation_id
}

/// Complete any operation
pub fn complete_operation(operation_id: &str, result: &str) {
    get_ui().complete_operation(operation_id, result);
}

/// Show final task completion
pub fn task_completed(summary: &str) {
    get_ui().show_task_completion(summary);
}

/// Clear all operations (for clean slate)
pub fn clear_operations() {
    get_ui().clear();
}

/// Get new real-time UI lines for chat integration
pub fn get_new_ui_lines() -> Vec<String> {
    get_ui().get_new_lines()
}

/// Convenience functions for quick operations
pub fn quick_analyze_complete(token_count: u32, result: &str) {
    let id = start_analyzing(token_count);
    complete_operation(&id, result);
}

pub fn quick_edit_complete(token_count: u32, result: &str) {
    let id = start_editing(token_count);
    complete_operation(&id, result);
}