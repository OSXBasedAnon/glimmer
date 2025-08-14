use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use crate::cli::colors::*;

// Define missing color constants
const GREEN_SUCCESS: &str = "\x1b[32m"; // Green
const CYAN_INFO: &str = "\x1b[36m";     // Cyan

/// Claude Code-style real-time display system
/// Shows function execution, file changes, and results above the thinking display
pub struct RealtimeDisplay {
    is_active: Arc<AtomicBool>,
    output_buffer: Arc<Mutex<Vec<String>>>,
}

impl RealtimeDisplay {
    pub fn new() -> Self {
        Self {
            is_active: Arc::new(AtomicBool::new(true)),
            output_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Show function call being executed
    pub async fn show_function_call(&self, function_name: &str, description: &str) {
        let message = format!(
            "\n{}Executing {} → {}{}", 
            BLUE_BRIGHT, function_name, description, RESET
        );
        self.add_output_line(&message).await;
    }

    /// Show file operation being performed
    pub async fn show_file_operation(&self, operation: &str, file_path: &str) {
        let message = format!(
            "{}  📝 {} {}{}", 
            GRAY_DIM, operation, file_path, RESET
        );
        self.add_output_line(&message).await;
    }

    /// Show code changes/diff
    pub async fn show_code_diff(&self, file_path: &str, lines_changed: usize) {
        let message = format!(
            "{}📄 Modified {} ({} lines changed){}", 
            GREEN_SUCCESS, file_path, lines_changed, RESET
        );
        self.add_output_line(&message).await;
    }


    /// Show research/web fetch results
    pub async fn show_research_result(&self, title: &str, url: &str) {
        let message = format!(
            "{}🔍 Found: {}\n{}   Source: {}{}", 
            CYAN_INFO, title, GRAY_DIM, url, RESET
        );
        self.add_output_line(&message).await;
    }

    /// Show function execution result
    pub async fn show_function_result(&self, function_name: &str, result_summary: &str) {
        let message = format!(
            "{}  ✅ {} completed → {}{}", 
            GREEN_SUCCESS, function_name, result_summary, RESET
        );
        self.add_output_line(&message).await;
    }

    /// Show error during execution
    pub async fn show_error(&self, operation: &str, error: &str) {
        let message = format!(
            "{}❌ {} failed: {}{}", 
            RED_ERROR, operation, error, RESET
        );
        self.add_output_line(&message).await;
    }

    /// Show directory listing
    pub async fn show_directory_listing(&self, path: &str, items: &[String]) {
        let message = format!(
            "{}📁 Contents of {}:{}\n{}{}{}", 
            CYAN_INFO, path, RESET,
            GRAY_DIM,
            items.join("\n"),
            RESET
        );
        self.add_output_line(&message).await;
    }

    /// Show file content preview
    pub async fn show_file_preview(&self, file_path: &str, preview: &str) {
        let message = format!(
            "{}Preview of {}:{}\n{}{}{}", 
            CYAN_INFO, file_path, RESET,
            GRAY_DIM,
            preview,
            RESET
        );
        self.add_output_line(&message).await;
    }

    /// Add a line to the output buffer and display immediately
    async fn add_output_line(&self, line: &str) {
        if self.is_active.load(Ordering::Relaxed) {
            // Add to buffer for history
            {
                let mut buffer = self.output_buffer.lock().await;
                buffer.push(line.to_string());
                
                // Keep buffer manageable (last 50 lines)
                if buffer.len() > 50 {
                    buffer.remove(0);
                }
            }
            
            // Skip direct printing in ratatui mode to avoid UI interference
            // The output is stored in buffer for potential future use
        }
    }

    /// Clear the display
    pub async fn clear(&self) {
        let mut buffer = self.output_buffer.lock().await;
        buffer.clear();
    }

    /// Get all output history
    pub async fn get_history(&self) -> Vec<String> {
        let buffer = self.output_buffer.lock().await;
        buffer.clone()
    }

    /// Stop the display
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
}

/// Global realtime display instance
static mut GLOBAL_DISPLAY: Option<RealtimeDisplay> = None;
static mut DISPLAY_INITIALIZED: bool = false;

/// Initialize the global realtime display
pub fn init_realtime_display() {
    unsafe {
        if !DISPLAY_INITIALIZED {
            GLOBAL_DISPLAY = Some(RealtimeDisplay::new());
            DISPLAY_INITIALIZED = true;
        }
    }
}

/// Get the global realtime display instance
pub fn get_realtime_display() -> Option<&'static RealtimeDisplay> {
    unsafe {
        GLOBAL_DISPLAY.as_ref()
    }
}

/// Convenience functions for global display
pub async fn show_function_call(function_name: &str, description: &str) {
    if let Some(display) = get_realtime_display() {
        display.show_function_call(function_name, description).await;
    }
}

pub async fn show_file_operation(operation: &str, file_path: &str) {
    if let Some(display) = get_realtime_display() {
        display.show_file_operation(operation, file_path).await;
    }
}

pub async fn show_code_diff(file_path: &str, lines_changed: usize) {
    if let Some(display) = get_realtime_display() {
        display.show_code_diff(file_path, lines_changed).await;
    }
}


pub async fn show_research_result(title: &str, url: &str) {
    if let Some(display) = get_realtime_display() {
        display.show_research_result(title, url).await;
    }
}

pub async fn show_function_result(function_name: &str, result_summary: &str) {
    if let Some(display) = get_realtime_display() {
        display.show_function_result(function_name, result_summary).await;
    }
}

pub async fn show_error(operation: &str, error: &str) {
    if let Some(display) = get_realtime_display() {
        display.show_error(operation, error).await;
    }
}

pub async fn show_directory_listing(path: &str, items: &[String]) {
    if let Some(display) = get_realtime_display() {
        display.show_directory_listing(path, items).await;
    }
}

pub async fn show_file_preview(file_path: &str, preview: &str) {
    if let Some(display) = get_realtime_display() {
        display.show_file_preview(file_path, preview).await;
    }
}