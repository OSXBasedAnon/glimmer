use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

/// Claude Code-style thinking display with reasoning


pub struct ThinkingDisplay {
    is_active: Arc<AtomicBool>,
    current_thought: Arc<std::sync::Mutex<String>>,
    start_time: Instant,
}

impl ThinkingDisplay {
    pub fn new() -> Self {
        Self {
            is_active: Arc::new(AtomicBool::new(false)),
            current_thought: Arc::new(std::sync::Mutex::new(String::new())),
            start_time: Instant::now(),
        }
    }

    /// Start thinking display with reasoning steps
    pub async fn start_thinking(&self, initial_thought: &str) -> Result<ThinkingHandle> {
        // RATATUI MODE: Use PersistentStatusBar instead of direct terminal manipulation
        // Update the persistent status bar which integrates with ratatui
        PersistentStatusBar::update_status(initial_thought);
        
        self.is_active.store(true, Ordering::Relaxed);
        
        if let Ok(mut current_thought) = self.current_thought.lock() {
            *current_thought = initial_thought.to_string();
        }

        Ok(ThinkingHandle {
            is_active: Arc::clone(&self.is_active),
            current_thought: Arc::clone(&self.current_thought),
            start_time: self.start_time,
            // No animation task in ratatui mode
        })
    }

    /// Start thinking display for function calls (Claude Code style)
    pub async fn start_function_thinking(&self, function_name: &str, reasoning: &str) -> Result<ThinkingHandle> {
        let thought = format!("{}({})", function_name, reasoning);
        self.start_thinking(&thought).await
    }

    /// Start thinking display for file operations
    pub async fn start_file_thinking(&self, operation: &str, file_path: &str) -> Result<ThinkingHandle> {
        let thought = format!("{} {} → analyzing file", operation, file_path);
        self.start_thinking(&thought).await
    }

    /// Start thinking display for research operations
    pub async fn start_research_thinking(&self, query: &str) -> Result<ThinkingHandle> {
        let thought = format!("Researching '{}' → fetching content", query);
        self.start_thinking(&thought).await
    }

    /// Update status during thinking (for Gemini API integration)
    pub fn update_status(&self, status: &str) {
        if let Ok(mut current_thought) = self.current_thought.lock() {
            *current_thought = status.to_string();
        }
        // Also update the persistent status bar
        PersistentStatusBar::update_status(status);
    }

    /// Update status with token information (displayed in grey after timing)
    pub fn update_status_with_tokens(&self, status: &str, input_tokens: u32, output_tokens: Option<u32>) {
        if let Ok(mut current_thought) = self.current_thought.lock() {
            let token_info = if let Some(out_tokens) = output_tokens {
                format!("{}→{}", input_tokens, out_tokens)
            } else {
                format!("{}", input_tokens)
            };
            // Store both status and token info - will be formatted in display loop
            *current_thought = format!("{}|TOKENS:{}", status, token_info);
        }
        // Also update the persistent status bar
        PersistentStatusBar::update_status_with_tokens(status, input_tokens, output_tokens);
    }

    /// Finish thinking display
    pub fn finish(&self) {
        self.is_active.store(false, Ordering::Relaxed);
        
        // Set persistent status bar back to resting
        PersistentStatusBar::set_resting();
        
        // Skip direct terminal manipulation in ratatui mode to avoid UI interference
        std::thread::sleep(Duration::from_millis(100));
    }

    /// Finish with error message
    pub fn finish_with_error(&self, error: &str) {
        if let Ok(mut current_thought) = self.current_thought.lock() {
            *current_thought = error.to_string();
        }
        
        std::thread::sleep(Duration::from_millis(500)); // Show error briefly
        
        self.is_active.store(false, Ordering::Relaxed);
        // Skip direct terminal manipulation in ratatui mode to avoid UI interference
        std::thread::sleep(Duration::from_millis(100));
    }
}

pub struct ThinkingHandle {
    is_active: Arc<AtomicBool>,
    current_thought: Arc<std::sync::Mutex<String>>,
    start_time: Instant,
    // No animation task in ratatui mode - status handled by PersistentStatusBar
}

impl ThinkingHandle {
    /// Create a dummy handle for fallback cases
    pub fn new_dummy() -> Self {
        Self {
            is_active: Arc::new(AtomicBool::new(false)),
            current_thought: Arc::new(std::sync::Mutex::new(String::new())),
            start_time: Instant::now(),
            // No animation task needed
        }
    }
    /// Update the thinking text (like Claude Code's reasoning steps)
    pub fn update_thought(&self, new_thought: &str) {
        if let Ok(mut thought) = self.current_thought.lock() {
            *thought = new_thought.to_string();
        }
    }

    /// Show progression of thinking steps (Claude Code style)
    pub fn progress_thought(&self, step: &str, detail: &str) {
        let thought = format!("{}({})", step, detail);
        self.update_thought(&thought);
    }

    /// Finish thinking with a result summary
    pub fn finish_with_summary(self, _summary: &str) {
        self.is_active.store(false, Ordering::Relaxed);
        
        // Set persistent status bar back to resting
        PersistentStatusBar::set_resting();
        
        // Wait for cleanup
        std::thread::sleep(Duration::from_millis(100));
        
        let elapsed = self.start_time.elapsed();
        let _elapsed_display = if elapsed.as_millis() < 1000 {
            format!("{}ms", elapsed.as_millis())
        } else {
            format!("{:.1}s", elapsed.as_secs_f64())
        };
        
        // Skip direct terminal manipulation in ratatui mode to avoid UI interference
    }

    /// Finish thinking silently
    pub fn finish(self) {
        self.is_active.store(false, Ordering::Relaxed);
        
        // Set persistent status bar back to resting
        PersistentStatusBar::set_resting();
        
        // Skip direct terminal manipulation in ratatui mode to avoid UI interference
        std::thread::sleep(Duration::from_millis(100));
    }

    /// Finish with an error message
    pub fn finish_with_error(self, _error: &str) {
        self.is_active.store(false, Ordering::Relaxed);
        
        // Set persistent status bar back to resting
        PersistentStatusBar::set_resting();
        
        std::thread::sleep(Duration::from_millis(100));
        // Skip direct terminal manipulation in ratatui mode to avoid UI interference
    }
}

impl Drop for ThinkingHandle {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
}

/// Persistent status bar that's always visible
static STATUS_BAR_ACTIVE: AtomicBool = AtomicBool::new(false);
static STATUS_BAR_MESSAGE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static STATUS_BAR_TOKENS: std::sync::Mutex<Option<(u32, Option<u32>)>> = std::sync::Mutex::new(None);
static STATUS_BAR_RESPONSE_TIME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static CHAT_INPUT_ACTIVE: AtomicBool = AtomicBool::new(false);
static SPINNER_IDX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static MEMORY_PERCENTAGE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static LAST_TYPING_TIME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Real-time AI reasoning capture for UI display
static CURRENT_AI_THINKING: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
static AI_INTERNAL_PROMPT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
static AI_REASONING_STEPS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());


pub struct PersistentStatusBar;

impl PersistentStatusBar {
    /// Start the persistent status bar that's always visible
    pub fn start() {
        if STATUS_BAR_ACTIVE.load(Ordering::Relaxed) {
            return; // Already running
        }
        
        STATUS_BAR_ACTIVE.store(true, Ordering::Relaxed);
        
        // Set initial idle state
        STATUS_BAR_MESSAGE.store(0, std::sync::atomic::Ordering::Relaxed); // 0 = "Resting"
        // The drawing loop is now managed by the main chat handler
        // to allow for a fully integrated UI.
    }

    /// Get the formatted string for the status bar line.
    pub fn get_status_bar_ui<'a>() -> Line<'a> {
        let spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        
        // Simple numeric state system - EXACTLY like the working 't' key approach
        let state = STATUS_BAR_MESSAGE.load(std::sync::atomic::Ordering::Relaxed);
        let (message, is_resting) = match state {
            0 => ("Resting", true),
            1 => ("Thinking", false),
            2 => ("Processing", false), 
            3 => ("Analyzing", false),
            4 => ("Reasoning", false),
            5 => ("Typing", false),
            6 => ("Complete", true),
            7 => ("Error", true),
            8 => ("No response", true),
            _ => ("Working", false), // Fallback
        };
        
        let spinner_idx = if is_resting {
            SPINNER_IDX.load(Ordering::Relaxed) % spinners.len() // Keep it moving slightly even when resting
        } else {
            SPINNER_IDX.fetch_add(1, Ordering::Relaxed) % spinners.len()
        };

        let memory_pct = MEMORY_PERCENTAGE.load(std::sync::atomic::Ordering::Relaxed);
        let response_time_ms = STATUS_BAR_RESPONSE_TIME.load(std::sync::atomic::Ordering::Relaxed);
        let tokens = STATUS_BAR_TOKENS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        
        let time_display = if response_time_ms > 0 {
            let seconds = response_time_ms as f64 / 1000.0;
            format!(" ({:.1}s)", seconds)
        } else {
            String::new()
        };
        
        let token_display = if let Some((input, output)) = tokens {
            if let Some(out) = output {
                format!(" mem {}% tokens {}→{}{}", memory_pct, input, out, time_display)
            } else {
                format!(" mem {}% tokens {}{}", memory_pct, input, time_display)
            }
        } else {
            format!(" mem {}%{}", memory_pct, time_display)
        };

        let mut spans = vec![
            Span::styled("⌘ ", Style::default().fg(Color::Rgb(203, 166, 247))),
            Span::styled(message, Style::default().fg(Color::Rgb(167, 139, 250))),
            Span::raw(" "),
            Span::styled(format!("{} ", spinners[spinner_idx]), Style::default().fg(Color::Rgb(167, 139, 250))),
            Span::styled(token_display, Style::default().fg(Color::DarkGray)),
        ];
        
        // Add ESC hint when AI is working (not resting)
        if !is_resting {
            spans.push(Span::styled(" (esc to interject)", Style::default().fg(Color::Rgb(100, 100, 100))));
        }
        
        Line::from(spans)
    }
    
    /// Update the status message - EXACTLY like the working 't' key approach
    pub fn update_status(message: &str) {
        let state = match message {
            "Resting" => 0,
            "Thinking" => 1, 
            "Processing" => 2,
            "Analyzing" => 3,
            "Reasoning" => 4,
            "Typing" => 5,
            "Complete" => 6,
            "Error" => 7,
            "No response" => 8,
            _ => 2, // Default to "Processing"
        };
        STATUS_BAR_MESSAGE.store(state, std::sync::atomic::Ordering::Relaxed);
        
        // Don't capture status updates for reasoning display - they're different things
    }
    
    /// Update status with token information
    pub fn update_status_with_tokens(message: &str, input_tokens: u32, output_tokens: Option<u32>) {
        // Update the status using the new atomic system
        Self::update_status(message);
        
        // Update tokens
        if let Ok(mut tokens) = STATUS_BAR_TOKENS.lock() {
            *tokens = Some((input_tokens, output_tokens));
        }
    }
    
    /// Update status with token information and response time
    pub fn update_status_with_tokens_and_timing(message: &str, input_tokens: u32, output_tokens: Option<u32>, response_time_ms: u32) {
        // Format with token counts in the exact format specified
        let token_format = if let Some(out_tokens) = output_tokens {
            format!("({}\u{2192}{})", input_tokens, out_tokens)
        } else {
            format!("({})", input_tokens)
        };
        
        // Don't reformat the message - it's already properly formatted from realtime.rs
        let formatted_message = format!("{} {}", message, token_format);
        
        // Update the AI thinking display with formatted message
        Self::set_ai_thinking(&formatted_message);
        
        // Update tokens
        if let Ok(mut tokens) = STATUS_BAR_TOKENS.lock() {
            *tokens = Some((input_tokens, output_tokens));
        }
        
        // Update response time
        STATUS_BAR_RESPONSE_TIME.store(response_time_ms, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Update response time
    pub fn update_response_time(response_time_ms: u32) {
        STATUS_BAR_RESPONSE_TIME.store(response_time_ms, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Set to resting state (preserves token information)
    pub fn set_resting() {
        STATUS_BAR_MESSAGE.store(0, std::sync::atomic::Ordering::Relaxed); // 0 = "Resting"
        // Note: We DON'T clear STATUS_BAR_TOKENS here to preserve token info
    }
    
    /// Stop the persistent status bar
    pub fn stop() {
        STATUS_BAR_ACTIVE.store(false, Ordering::Relaxed);
    }
    
    /// Show chat input prompt in status bar
    pub fn show_chat_input() {
        CHAT_INPUT_ACTIVE.store(true, Ordering::Relaxed);
    }
    
    /// Hide chat input prompt and return to status display
    pub fn hide_chat_input() {
        CHAT_INPUT_ACTIVE.store(false, Ordering::Relaxed);
    }
    
    /// Update memory percentage
    pub fn update_memory_percentage(percentage: u8) {
        MEMORY_PERCENTAGE.store(percentage.min(100), std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Get current memory percentage
    pub fn get_memory_percentage() -> u8 {
        MEMORY_PERCENTAGE.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Set the current AI thinking content (actual prompts/reasoning) - rate limited to prevent UI spam
    pub fn set_ai_thinking(content: &str) {
        // Filter out status messages but keep actual AI reasoning
        let formatted_content = if content.contains("●") || content.contains("⎿") {
            // Skip status indicator messages - they belong in chat, not thinking display
            return;
        } else if content == "Complete" {
            // Skip generic "Complete" status
            return;
        } else if content.is_empty() {
            // Allow clearing the thinking display
            String::new()
        } else {
            // Keep all substantial AI reasoning content
            content.to_string()
        };
        
        // Rate limit status updates to prevent UI corruption but allow real-time feel
        static LAST_UPDATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static LAST_CONTENT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
        
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
        let last = LAST_UPDATE.load(std::sync::atomic::Ordering::Relaxed);
        
        // Check if content has actually changed to avoid spam
        if let Ok(mut last_content) = LAST_CONTENT.try_lock() {
            if *last_content == formatted_content {
                return; // Same content, skip update
            }
            *last_content = formatted_content.clone();
        }
        
        // Reduced rate limiting for better real-time feel - only 50ms between updates
        if now - last < 50 {
            // Allow high-priority updates through immediately
            if !formatted_content.contains("Strategy:") {
                return;
            }
        }
        LAST_UPDATE.store(now, std::sync::atomic::Ordering::Relaxed);
        
        if let Ok(mut thinking) = CURRENT_AI_THINKING.lock() {
            // Avoid duplicate consecutive messages
            if *thinking != formatted_content {
                *thinking = formatted_content;
            }
        }
    }
    
    /// Get current AI thinking content
    pub fn get_ai_thinking() -> String {
        if let Ok(thinking) = CURRENT_AI_THINKING.lock() {
            thinking.clone()
        } else {
            String::new()
        }
    }
    
    /// Set the AI's internal prompt for debugging
    pub fn set_ai_internal_prompt(prompt: &str) {
        if let Ok(mut internal_prompt) = AI_INTERNAL_PROMPT.lock() {
            *internal_prompt = prompt.to_string();
        }
    }
    
    /// Get the AI's internal prompt
    pub fn get_ai_internal_prompt() -> String {
        if let Ok(internal_prompt) = AI_INTERNAL_PROMPT.lock() {
            internal_prompt.clone()
        } else {
            String::new()
        }
    }
    
    /// Add a colored reasoning step with proper ratatui formatting
    pub fn add_colored_reasoning_step(indicator_color: (u8, u8, u8), text: &str) {
        // Format for ratatui with color information embedded as a special marker
        let formatted = format!("●<color:{},{},{}>{}", indicator_color.0, indicator_color.1, indicator_color.2, text);
        Self::add_reasoning_step(&formatted);
    }
    
    /// Add a reasoning step to the current thinking - ONLY for ● status indicators that go to chat history
    pub fn add_reasoning_step(step: &str) {
        // Strip ANSI codes first to check for ● or ⎿
        let clean_step = strip_ansi_codes_simple(step);
        
        // ONLY handle UI operation updates that contain ● or ⎿ - these go to chat history
        if clean_step.contains("●") || clean_step.contains("⎿") {
            // This is a real-time UI update - add directly to reasoning steps without rate limiting
            if let Ok(mut steps) = AI_REASONING_STEPS.lock() {
                steps.push(step.to_string());
            }
            return;
        }
        
        // DO NOT add regular AI reasoning here - that should use set_ai_thinking instead
        // This function is specifically for ● status indicators in chat history
    }
    
    /// Get the latest reasoning steps  
    pub fn get_latest_reasoning_steps() -> Vec<String> {
        if let Ok(steps) = AI_REASONING_STEPS.lock() {
            steps.clone()
        } else {
            Vec::new()
        }
    }
    
    /// Update typing status with timeout detection
    pub fn update_typing_status() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        LAST_TYPING_TIME.store(now, std::sync::atomic::Ordering::Relaxed);
        Self::update_status("Typing");
    }
    
    /// Check if user is still typing (called from the main UI loop)
    pub fn check_typing_timeout() {
        let current_state = STATUS_BAR_MESSAGE.load(std::sync::atomic::Ordering::Relaxed);
        
        // Only check timeout if currently in "Typing" state (state 5)
        if current_state == 5 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            let last_typing = LAST_TYPING_TIME.load(std::sync::atomic::Ordering::Relaxed);
            
            // If more than 2 seconds since last typing, go back to resting
            if now > last_typing && (now - last_typing) >= 2 {
                Self::set_resting();
            }
        }
    }
    
    /// Clear all AI thinking data (called when starting new conversation)
    pub fn clear_ai_thinking() {
        if let Ok(mut thinking) = CURRENT_AI_THINKING.lock() {
            thinking.clear();
        }
        if let Ok(mut prompt) = AI_INTERNAL_PROMPT.lock() {
            prompt.clear();
        }
        if let Ok(mut steps) = AI_REASONING_STEPS.lock() {
            steps.clear();
        }
    }
    
}

/// Simple ANSI code stripper for internal use
fn strip_ansi_codes_simple(text: &str) -> String {
    // Remove ANSI escape sequences like \x1b[38;2;255;204;92m and \x1b[0m
    let mut result = String::new();
    let mut in_escape = false;
    let mut chars = text.chars();
    
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            in_escape = true;
            continue;
        }
        
        if in_escape {
            if ch == 'm' {
                in_escape = false;
            }
            continue;
        }
        
        result.push(ch);
    }
    
    result
}

/// Helper functions for common thinking patterns

pub async fn think_about_request(request: &str) -> Result<ThinkingHandle> {
    let display = ThinkingDisplay::new();
    display.start_thinking(&format!("Analyzing '{}'", request)).await
}

pub async fn think_about_function_call(func_name: &str, purpose: &str) -> Result<ThinkingHandle> {
    let display = ThinkingDisplay::new();
    display.start_function_thinking(func_name, purpose).await
}

pub async fn think_about_file_operation(file_path: &str, operation: &str) -> Result<ThinkingHandle> {
    let display = ThinkingDisplay::new();
    display.start_file_thinking(operation, file_path).await
}

/// Simple function to start thinking with a message (Claude Code style)
pub async fn show_thinking_spinner() -> ThinkingHandle {
    let display = ThinkingDisplay::new();
    match display.start_thinking("Processing your request").await {
        Ok(handle) => handle,
        Err(_) => ThinkingHandle::new_dummy(),
    }
}

/// Clear any thinking display
pub fn clear_thinking() {
    // Skip direct terminal manipulation in ratatui mode to avoid UI interference
}

/// Wrap async operations with thinking display
pub async fn with_thinking<F, T>(thought: &str, future: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let thinking = think_about_request(thought).await?;
    
    match future.await {
        Ok(result) => {
            thinking.finish_with_summary("Completed successfully");
            Ok(result)
        },
        Err(e) => {
            thinking.finish_with_error(&format!("Failed: {}", e));
            Err(e)
        }
    }
}

/// Advanced thinking for multi-step operations
pub struct MultiStepThinking {
    handle: ThinkingHandle,
    current_step: usize,
    total_steps: usize,
}

impl MultiStepThinking {
    pub async fn new(operation: &str, total_steps: usize) -> Result<Self> {
        let display = ThinkingDisplay::new();
        let handle = display.start_thinking(&format!("{} (step 1/{}) → preparing", operation, total_steps)).await?;
        
        Ok(Self {
            handle,
            current_step: 1,
            total_steps,
        })
    }

    pub fn next_step(&mut self, step_description: &str) {
        self.current_step += 1;
        let thought = format!("step {}/{} → {}", self.current_step, self.total_steps, step_description);
        self.handle.update_thought(&thought);
    }

    pub fn finish_with_result(self, result: &str) {
        self.handle.finish_with_summary(&format!("Completed {} → {}", self.total_steps, result));
    }
}

/// Show detailed reasoning like Claude Code does
// Removed duplicate functions - using the ones defined earlier

pub fn display_reasoning(reasoning: &[&str]) {
    // Use status bar instead of println to avoid ratatui UI scrambling
    if reasoning.is_empty() {
        PersistentStatusBar::set_ai_thinking("Reasoning...");
    } else {
        let reasoning_text = reasoning.iter().enumerate()
            .map(|(i, reason)| format!("{}. {}", i + 1, reason))
            .collect::<Vec<_>>()
            .join(" | ");
        PersistentStatusBar::set_ai_thinking(&format!("Reasoning: {}", reasoning_text));
    }
}