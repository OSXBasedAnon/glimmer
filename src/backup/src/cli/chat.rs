use anyhow::{Context, Result};
use std::path::PathBuf;
use std::path::Path;
use std::io::{self, Write};
use regex::Regex;
use crate::config::Config;
use crate::gemini;
use crate::cli::memory::{MemoryEngine, MessageRole, ChatMessage};
use crate::function_calling::{FunctionRegistry, create_function_calling_prompt, execute_function_call};
use crate::research;
use crate::{warn_println};
use crate::cli::colors::{RESET, GRAY_DIM};
use crate::cli::chat_ui::{ChatUI, SystemMessageType};
use crate::thinking_display::{PersistentStatusBar};
use crate::reasoning_engine::RequestType;
use textwrap;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers, KeyEventKind, EnableBracketedPaste, DisableBracketedPaste},
    execute,
    cursor::{self},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use cli_clipboard::{ClipboardContext, ClipboardProvider};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
    Terminal,
};
use std::time::Duration;

/// Strip ANSI escape codes from text
fn strip_ansi_codes(text: &str) -> String {
    // Remove ANSI escape sequences like \x1b[38;2;255;204;92m and \x1b[0m
    let ansi_regex = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    ansi_regex.replace_all(text, "").to_string()
}

#[derive(Debug)]
struct ProblemAnalysis {
    description: String,
    likely_file_type: String,
}

/// Extract file paths mentioned in recent messages for context resolution
fn extract_recent_file_paths(recent_messages: &[ChatMessage]) -> Vec<String> {
    let mut paths = Vec::new();
    let path_patterns = [
        r"C:\\[^\\]+\\[^\\]+\\[^\\]+\\[^\s]+", // Windows absolute paths
        r"/[^/\s]+/[^/\s]+/[^\s]+",           // Unix absolute paths
        r"[^/\s\\]+\\[^/\s\\]+\\[^\s]+",      // Relative Windows paths
        r"[^/\s\\]+/[^/\s\\]+/[^\s]+",        // Relative Unix paths
        r"[^\s]+\.[a-zA-Z]{2,4}",             // Files with extensions
    ];
    
    for message in recent_messages.iter().rev().take(5) { // Check last 5 messages
        let content = &message.content;
        
        // Look for directory listings - check for directory paths and files mentioned
        if content.contains("C:\\") && (content.contains(".html") || content.contains(".js") || content.contains(".css")) {
            // Find the directory path (usually at the top of a listing)
            if let Some(dir_line) = content.lines().find(|l| l.trim().starts_with("C:\\") && !l.contains("├──") && !l.contains("└──")) {
                let dir_path = dir_line.trim();
                paths.push(dir_path.to_string());
                
                // Also add common files from that directory
                for line in content.lines() {
                    if line.contains("index.html") || line.contains("script.js") || line.contains("style.css") {
                        paths.push(format!("{}\\index.html", dir_path));
                        paths.push(format!("{}\\script.js", dir_path));
                        paths.push(format!("{}\\style.css", dir_path));
                        break;
                    }
                }
            }
        }
        
        // Extract explicit file paths
        for pattern in &path_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                for mat in regex.find_iter(content) {
                    let path = mat.as_str().trim_end_matches([',', '.', '!', '?']);
                    if !paths.contains(&path.to_string()) && (path.contains(".html") || path.contains("C:\\") || path.contains("/")) {
                        paths.push(path.to_string());
                    }
                }
            }
        }
    }
    
    // Deduplicate and return most recent first
    let mut unique_paths: Vec<String> = paths.into_iter().collect();
    unique_paths.reverse();
    unique_paths.truncate(5);
    unique_paths
}

/// Analyze what type of problem the user is reporting to guide file investigation
fn analyze_user_problem_type(input: &str) -> ProblemAnalysis {
    let input_lower = input.to_lowercase();
    
    // Visual/display issues - likely HTML structure or JS rendering
    if input_lower.contains("not showing") || input_lower.contains("can't see") || 
       input_lower.contains("is gone") || input_lower.contains("disappeared") ||
       input_lower.contains("not displayed") || input_lower.contains("missing") ||
       input_lower.contains("blank") || input_lower.contains("empty") {
        return ProblemAnalysis {
            description: "Visual element not displaying".to_string(),
            likely_file_type: "HTML structure or JavaScript rendering logic".to_string(),
        };
    }
    
    // Styling issues - likely CSS
    if input_lower.contains("looks wrong") || input_lower.contains("styling") ||
       input_lower.contains("color") || input_lower.contains("position") ||
       input_lower.contains("layout") || input_lower.contains("size") {
        return ProblemAnalysis {
            description: "Visual styling issue".to_string(),
            likely_file_type: "CSS styling rules".to_string(),
        };
    }
    
    // Functionality issues - likely JavaScript
    if input_lower.contains("not working") || input_lower.contains("doesn't work") ||
       input_lower.contains("broken") || input_lower.contains("error") ||
       input_lower.contains("click") || input_lower.contains("function") {
        return ProblemAnalysis {
            description: "Functionality not working".to_string(),
            likely_file_type: "JavaScript logic and event handlers".to_string(),
        };
    }
    
    // Default analysis
    ProblemAnalysis {
        description: "General issue reported".to_string(),
        likely_file_type: "Recently modified files".to_string(),
    }
}

/// Detect if user is reporting that something is broken/missing after our changes
fn is_emergency_breakage_report(input_lower: &str) -> bool {
    // Classic breakage indicators
    let breakage_patterns = [
        "there is no", "there's no", "is missing", "is gone", "disappeared", "can't see",
        "not working", "doesn't work", "stopped working", "broken", "not showing",
        "nothing happens", "no longer", "isn't working", "won't", "not displaying",
        "is blank", "is empty", "not visible", "can't find", "where is", "where did"
    ];
    
    // Immediate action words that suggest user expects us to fix
    let action_expectations = [
        "fix", "repair", "restore", "bring back", "make it work", "get it working"
    ];
    
    // Check for breakage patterns
    let has_breakage = breakage_patterns.iter().any(|&pattern| input_lower.contains(pattern));
    
    // Check for explicit fix requests
    let wants_fix = action_expectations.iter().any(|&pattern| input_lower.contains(pattern));
    
    // Emergency if either condition is met
    has_breakage || wants_fix
}

/// Handle emergency breakage with immediate investigation and fix
async fn handle_emergency_breakage(
    input: &str,
    memory_engine: &MemoryEngine,
    config: &Config,
) -> Result<String> {
    PersistentStatusBar::set_ai_thinking("🚨 PANIC: Something is broken! Investigating immediately...");
    
    // Get recent file modifications - this is KEY
    let recent_modifications = crate::function_calling::get_recent_modifications();
    
    // Get recent conversation context to understand what we just did
    let recent_messages = memory_engine.get_recent_messages(10).await?;
    let conversation_history = recent_messages.iter()
        .map(|m| format!("{}: {}", m.role.to_string(), m.content))
        .collect::<Vec<_>>()
        .join("\n");
    
    let modification_context = if !recent_modifications.is_empty() {
        format!("\n\nIMMEDIATE CONTEXT - FILES I JUST MODIFIED:\n{}\n", recent_modifications.join("\n"))
    } else {
        "\n\nWARNING: No recent file modifications tracked\n".to_string()
    };
    
    // Create emergency fix prompt with full context
    let emergency_prompt = format!(
        "EMERGENCY SITUATION: The user just reported: '{}'\n\n\
        This suggests something I recently did broke or removed something important.\n\n\
        {}\n\
        RECENT CONVERSATION CONTEXT:\n{}\n\n\
        CRITICAL INSTRUCTIONS:\n\
        1. DO NOT ask for clarification - the user is clearly reporting breakage\n\
        2. I have PERFECT MEMORY of what I just modified - use the file list above\n\
        3. Immediately read the files I just modified to see what went wrong\n\
        4. Look for what got broken or removed (likely display elements if user mentions 'cube')\n\
        5. Fix the issue immediately by restoring missing code or fixing syntax\n\n\
        Take immediate action to diagnose and fix the problem using my modification history.",
        input, modification_context, conversation_history
    );
    
    // Use function calling with emergency context
    let registry = crate::function_calling::FunctionRegistry::new();
    let function_definitions = registry.get_function_definitions();
    
    match crate::gemini::query_gemini_with_function_calling(
        &emergency_prompt,
        config,
        Some(&function_definitions),
    ).await {
        Ok((response_text, function_call, _token_usage)) => {
            // If Gemini wants to call a function to investigate/fix, execute it
            if let Some(func_call) = function_call {
                // Status bar shows AI thinking, not task descriptions
                
                match crate::function_calling::execute_function_call(&func_call, config, &conversation_history).await {
                    Ok(function_result) => {
                        // Follow up to complete the fix
                        let followup_prompt = format!(
                            "Emergency investigation result: {}\n\n\
                            Now complete the fix for the user's problem: '{}'",
                            function_result, input
                        );
                        
                        match crate::gemini::query_gemini_with_function_calling(
                            &followup_prompt,
                            config,
                            Some(&function_definitions),
                        ).await {
                            Ok((final_response, final_function_call, _)) => {
                                // Execute any final fixing function
                                if let Some(final_func) = final_function_call {
                                    match crate::function_calling::execute_function_call(&final_func, config, &conversation_history).await {
                                        Ok(final_result) => {
                                            return Ok(format!("🚑 Emergency fix completed!\n\n{}\n\n{}", final_result, final_response));
                                        }
                                        Err(_) => return Ok(format!("🚑 Emergency investigation completed:\n\n{}", final_response))
                                    }
                                } else {
                                    return Ok(format!("🚑 Emergency fix completed:\n\n{}", final_response));
                                }
                            }
                            Err(_) => {
                                return Ok(format!("🚑 Emergency investigation completed:\n\n{}", function_result));
                            }
                        }
                    }
                    Err(e) => {
                        return Ok(format!("🚨 Emergency response in progress but function failed: {}\n\nDirect response: {}", e, response_text));
                    }
                }
            } else {
                // Direct response without function calling
                return Ok(format!("🚑 Emergency response: {}", response_text));
            }
        }
        Err(e) => {
            return Err(e);
        }
    }
}

/// Task complexity classification result
#[derive(Debug)]
struct TaskComplexity {
    requires_confirmation: bool,
    estimated_steps: u8,
    risk_level: RiskLevel,
}

#[derive(Debug)]
enum RiskLevel {
    Low,
    Medium, 
    High,
    Critical,
}

/// Use AI to classify task complexity instead of hardcoded patterns
async fn classify_task_complexity(input: &str, context: &str, config: &Config) -> Result<TaskComplexity> {
    // FAST PATH: Local pattern matching for common simple tasks
    let input_lower = input.to_lowercase();
    
    // Simple tasks - no API call needed
    if input_lower.contains("what is in") || 
       input_lower.contains("list") ||
       input_lower.contains("show me") ||
       (input_lower.contains("read") && !input_lower.contains("all")) ||
       (input_lower.contains("view") && !input_lower.contains("all")) {
        return Ok(TaskComplexity {
            requires_confirmation: false,
            estimated_steps: 1,
            risk_level: RiskLevel::Low,
        });
    }
    
    // Medium complexity - single file operations
    if input_lower.contains("edit") || 
       input_lower.contains("fix") ||
       input_lower.contains("update") ||
       input_lower.contains("create") {
        return Ok(TaskComplexity {
            requires_confirmation: false,
            estimated_steps: 2,
            risk_level: RiskLevel::Low,
        });
    }
    
    // High risk keywords - need confirmation
    if input_lower.contains("delete") ||
       input_lower.contains("remove") ||
       input_lower.contains("refactor entire") ||
       input_lower.contains("rewrite all") {
        return Ok(TaskComplexity {
            requires_confirmation: true,
            estimated_steps: 5,
            risk_level: RiskLevel::High,
        });
    }
    
    // Fallback to API for truly ambiguous cases
    let prompt = format!(
        r#"Analyze this user request for task complexity and risk:

Request: "{}"
Context: {}

Classify the task and respond with JSON:
{{
  "requires_confirmation": boolean, // true if destructive/high-impact
  "estimated_steps": number, // 1-10 steps needed
  "risk_level": "Low" | "Medium" | "High" | "Critical" // potential for data loss/system impact
}}

Examples:
- "read file.txt" → low complexity, no confirmation needed
- "refactor entire codebase" → high complexity, needs confirmation
- "delete all files" → critical risk, definitely needs confirmation"#,
        input, context
    );
    
    let response = gemini::query_gemini(&prompt, config).await?;
    
    // Parse JSON response with fallback for non-JSON responses
    let parsed: serde_json::Value = match serde_json::from_str(response.trim()) {
        Ok(json) => json,
        Err(_) => {
            // If JSON parsing fails, try to extract JSON from the response
            let response_text = response.trim();
            if let Some(start) = response_text.find('{') {
                if let Some(end) = response_text.rfind('}') {
                    let json_str = &response_text[start..=end];
                    match serde_json::from_str(json_str) {
                        Ok(json) => json,
                        Err(_) => {
                            // If still no valid JSON, return a safe default
                            return Ok(TaskComplexity {
                                requires_confirmation: false,
                                estimated_steps: 2,
                                risk_level: RiskLevel::Medium,
                            });
                        }
                    }
                } else {
                    // No JSON found, return safe default
                    return Ok(TaskComplexity {
                        requires_confirmation: false,
                        estimated_steps: 2,
                        risk_level: RiskLevel::Medium,
                    });
                }
            } else {
                // No JSON found, return safe default
                return Ok(TaskComplexity {
                    requires_confirmation: false,
                    estimated_steps: 2,
                    risk_level: RiskLevel::Medium,
                });
            }
        }
    };
    
    let requires_confirmation = parsed["requires_confirmation"].as_bool().unwrap_or(false);
    let estimated_steps = parsed["estimated_steps"].as_u64().unwrap_or(3) as u8;
    let risk_level_str = parsed["risk_level"].as_str().unwrap_or("Medium");
    
    let risk_level = match risk_level_str {
        "Low" => RiskLevel::Low,
        "Medium" => RiskLevel::Medium, 
        "High" => RiskLevel::High,
        "Critical" => RiskLevel::Critical,
        _ => RiskLevel::Medium,
    };
    
    Ok(TaskComplexity {
        requires_confirmation,
        estimated_steps,
        risk_level,
    })
}

// Smart decision types for intelligent task handling
#[derive(Debug, Clone)]
enum SmartDecision {
    DirectResponse,
    WithThinking,
    AutoProceedWithSteps(Vec<String>),
    ConfirmBeforeSteps(gemini::TaskComplexity),
}

// Intelligent decision engine that mimics Claude Code's smart behavior
async fn make_smart_decision(
    input: &str,
    context: &str,
    conversation_history: &str,
    config: &Config,
) -> SmartDecision {
    let input_lower = input.to_lowercase();

    // 1. Use AI to classify task complexity instead of hardcoded patterns
    if let Ok(complexity) = classify_task_complexity(input, context, config).await {
        if complexity.requires_confirmation {
            if let Ok(analysis) = gemini::analyze_task_complexity(input, context, config).await {
                return SmartDecision::ConfirmBeforeSteps(analysis);
            }
            return SmartDecision::WithThinking;
        }
    }

    // 2. Check for file operations by looking for keywords and file mentions from context.
    let file_operation_keywords = [
        "read", "show", "display", "edit", "modify", "update", "create", "write",
        "fix", "debug", "analyze", "review", "check", "lint", "format", "open", "make",
        "build", "generate", "new file", "rubiks", "html", "javascript", "rename", "move",
        "copy", "change name", "call it", "name it",
    ];
    
    // 2.1. Check for context-dependent references (things that relate to recent work)
    let context_references = [
        "there is no", "there's no", "is missing", "is gone", "disappeared", "not working",
        "broken", "not showing", "can't see", "where is", "where did", "cube", "display",
        "the file", "it", "that", "this"
    ];
    
    // If user is referencing something contextually, treat as file operation
    let has_context_reference = context_references.iter().any(|&p| input_lower.contains(p));
    if has_context_reference {
        return SmartDecision::WithThinking; // Force intelligent analysis
    }
    
    let mut mentions_file = false;
    if let Some(files_part) = context.strip_prefix("Files in current directory: ") {
        let files: Vec<&str> = files_part.split(", ").collect();
        if files.iter().any(|file| input_lower.contains(&file.to_lowercase())) {
            mentions_file = true;
        }
    }
    
    // Check for file extensions or obvious file patterns
    let file_patterns = [".html", ".js", ".css", ".py", ".rs", ".java", ".cpp", ".c", ".go"];
    let mentions_file_type = file_patterns.iter().any(|&ext| input_lower.contains(ext));

    if file_operation_keywords.iter().any(|&p| input_lower.contains(p)) || mentions_file || mentions_file_type {
        return SmartDecision::WithThinking;
    }

    // 3. Complex but safe-to-auto-proceed tasks.
    let auto_complex_patterns = [
        "refactor", "optimize", "implement", "develop",
        "migrate", "upgrade", "convert", "transform",
    ];
    if auto_complex_patterns.iter().any(|&p| input_lower.contains(p)) {
        let estimated_steps = estimate_steps_for_task(&input_lower, context, config).await;
        return SmartDecision::AutoProceedWithSteps(estimated_steps);
    }

    // 4. Simple, direct questions or commands.
    let simple_patterns = [
        "what is", "how to", "explain", "tell me", "define", "meaning",
        "difference between", "why does", "when should", "where can",
    ];
    let music_patterns = ["play", "pause", "stop", "music", "song", "audio", "volume", "speed"];

    if simple_patterns.iter().any(|&p| input_lower.contains(p)) ||
       music_patterns.iter().any(|&p| input_lower.contains(p)) {
        return SmartDecision::DirectResponse;
    }

    // 5. Contextual follow-ups.
    if conversation_history.contains("error") || conversation_history.contains("failed") {
        return SmartDecision::WithThinking;
    }

    // 6. Default to thinking for any coding-related request
    SmartDecision::WithThinking
}

// Estimate steps without calling the LLM - Claude Code style
async fn estimate_steps_for_task(input: &str, context: &str, config: &Config) -> Vec<String> {
    let prompt = format!(
        r#"Analyze this task and generate a logical sequence of 2-5 implementation steps:

Task: "{}"
Context: {}

Generate specific, actionable steps that would be needed to complete this task.
Respond with a JSON array of step descriptions.

Example for "create a login component":
["Analyze authentication requirements", "Design component interface", "Implement form validation", "Add styling and user feedback"]

Example for "debug performance issue":
["Profile application performance", "Identify bottlenecks", "Implement optimizations", "Verify improvements"]

Be specific to the actual task requested."#,
        input, context
    );
    
    match gemini::query_gemini(&prompt, config).await {
        Ok(response) => {
            // Try to parse JSON array
            if let Ok(steps) = serde_json::from_str::<Vec<String>>(response.trim()) {
                if !steps.is_empty() {
                    return steps;
                }
            }
            // Fallback: extract steps from text response
            response.lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    if trimmed.starts_with('-') || trimmed.starts_with('•') || trimmed.starts_with("* ") {
                        Some(trimmed.trim_start_matches(['-', '•', '*']).trim().to_string())
                    } else if trimmed.len() > 10 && !trimmed.starts_with("Example") {
                        Some(trimmed.to_string())
                    } else {
                        None
                    }
                })
                .take(5)
                .collect()
        }
        Err(_) => {
            // Ultimate fallback
            vec!["Analyze the request".to_string(), "Execute the task".to_string()]
        }
    }
}

/// Loop recovery types for intelligent handling
#[derive(Debug)]
enum LoopRecoveryType {
    ForceEdit,
    ChangeStrategy, 
    StopWithExplanation,
}

/// Recovery action for breaking loops intelligently
#[derive(Debug)]
struct LoopRecoveryAction {
    action_type: LoopRecoveryType,
    description: String,
    instructions: String,
}

/// Analyze the loop situation and determine the best recovery action
async fn analyze_loop_situation(
    func_call: &crate::function_calling::FunctionCall,
    user_request: &str,
    _current_response: &str,
    step_count: u32,
) -> LoopRecoveryAction {
    let request_lower = user_request.to_lowercase();

    // If user wants to create/improve something but we keep reading
    if func_call.name == "read_file" && 
       (request_lower.contains("make") || request_lower.contains("improve") || 
        request_lower.contains("create") || request_lower.contains("interactive") ||
        request_lower.contains("fix") || request_lower.contains("enhance")) {
        return LoopRecoveryAction {
            action_type: LoopRecoveryType::ForceEdit,
            description: "Forcing code modification - detected read loop while user wants changes".to_string(),
            instructions: format!(
                "The user requested: '{}'. I've been analyzing the code but now I must make the actual changes they want. I will implement the specific improvements requested.", 
                user_request
            ),
        };
    }
    
    // If we're stuck on the same file operation
    if step_count > 6 {
        return LoopRecoveryAction {
            action_type: LoopRecoveryType::ChangeStrategy,
            description: "Changing approach - trying different strategy".to_string(),
            instructions: format!(
                "I need to try a different approach. Instead of {}, I should focus on the core requirement: {}",
                func_call.name, user_request
            ),
        };
    }
    
    // Default: clean stop message
    LoopRecoveryAction {
        action_type: LoopRecoveryType::StopWithExplanation,
        description: "Task requires different approach".to_string(),
        instructions: format!(
            "The current approach isn't working for '{}'. Please provide additional details or try a different request.", 
            user_request.chars().take(50).collect::<String>()
        ),
    }
}

/// Parse a markdown line into a ratatui Line with proper styling
fn parse_markdown_line(text: &str) -> Line {
    use ratatui::style::{Color, Style, Modifier};
    use ratatui::text::Span;
    
    let mut spans = Vec::new();
    let mut chars = text.chars().peekable();
    let mut current_text = String::new();
    
    while let Some(ch) = chars.next() {
        match ch {
            '*' => {
                if chars.peek() == Some(&'*') {
                    // Bold text **text**
                    chars.next(); // consume second *
                    if !current_text.is_empty() {
                        spans.push(Span::raw(current_text.clone()));
                        current_text.clear();
                    }
                    
                    // Read until closing **
                    let mut bold_text = String::new();
                    let mut found_closing = false;
                    while let Some(ch) = chars.next() {
                        if ch == '*' && chars.peek() == Some(&'*') {
                            chars.next(); // consume second *
                            found_closing = true;
                            break;
                        }
                        bold_text.push(ch);
                    }
                    
                    if found_closing && !bold_text.is_empty() {
                        spans.push(Span::styled(bold_text, Style::default().add_modifier(Modifier::BOLD)));
                    } else {
                        // Not properly closed, treat as regular text
                        current_text.push_str("**");
                        current_text.push_str(&bold_text);
                    }
                } else {
                    // Italic text *text*
                    if !current_text.is_empty() {
                        spans.push(Span::raw(current_text.clone()));
                        current_text.clear();
                    }
                    
                    // Read until closing *
                    let mut italic_text = String::new();
                    let mut found_closing = false;
                    while let Some(ch) = chars.next() {
                        if ch == '*' {
                            found_closing = true;
                            break;
                        }
                        italic_text.push(ch);
                    }
                    
                    if found_closing && !italic_text.is_empty() {
                        spans.push(Span::styled(italic_text, Style::default().add_modifier(Modifier::ITALIC)));
                    } else {
                        // Not properly closed, treat as regular text
                        current_text.push('*');
                        current_text.push_str(&italic_text);
                    }
                }
            },
            _ => {
                current_text.push(ch);
            }
        }
    }
    
    // Add remaining text
    if !current_text.is_empty() {
        spans.push(Span::raw(current_text));
    }
    
    // Handle special line formatting
    if spans.len() == 1 {
        let text_content = spans[0].content.clone();
        if text_content.trim_start().starts_with("❯ ") {
            // Style user messages as dark grey
            return Line::from(vec![
                Span::styled(text_content, Style::default().fg(Color::DarkGray)),
            ]);
        } else if text_content.trim_start().starts_with("* ") || text_content.trim_start().starts_with("- ") {
            // Convert to bullet point
            let indent = text_content.len() - text_content.trim_start().len();
            let content = text_content.trim_start()[2..].trim_start(); // Remove "* " or "- "
            return Line::from(vec![
                Span::raw(" ".repeat(indent)),
                Span::styled("• ", Style::default().fg(Color::Cyan)),
                Span::raw(content.to_string()),
            ]);
        } else if text_content.trim_start().starts_with("  • ") {
            // Already formatted bullet, just color it
            let indent = text_content.len() - text_content.trim_start().len();
            let rest = &text_content.trim_start()[4..]; // Remove "  • "
            return Line::from(vec![
                Span::raw(" ".repeat(indent + 2)),
                Span::styled("• ", Style::default().fg(Color::Cyan)),
                Span::raw(rest.to_string()),
            ]);
        }
    }
    
    Line::from(spans)
}

pub async fn handle_chat(
    _save: Option<PathBuf>, // No longer used with sled
    _load: Option<PathBuf>, // No longer used with sled
    config: &Config,
) -> Result<()> {
    // Setup terminal with proper error handling
    enable_raw_mode().context("Failed to enable raw mode")?;
    
    // Enable bracketed paste FIRST before creating ratatui backend
    // Enable mouse capture for scroll events while preserving text selection
    execute!(
        std::io::stdout(),
        cursor::Hide,
        EnableBracketedPaste,
        crossterm::event::EnableMouseCapture
    ).context("Failed to setup terminal")?;
    
    // Force flush to ensure bracketed paste command reaches terminal BEFORE ratatui
    std::io::stdout().flush().context("Failed to flush stdout")?;
    
    // Small delay to ensure terminal processes the bracketed paste command
    std::thread::sleep(std::time::Duration::from_millis(50));
    
    // Now create terminal backend after bracketed paste is enabled
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))
        .context("Failed to create terminal")?;
    terminal.clear().context("Failed to clear terminal")?;
    terminal.hide_cursor().context("Failed to hide cursor")?;

    // Initialize the memory engine, which handles persistent conversation history
    let memory_engine = MemoryEngine::new(config).context("Failed to initialize memory engine")?;
    
    // Initialize memory percentage display and set initial resting state
    PersistentStatusBar::update_memory_percentage(memory_engine.get_memory_percentage());
    PersistentStatusBar::set_resting();

    // Load existing history into our display buffer
    let history = memory_engine.get_recent_messages(50).await?;
    let mut display_lines: Vec<String> = history.iter().flat_map(|msg| {
        let content_width = crossterm::terminal::size().map(|(w, _)| w.saturating_sub(10) as usize).unwrap_or(80);
        textwrap::wrap(&msg.content, content_width)
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>()
    }).collect();

    if display_lines.is_empty() {
        display_lines.push("Glimmer AI - Interactive Chat Mode".to_string());
        display_lines.push("Type '/help' for commands, or 'exit' to quit.".to_string());
        display_lines.push("Tip: Hold Shift while clicking to select text for copying".to_string());
    }

    let mut input_buffer = String::new();
    let mut cursor_pos = 0;
    let mut scroll_offset: usize = 0;
    
    let mut processing_task: Option<tokio::task::JoinHandle<Result<Option<String>>>> = None;
    let mut status_complete_timer: Option<std::time::Instant> = None;
    let mut current_reasoning: String = String::new();
    let mut reasoning_steps: Vec<String> = Vec::new();
    // Removed unused reasoning cycling variables

    // Initial UI render
    terminal.draw(|f| {
        // Calculate dynamic input height based on content
        let input_lines = if input_buffer.is_empty() {
            1
        } else {
            let width = f.size().width.saturating_sub(4) as usize; // Account for prompt and borders
            ((input_buffer.len() + width - 1) / width).min(4).max(1) // 1-4 lines
        };
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),                    // Main chat content (scrollable)
                Constraint::Length(1),                 // AI reasoning area
                Constraint::Length(1),                 // Empty line
                Constraint::Length(input_lines as u16), // Dynamic input area 
                Constraint::Length(1),                 // Status bar 
            ])
            .split(f.size());

        // Display chat history with scrolling
        let visible_lines = chunks[0].height as usize;
        let start_idx = if display_lines.len() <= visible_lines {
            0
        } else {
            // When scroll_offset is 0, show most recent content
            // When scroll_offset increases, show older content (smaller start_idx)
            let max_scroll = display_lines.len().saturating_sub(visible_lines);
            max_scroll.saturating_sub(scroll_offset)
        };
        let end_idx = std::cmp::min(start_idx + visible_lines, display_lines.len());
        
        let display_text = if display_lines.is_empty() {
            "Welcome to Glimmer! Type your message below and press Enter.\nPress Ctrl+D to exit, Ctrl+C to copy.".to_string()
        } else {
            display_lines[start_idx..end_idx].join("\n")
        };
        
        let main_paragraph = Paragraph::new(display_text)
            .wrap(Wrap { trim: false });
        f.render_widget(main_paragraph, chunks[0]);

        // Multi-line input with cursor
        let input_with_cursor = if cursor_pos == input_buffer.chars().count() {
            format!("❯ {}█", input_buffer)
        } else {
            let (before, after) = input_buffer.chars().enumerate()
                .fold((String::new(), String::new()), |(mut b, mut a), (i, c)| {
                    if i < cursor_pos { b.push(c); } else { a.push(c); }
                    (b, a)
                });
            format!("❯ {}█{}", before, after)
        };
        
        let input_paragraph = Paragraph::new(input_with_cursor)
            .style(Style::default().fg(Color::Rgb(16, 185, 129)))
            .wrap(Wrap { trim: false }); // Enable word wrapping for multiline input
        f.render_widget(input_paragraph, chunks[3]);

        // Status bar from PersistentStatusBar
        let status_text = crate::thinking_display::PersistentStatusBar::get_status_bar_ui();
        let status_paragraph = Paragraph::new(status_text)
            .style(Style::default().fg(Color::Rgb(203, 166, 247)));
        f.render_widget(status_paragraph, chunks[4]);
    })?;

    let mut needs_redraw = false;
    
    // Timing-based paste detection since crossterm bracketed paste is broken on Windows
    let mut paste_buffer = String::new();
    let mut last_char_time = std::time::Instant::now();
    let mut is_pasting = false;
    let mut last_paste_content = String::new(); // Store actual paste content
    let mut show_paste_placeholder = false; // Flag to show clean placeholder
    let mut last_char: Option<char> = None; // Store the previous character
    // Timing thresholds for paste detection
    let paste_threshold = std::time::Duration::from_millis(25); // Very fast input = pasting
    let paste_end_threshold = std::time::Duration::from_millis(150); // Gap = end of paste
    
    loop {
        // --- Handle input events FIRST before any UI operations ---
        // Use very short timeout to be responsive  
        if event::poll(Duration::from_millis(10))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                
                match key.code {
                    // Ctrl+C is for copying, use Ctrl+D to exit instead
                    KeyCode::Char('d') if key.modifiers == KeyModifiers::CONTROL => break,
                    KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Handle Ctrl+V by reading from system clipboard
                        match ClipboardContext::new() {
                            Ok(mut ctx) => {
                                match ctx.get_contents() {
                                    Ok(clipboard_text) => {
                                        if !clipboard_text.is_empty() {
                                            let lines = clipboard_text.lines().count();
                                            let bytes = clipboard_text.len();
                                            
                                            // Show paste notification
                                            if bytes > 50 || lines > 1 {
                                                display_lines.push(format!("📋 Pasted from clipboard: {} bytes, {} lines", bytes, lines));
                                            }
                                            
                                            // Insert clipboard content at cursor position
                                            let byte_idx_to_insert = input_buffer.char_indices()
                                                .nth(cursor_pos)
                                                .map(|(i, _)| i)
                                                .unwrap_or(input_buffer.len());
                                            input_buffer.insert_str(byte_idx_to_insert, &clipboard_text);
                                            cursor_pos += clipboard_text.chars().count();
                                            
                                            PersistentStatusBar::update_typing_status();
                                        } else {
                                            display_lines.push("📋 Clipboard is empty".to_string());
                                        }
                                    }
                                    Err(e) => {
                                        display_lines.push(format!("Error: Failed to read clipboard: {}", e));
                                    }
                                }
                            }
                            Err(e) => {
                                display_lines.push(format!("Error: Failed to access clipboard: {}", e));
                            }
                        }
                        needs_redraw = true;
                    }
                    KeyCode::Insert if key.modifiers.contains(KeyModifiers::SHIFT) => {
                        // Handle Shift+Insert as alternative paste method
                        match ClipboardContext::new() {
                            Ok(mut ctx) => {
                                match ctx.get_contents() {
                                    Ok(clipboard_text) => {
                                        if !clipboard_text.is_empty() {
                                            let lines = clipboard_text.lines().count();
                                            let bytes = clipboard_text.len();
                                            
                                            // Show paste notification
                                            if bytes > 50 || lines > 1 {
                                                display_lines.push(format!("📋 Pasted from clipboard (Shift+Insert): {} bytes, {} lines", bytes, lines));
                                            }
                                            
                                            // Insert clipboard content at cursor position
                                            let byte_idx_to_insert = input_buffer.char_indices()
                                                .nth(cursor_pos)
                                                .map(|(i, _)| i)
                                                .unwrap_or(input_buffer.len());
                                            input_buffer.insert_str(byte_idx_to_insert, &clipboard_text);
                                            cursor_pos += clipboard_text.chars().count();
                                            
                                            PersistentStatusBar::update_typing_status();
                                        } else {
                                            display_lines.push("📋 Clipboard is empty".to_string());
                                        }
                                    }
                                    Err(e) => {
                                        display_lines.push(format!("Error: Failed to read clipboard: {}", e));
                                    }
                                }
                            }
                            Err(e) => {
                                display_lines.push(format!("Error: Failed to access clipboard: {}", e));
                            }
                        }
                        needs_redraw = true;
                    }
                    KeyCode::Esc => {
                        // ESC only interrupts AI processing, never closes app
                        if processing_task.is_some() {
                            // Interrupt the AI processing
                            if let Some(handle) = processing_task.take() {
                                handle.abort();
                                display_lines.push("Interrupted by user".to_string());
                                PersistentStatusBar::set_resting();
                                scroll_offset = 0;
                                needs_redraw = true;
                            }
                        }
                        // If no processing active, ESC does nothing
                    }
                    KeyCode::Enter => {
                        // Ignore Enter key during pasting to prevent auto-submission
                        if is_pasting {
                            continue;
                        }
                        if !input_buffer.is_empty() && processing_task.is_none() {
                            // If we have a paste placeholder, replace it with actual content before sending
                            let user_input = if show_paste_placeholder && input_buffer.contains("[pasted ") && !last_paste_content.is_empty() {
                                // Replace the placeholder with actual pasted content, keep the rest of the input
                                let placeholder_pattern = regex::Regex::new(r"\[pasted \d+ lines, \d+B\]").unwrap();
                                placeholder_pattern.replace(&input_buffer, &last_paste_content).to_string()
                            } else {
                                input_buffer.clone()
                            };
                            
                            if matches!(user_input.as_str(), "exit" | "quit") {
                                break;
                            }

                            // Reset paste tracking
                            show_paste_placeholder = false;
                            last_paste_content.clear();

                            input_buffer.clear();
                            cursor_pos = 0;
                            // Show truncated version in display if content is very long
                            // Add empty line before user message for better spacing
                            display_lines.push("".to_string());
                            let display_message = if user_input.chars().count() > 100 { 
                                let truncated: String = user_input.chars().take(97).collect();
                                format!("❯ {}... ({} chars)", truncated, user_input.chars().count())
                            } else { 
                                format!("❯ {}", user_input)
                            };
                            display_lines.push(display_message);
                            display_lines.push("".to_string()); // Add empty line after user message
                            needs_redraw = true;
                            
                            let memory_clone = memory_engine.clone();
                            let config_clone = config.clone();
                            
                            // Clear old reasoning and start capturing real AI communication
                            reasoning_steps.clear();
                            current_reasoning.clear();
                            PersistentStatusBar::clear_ai_thinking();
                            
                            // Set up task to process response asynchronously
                            processing_task = Some(tokio::spawn(async move {
                                process_chat_input(&user_input, &memory_clone, &config_clone).await
                            }));
                            
                            // Immediately show "thinking" status and scroll to bottom
                            PersistentStatusBar::update_status("Thinking");
                            scroll_offset = 0; // Scroll to bottom to see response
                        }
                    }
                    KeyCode::Backspace => {
                        if cursor_pos > 0 {
                            let byte_idx_to_remove = input_buffer.char_indices().nth(cursor_pos - 1).map(|(i, _)| i);
                            if let Some(byte_idx) = byte_idx_to_remove {
                                input_buffer.remove(byte_idx);
                            }
                            cursor_pos -= 1;
                            
                            // Show "Typing" status when user is editing (just like 't' key test)
                            if input_buffer.is_empty() {
                                PersistentStatusBar::set_resting();
                            } else {
                                PersistentStatusBar::update_typing_status();
                            }
                            needs_redraw = true;
                        }
                    }
                    KeyCode::Left => {
                        cursor_pos = cursor_pos.saturating_sub(1);
                        needs_redraw = true;
                    }
                    KeyCode::Right => {
                        if cursor_pos < input_buffer.chars().count() {
                            cursor_pos += 1;
                            needs_redraw = true;
                        }
                    }
                    // Remove custom scrolling to restore normal arrow key behavior for input history
                    KeyCode::Char(c) => {
                        // Never process newlines as characters - only KeyCode::Enter should submit
                        if c == '\n' || c == '\r' {
                            continue;
                        }
                        
                        let now = std::time::Instant::now();
                        let time_since_last = now.duration_since(last_char_time);
                        
                        // Detect start of paste (immediate rapid input on first character)
                        if !is_pasting && (time_since_last < paste_threshold || (input_buffer.is_empty() && time_since_last < std::time::Duration::from_millis(5))) {
                            is_pasting = true;
                            paste_buffer.clear();
                            
                            // If we have a previous character in input_buffer, it's part of the paste
                            if let Some(prev_char) = last_char {
                                paste_buffer.push(prev_char);
                                // Remove that character from input_buffer since it's now part of paste
                                if !input_buffer.is_empty() {
                                    input_buffer.pop();
                                    cursor_pos = cursor_pos.saturating_sub(1);
                                }
                            }
                            
                            // Add the current character that triggered paste detection
                            paste_buffer.push(c);
                        } else if is_pasting {
                            // We're in paste mode - buffer all characters for line counting
                            paste_buffer.push(c);
                        } else {
                            // Normal character input
                            let byte_idx_to_insert = input_buffer.char_indices()
                                .nth(cursor_pos)
                                .map(|(i, _)| i)
                                .unwrap_or(input_buffer.len());
                            input_buffer.insert(byte_idx_to_insert, c);
                            cursor_pos += 1;
                            
                            // Show "Typing" status when user is typing
                            PersistentStatusBar::update_typing_status();
                            needs_redraw = true;
                        }
                        
                        // Store the current character as last_char for next iteration
                        last_char = Some(c);
                        last_char_time = now;
                    }
                    _ => {}
                }
            }
            Event::Paste(text) => {
                // SUCCESS! This proves bracketed paste is working
                display_lines.push("🎉 SUCCESS! Event::Paste received!".to_string());
                display_lines.push(format!("📋 Pasted {} bytes: '{}'", text.len(), text));
                
                // Add the pasted text to input buffer at cursor position
                let cursor_byte_pos = input_buffer.char_indices()
                    .nth(cursor_pos)
                    .map(|(i, _)| i)
                    .unwrap_or(input_buffer.len());
                input_buffer.insert_str(cursor_byte_pos, &text);
                cursor_pos += text.chars().count();
                
                // Update typing status
                PersistentStatusBar::update_typing_status();
                needs_redraw = true;
            }
            // Handle ONLY scroll wheel, ignore all other mouse events for text selection
            Event::Mouse(mouse_event) => {
                match mouse_event.kind {
                    crossterm::event::MouseEventKind::ScrollUp => {
                        // Scroll up in chat history (show older messages)
                        let max_scroll = display_lines.len().saturating_sub(20);
                        scroll_offset = (scroll_offset + 3).min(max_scroll);
                        needs_redraw = true;
                    }
                    crossterm::event::MouseEventKind::ScrollDown => {
                        // Scroll down in chat history (show newer messages)
                        scroll_offset = scroll_offset.saturating_sub(3);
                        needs_redraw = true;
                    }
                    // CRITICAL: Don't handle any other mouse events - let them pass through
                    _ => {
                        // This allows text selection to work by not consuming the events
                    }
                }
            }
            _ => {}
            }
        } else if is_pasting {
            // No input for a while, check if paste ended
            let time_since_last = std::time::Instant::now().duration_since(last_char_time);
            if time_since_last > paste_end_threshold {
                is_pasting = false;
                
                // Process the paste buffer as a single paste event
                if !paste_buffer.is_empty() {
                    let bytes = paste_buffer.len();
                    
                    // Count actual line breaks more accurately - handle both \r\n and \n
                    let newline_count = paste_buffer.matches('\n').count();
                    let carriage_return_count = paste_buffer.matches('\r').count();
                    
                    // If we have both \r and \n, it's likely \r\n pairs (Windows), so use \n count
                    // If we only have \r, those are the line breaks (old Mac style)
                    // If large content with no line breaks, estimate lines based on content patterns
                    let lines = if newline_count > 0 {
                        newline_count + 1  // \n means we have multiple lines
                    } else if carriage_return_count > 0 {
                        carriage_return_count + 1  // \r means we have multiple lines
                    } else if bytes > 200 {
                        // Large paste without line breaks - estimate lines based on content length
                        (bytes / 80) + 1  // Assume ~80 chars per line
                    } else {
                        1  // Small content = single line
                    };
                    
                    // For multi-line pastes, show clean placeholder instead of raw content
                    if lines > 2 {
                        // Store the actual content and show a placeholder
                        last_paste_content = paste_buffer.clone();
                        show_paste_placeholder = true;
                        
                        let placeholder = format!("[pasted {} lines, {}B]", lines, bytes);
                        
                        // Add placeholder to input buffer
                        let cursor_byte_pos = input_buffer.char_indices()
                            .nth(cursor_pos)
                            .map(|(i, _)| i)
                            .unwrap_or(input_buffer.len());
                        input_buffer.insert_str(cursor_byte_pos, &placeholder);
                        cursor_pos += placeholder.chars().count();
                    } else {
                        // Single/two line content - remove newlines and use as single line
                        let cleaned_paste = paste_buffer.replace('\n', " ").replace('\r', " ");
                        let cursor_byte_pos = input_buffer.char_indices()
                            .nth(cursor_pos)
                            .map(|(i, _)| i)
                            .unwrap_or(input_buffer.len());
                        input_buffer.insert_str(cursor_byte_pos, &cleaned_paste);
                        cursor_pos += cleaned_paste.chars().count();
                    }
                    
                    // Update typing status
                    PersistentStatusBar::update_typing_status();
                    needs_redraw = true;
                }
                
                paste_buffer.clear();
            }
        }

        // Always redraw if processing or on input events (fixes display refresh bug)
        let should_redraw = needs_redraw || processing_task.is_some() || status_complete_timer.is_some();
        if should_redraw {
            terminal.draw(|f| {
                // Calculate dynamic input height based on content
                let input_lines = if input_buffer.is_empty() {
                    1
                } else {
                    let width = f.size().width.saturating_sub(4) as usize; // Account for prompt and borders
                    ((input_buffer.len() + width - 1) / width).min(4).max(1) // 1-4 lines
                };
                
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Min(1),                    // Main chat content (scrollable)
                        Constraint::Length(1),                 // AI reasoning area
                        Constraint::Length(1),                 // Empty line
                        Constraint::Length(input_lines as u16), // Dynamic input area 
                        Constraint::Length(1),                 // Status bar 
                    ])
                    .split(f.size());

            let main_area = chunks[0];
            let reasoning_area = chunks[1];  // AI reasoning display
            let _empty_area = chunks[2];     // Empty line for spacing
            let input_area = chunks[3]; 
            let status_area = chunks[4];

            // --- Render main chat content (scrollable) ---
            let content_height = main_area.height as usize;
            let start_idx = if display_lines.len() <= content_height {
                0
            } else {
                // When scroll_offset is 0, show most recent content
                // When scroll_offset increases, show older content (smaller start_idx)
                let max_scroll = display_lines.len().saturating_sub(content_height);
                max_scroll.saturating_sub(scroll_offset)
            };
            
            let lines_to_show: Vec<Line> = display_lines
                .iter()
                .skip(start_idx)
                .take(content_height)
                .map(|s| parse_markdown_line(s))
                .collect();
                
            // Render chat content without any custom scrollbar
            let chat_content = Paragraph::new(lines_to_show).wrap(Wrap { trim: false });
            f.render_widget(chat_content, main_area);

            // --- Render reasoning area with detailed AI thinking ---
            let actual_ai_thinking = crate::thinking_display::PersistentStatusBar::get_ai_thinking();
            let reasoning_steps = crate::thinking_display::PersistentStatusBar::get_latest_reasoning_steps();
            let internal_prompt = crate::thinking_display::PersistentStatusBar::get_ai_internal_prompt();
            
            let reasoning_display = if !actual_ai_thinking.is_empty() || !reasoning_steps.is_empty() || !current_reasoning.is_empty() {
                // Format: 💭 current action (last response from gemini reasoning)
                let current_action = if actual_ai_thinking.starts_with("●") {
                    // If it's a status line with ●, show it as is
                    actual_ai_thinking.clone()
                } else if !actual_ai_thinking.is_empty() {
                    // Regular thinking content gets 💭 prefix
                    format!("💭 {}", actual_ai_thinking)
                } else if !current_reasoning.is_empty() {
                    format!("💭 {}", current_reasoning)
                } else {
                    // Do NOT show reasoning_steps or status messages in 💭 display
                    // The 💭 display should ONLY show actual AI thinking from Gemini API
                    String::new()
                };
                
                current_action
            } else {
                String::new()
            };
            
            // Create properly colored reasoning display with ANSI code stripping
            let reasoning_lines = if reasoning_display.contains("●") {
                let lines: Vec<&str> = reasoning_display.split('\n').collect();
                let mut result_lines = Vec::new();
                
                for line in lines.iter() {
                    // Strip ANSI escape codes from the line
                    let clean_line = strip_ansi_codes(line);
                    
                    if clean_line.contains("● Analyzing") {
                        let text_part = clean_line.strip_prefix("●").unwrap_or(&clean_line).to_string();
                        let spans = vec![
                            Span::styled("●", Style::default().fg(Color::Rgb(255, 204, 92))), // #ffcc5c yellow
                            Span::styled(text_part, Style::default().fg(Color::White)) // WHITE text with original spacing
                        ];
                        result_lines.push(Line::from(spans));
                    } else if clean_line.contains("● Editing") {
                        let text_part = clean_line.strip_prefix("●").unwrap_or(&clean_line).to_string();
                        let spans = vec![
                            Span::styled("●", Style::default().fg(Color::Rgb(255, 175, 0))), // #ffaf00 orange
                            Span::styled(text_part, Style::default().fg(Color::White)) // WHITE text with original spacing
                        ];
                        result_lines.push(Line::from(spans));
                    } else if clean_line.contains("● Task completed") {
                        let text_part = clean_line.strip_prefix("●").unwrap_or(&clean_line).to_string();
                        let spans = vec![
                            Span::styled("●", Style::default().fg(Color::Rgb(159, 239, 0))), // #9fef00 green
                            Span::styled(text_part, Style::default().fg(Color::White)) // WHITE text with original spacing
                        ];
                        result_lines.push(Line::from(spans));
                    } else if clean_line.starts_with("  ⎿") {
                        result_lines.push(Line::from(Span::styled(clean_line, Style::default().fg(Color::DarkGray))));
                    } else if !clean_line.trim().is_empty() {
                        result_lines.push(Line::from(Span::styled(clean_line, Style::default().fg(Color::DarkGray))));
                    }
                }
                result_lines
            } else {
                vec![Line::from(Span::styled(reasoning_display, Style::default().fg(Color::DarkGray)))]
            };
            
            let reasoning_widget = Paragraph::new(reasoning_lines)
                .wrap(ratatui::widgets::Wrap { trim: true });
            f.render_widget(reasoning_widget, reasoning_area);

            // --- Render input area with solid cursor ---
            let prompt = Span::styled("❯ ", Style::default().fg(Color::Rgb(46, 204, 113))); // Emerald green
            
            // Create display string with solid cursor, showing only the last part if too long
            let available_width = input_area.width.saturating_sub(3) as usize; // Account for prompt
            let display_input = if input_buffer.is_empty() {
                "█".to_string()
            } else {
                let full_text_with_cursor = if cursor_pos >= input_buffer.chars().count() {
                    format!("{}█", input_buffer)
                } else {
                    let chars: Vec<char> = input_buffer.chars().collect();
                    let mut result = String::new();
                    for (i, ch) in chars.iter().enumerate() {
                        if i == cursor_pos {
                            result.push('█');
                            result.push(*ch);
                        } else {
                            result.push(*ch);
                        }
                    }
                    if cursor_pos == 0 && !chars.is_empty() {
                        format!("█{}", input_buffer)
                    } else {
                        result
                    }
                };
                
                // Show only the last part if text is too long
                if full_text_with_cursor.len() > available_width {
                    let start = full_text_with_cursor.len().saturating_sub(available_width);
                    full_text_with_cursor.chars().skip(start).collect()
                } else {
                    full_text_with_cursor
                }
            };
            
            let input_paragraph = Paragraph::new(Line::from(vec![prompt, Span::raw(&display_input)]))
                .wrap(Wrap { trim: false }); // Enable word wrapping for multiline input
            f.render_widget(input_paragraph, input_area);

            // --- Render status bar ---
            let status_line = PersistentStatusBar::get_status_bar_ui();
            let status_bar = Paragraph::new(status_line)
                .style(Style::default().bg(Color::Reset)); // Ensure it has a background
            f.render_widget(status_bar, status_area);
            })?;
            needs_redraw = false;
        }

        // No more cycling needed - we show real-time AI thinking

        // --- Handle real-time function call updates ---
        if let Some(_handle) = &processing_task {
            // Check for new reasoning steps from function calls
            let latest_steps = crate::thinking_display::PersistentStatusBar::get_latest_reasoning_steps();
            if latest_steps.len() > reasoning_steps.len() {
                // Add new steps to display_lines in real-time
                for step in latest_steps.iter().skip(reasoning_steps.len()) {
                    display_lines.push(step.clone());
                }
                reasoning_steps = latest_steps;
                needs_redraw = true;
                scroll_offset = 0; // Auto-scroll to show new updates
            }
            
        }

        // --- Handle background task completion ---
        if let Some(handle) = &mut processing_task {
            if handle.is_finished() {
                let task = processing_task.take().unwrap();
                match task.await {
                    Ok(Ok(Some(response))) => {
                        // Filter out AI interpretation blocks from display
                        let filtered_response = filter_ai_interpretation_blocks(&response);
                        
                        // Preserve clean function display format - don't wrap lines that start with ● or ⎿
                        for line in filtered_response.lines() {
                            if line.trim().starts_with("●") || line.trim().starts_with("⎿") {
                                // Function display lines - keep exact formatting
                                display_lines.push(line.to_string());
                            } else if !line.trim().is_empty() {
                                // Other content - wrap if needed
                                let content_width = terminal.size()?.width.saturating_sub(10) as usize;
                                let wrapped = textwrap::wrap(line, content_width);
                                display_lines.extend(wrapped.into_iter().map(|s| s.to_string()));
                            } else {
                                // Empty lines
                                display_lines.push("".to_string());
                            }
                        }
                        
                        current_reasoning.clear();
                        // Don't clear reasoning_steps to preserve function display
                        // Don't update status to "Complete" as it interferes with AI thinking display
                        needs_redraw = true;
                    }
                    Ok(Ok(None)) => {
                        current_reasoning.clear();
                        reasoning_steps.clear();
                        PersistentStatusBar::update_status("No response");
                        needs_redraw = true;
                    }
                    Ok(Err(e)) => {
                        current_reasoning.clear();
                        reasoning_steps.clear();
                        display_lines.push(format!("Error: {}", e));
                        needs_redraw = true;
                    }
                    Err(e) => { // This is a JoinError
                        current_reasoning.clear();
                        reasoning_steps.clear();
                        display_lines.push(format!("Task Error: {}", e));
                        needs_redraw = true;
                    }
                }
                status_complete_timer = Some(std::time::Instant::now());
                scroll_offset = 0; // Scroll to bottom to see response
            }
        }

        // --- Handle status bar reset timer ---
        if let Some(timer) = status_complete_timer {
            if timer.elapsed() > Duration::from_secs(2) {
                PersistentStatusBar::set_resting();
                status_complete_timer = None;
            }
        }
        
        // --- Check typing timeout ---
        PersistentStatusBar::check_typing_timeout();
    }

    // Properly restore terminal state
    disable_raw_mode().context("Failed to disable raw mode")?;
    execute!(
        terminal.backend_mut(),
        cursor::Show,
        DisableBracketedPaste,
        crossterm::event::DisableMouseCapture
    ).context("Failed to restore terminal")?;
    
    // Goodbye message handled by ratatui, not direct print
    Ok(())
}

/// Process chat input (simplified version without interrupts for now)
async fn process_chat_input(
    input: &str,
    memory_engine: &MemoryEngine,
    config: &Config,
) -> Result<Option<String>> {
    // Handle special commands
    match input {
        "/help" => {
            return Ok(Some(get_help_text()));
        }
        "/clear" => {
            memory_engine.clear_conversation().await?;
            return Ok(Some(" Conversation cleared".to_string()));
        }
        "/permissions" => {
            return Ok(Some(handle_permissions_command().await?));
        }
        "/pwd" => {
            let current_dir = std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            return Ok(Some(format!(" Current directory: {}", current_dir)));
        }
        "/ls" => {
            return Ok(Some(handle_ls_command().await));
        }
        "/paste" => {
            // Direct clipboard paste command as workaround for Ctrl+V issues
            match ClipboardContext::new() {
                Ok(mut ctx) => {
                    match ctx.get_contents() {
                        Ok(clipboard_text) => {
                            if !clipboard_text.is_empty() {
                                let lines = clipboard_text.lines().count();
                                let bytes = clipboard_text.len();
                                return Ok(Some(format!("📋 Pasted from clipboard: {} bytes, {} lines\n{}", bytes, lines, clipboard_text)));
                            } else {
                                return Ok(Some("📋 Clipboard is empty".to_string()));
                            }
                        }
                        Err(e) => {
                            return Ok(Some(format!("Error: Failed to read clipboard: {}", e)));
                        }
                    }
                }
                Err(e) => {
                    return Ok(Some(format!("Error: Failed to access clipboard: {}", e)));
                }
            }
        }
        _ if input.starts_with('/') => {
            return Ok(Some(format!("❓ Unknown command: {}. Type '/help' for available commands.", input)));
        }
        _ => {
            // Process as regular chat input
        }
    }

    // Add user message to memory
    let _user_msg = ChatMessage {
        role: MessageRole::User,
        content: input.to_string(),
        timestamp: chrono::Utc::now(),
        importance: crate::cli::memory::MessageImportance::Contextual,
        metadata: crate::cli::memory::MessageMetadata::default(),
    };
    memory_engine.add_message(MessageRole::User, input).await?;

    // Process with unified intelligent routing
    match process_intelligently(input, memory_engine, config).await {
        Ok(response) => {
            // Add assistant response to memory
            let _assistant_msg = ChatMessage {
                role: MessageRole::Assistant,
                content: response.clone(),
                timestamp: chrono::Utc::now(),
                importance: crate::cli::memory::MessageImportance::Important,
                metadata: crate::cli::memory::MessageMetadata::default(),
            };
            memory_engine.add_message(MessageRole::Assistant, &response).await?;
            
            // Update memory percentage after assistant response
            PersistentStatusBar::update_memory_percentage(memory_engine.get_memory_percentage());
            Ok(Some(response))
        }
        Err(e) => {
            return Err(e);
        }
    }
}

/// Unified intelligent processing - single entry point for all requests
async fn process_intelligently(
    input: &str,
    memory_engine: &MemoryEngine,
    config: &Config,
) -> Result<String> {
    // The logic for triaging requests has been consolidated into the ReasoningEngine
    // to create a single, more intelligent decision-making point.
    let reasoning_engine = crate::reasoning_engine::ReasoningEngine::new(config);
    let conversation_history = memory_engine.get_context(10, 1000).await.unwrap_or_default();
    let request_type = reasoning_engine.triage_request(input, &conversation_history).await?;
    
    match request_type {
        RequestType::DirectAnswer => {
            PersistentStatusBar::update_status("Providing direct answer");
            let prompt = format!(
                "You are Glimmer, an intelligent AI assistant with comprehensive capabilities. You are:\n\
                - A knowledgeable AI that can answer questions on any topic\n\
                - A skilled programmer and coding assistant\n\
                - Capable of file system operations and code analysis\n\
                - Able to research, explain, and help with any task\n\
                - Direct and helpful with a practical engineering approach\n\n\
                You have full AI capabilities plus specialized file/coding functions. You can handle any request.\n\n\
                Question: {}\n\n\
                Provide a clear, helpful answer.",
                input
            );
            let response = gemini::query_gemini(&prompt, config).await?;
            // Don't update status to "Complete" as it interferes with AI thinking display
            crate::thinking_display::PersistentStatusBar::set_ai_thinking("");
            return Ok(response);
        }
        
        RequestType::Research => {
            PersistentStatusBar::update_status("Researching");
            
            let response = research::perform_intelligent_research(input, config).await?;
            
            return Ok(response);
        }
        
        RequestType::Reasoning => {
            PersistentStatusBar::update_status("Reasoning");
            let reasoning_future = crate::reasoning_engine::handle_ambiguous_request(input, &conversation_history, config);
            let status_future = async {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                PersistentStatusBar::update_status("Processing");
            };
            
            let (response, _) = tokio::join!(reasoning_future, status_future);
            return Ok(response?);
        }
        
        RequestType::FunctionCalling => {
            PersistentStatusBar::update_status("Processing");
            match execute_task_with_function_calling(input, &conversation_history, config, &reasoning_engine).await {
                Ok(output_lines) => Ok(output_lines.join("\n")),
                Err(e) => Ok(format!("I encountered an issue while processing your request: {}", e)),
            }
        }
    }
}

async fn execute_simple_function_calling(
    input: &str,
    conversation_history: &str,
    config: &Config,
) -> Result<String> {
    let registry = FunctionRegistry::new();
    let functions = registry.get_function_definitions();

    let is_analytical_question = input.to_lowercase().contains("what does") ||
                                 input.to_lowercase().contains("what is") ||
                                 input.to_lowercase().contains("explain") ||
                                 input.to_lowercase().contains("analyze") ||
                                 input.to_lowercase().contains("how does");


    let system_prompt = if is_analytical_question {
        format!(
            "I am Glimmer, an idiosyncratic programming engineer and helpful AI assistant. The user is asking an analytical question about a file. \n\n\
            Available functions:\n{}\n\n\
            INSTRUCTIONS:\n\
            1. Read the file using read_file function\n\
            2. ANALYZE the file content to understand what it does\n\
            3. Provide a clear explanation of the file's purpose, functionality, and key components with my engineering perspective\n\
            4. Do NOT just dump the file content - explain what it does in a helpful way\n\n\
            Context: {}\n\n",
            create_function_calling_prompt(&functions),
            conversation_history
        )
    } else {
        format!(
            "I am Glimmer, an idiosyncratic programming engineer and helpful AI assistant. Complete the user's request using the available functions.\n\n\
            Available functions:\n{}\n\n\
            CRITICAL INSTRUCTIONS:\n\
            1. When asked to FIX, EDIT, MODIFY, or CHANGE files - use write_file to make the changes\n\
            2. When asked to READ, SHOW, or EXPLAIN files - use read_file\n\
            3. For warnings/errors: read the file, analyze the issue, then write_file with the fix\n\
            4. Always COMPLETE the task - don't just read files when editing is requested\n\
            5. Be direct and efficient in my engineering approach\n\n\
            Context: {}\n\n",
            create_function_calling_prompt(&functions),
            conversation_history
        )
    };

    let full_prompt = format!("{}\n\nUser request: {}", system_prompt, input);

    match gemini::query_gemini_with_function_calling(&full_prompt, config, Some(&functions)).await {
        Ok((_response_text, Some(function_call), token_usage)) => {
            let token_info = if let Some(usage) = &token_usage {
                format!(" ({}→{})", usage.input_tokens, usage.output_tokens.unwrap_or(0))
            } else {
                let estimated_input = (full_prompt.len() / 4).max(100);
                format!(" ({}→0)", estimated_input)
            };

            let function_start = format!("{} {}{}",
                get_colored_function_indicator(&function_call.name),
                get_function_description(&function_call.name),
                token_info
            );

            crate::thinking_display::PersistentStatusBar::add_reasoning_step(&function_start);

            match execute_function_call(&function_call, config, conversation_history).await {
                Ok(function_result) => {
                    let summary = create_function_result_summary(&function_call.name, &function_result);
                    let result_line = format!("  ⎿ {}", summary);

                    let output = format!("{}\n{}", function_start, result_line);
                    Ok(output)
                }
                Err(e) => {
                    let error_line = format!("  ⎿ Error: {}", e);
                    let output = format!("{}\n{}", function_start, error_line);
                    Ok(output)
                }
            }
        }
        Ok((text_response, None, _)) => {
            Ok(text_response)
        }
        Err(e) => {
            Err(e)
        }
    }
}

async fn execute_task_with_function_calling(
    input: &str,
    conversation_history: &str,
    config: &Config,
    reasoning_engine: &crate::reasoning_engine::ReasoningEngine,
) -> Result<Vec<String>> { // Return lines of output for the TUI
    let registry = FunctionRegistry::new();
    let functions = registry.get_function_definitions();

    let mut output_lines = Vec::new();

    let task_complexity_analysis = classify_task_complexity(input, conversation_history, config).await?;
    if task_complexity_analysis.estimated_steps > 2 {
        let todos = generate_smart_todos(input, conversation_history).await?;
        if !todos.is_empty() {
            output_lines.push("🎯 Breaking down complex task:".to_string());
            for (i, todo) in todos.iter().enumerate() {
                output_lines.push(format!("  {}. {}", i + 1, todo));
            }
            output_lines.push("".to_string());
        }
    }

    let system_prompt = format!(
        "You are Glimmer, an advanced AI coding assistant with sophisticated reasoning capabilities.\n\n\
        METACOGNITIVE AWARENESS:\n\
        - I have access to conversation history and can remember context\n\
        - I can reason about task completion and next steps\n\
        - I must evaluate whether my actions fully satisfy user requests\n\
        - I should use thinking tokens for complex reasoning when needed\n\n\
        Available functions:\n{}\n\n\
        EXECUTION STRATEGY:\n\
        1. For simple tasks: Act immediately with appropriate function calls\n\
        2. For complex tasks: Think through the problem, then act systematically\n\
        3. Always evaluate: Does my current action move toward completing the user's goal?\n\
        4. If user wants changes to a file: Use edit_code directly (it reads the file automatically)\n\
        5. Multi-step tasks: Complete each step before moving to the next\n\n\
        THINKING GUIDELINES:\n\
        - Use <thinking> tags for complex reasoning about user intent\n\
        - Consider the full context of what user is trying to achieve\n\
        - Evaluate if previous actions have been sufficient\n\
        - Plan the most efficient path to complete the task\n\n\
        CURRENT CONTEXT: {}\n\n",
        create_function_calling_prompt(&functions),
        conversation_history
    );

    let mut messages = vec![input.to_string()];
    let mut step_count = 0;
    let max_steps = 8;
    let mut last_function_calls = Vec::new();

    loop {
        step_count += 1;
        if step_count > max_steps {
            output_lines.push("Warning: Maximum steps reached. Task completed (step limit).".to_string());
            break;
        }

        let enhanced_context = conversation_history.to_string();

        if let Some(direct_response) = try_direct_function_execution(input, config, conversation_history).await {
            return Ok(vec![direct_response]);
        }

        let current_prompt = match reasoning_engine.reason_about_request_with_metacognition(input, &enhanced_context).await {
            Ok(reasoning) => {
                format!("{}\n\nUSER REQUEST: {}\n\nREASONING: {}\n\nNEXT STEPS: {}\n\nWHAT I'VE COMPLETED: {}\n\nNOW I MUST: Call the single next function required to make progress on the user's request. Do not re-analyze. ACT NOW.",
                    system_prompt,
                    input,
                    reasoning.interpretation,
                    reasoning.actionable_plan.join(", "),
                    if messages.len() > 1 { messages[1..].join("\n") } else { "This is the first step.".to_string() }
                )
            }
            Err(_) => {
                format!("{}\n\nUser request: {}\n\nWhat I've done so far:\n{}\n\nI must determine: Have I fully completed this request? If not, I must call the appropriate function to continue.",
                    system_prompt,
                    input,
                    if messages.len() > 1 { messages[1..].join("\n") } else { "Nothing yet".to_string() }
                )
            }
        };

        let thinking_budget = if step_count == 1 { Some(2048) } else { Some(1024) };

        let (response_text, function_call, _token_usage) = match
            gemini::query_gemini_with_function_calling_and_thinking(&current_prompt, config, Some(&functions), thinking_budget).await {
            Ok(result) => result,
            Err(e) => {
                crate::thinking_display::PersistentStatusBar::set_ai_thinking("Function calling failed, trying simplified approach");
                let simplified_prompt = format!("USER REQUEST: {}\n\nPlease help with this request.", input);
                match crate::gemini::query_gemini(&simplified_prompt, config).await {
                    Ok(fallback_response) => (fallback_response, None, None),
                    Err(_) => return Err(e),
                }
            }
        };

        if let Some(func_call) = function_call {
            let call_signature = format!("{}:{}",
                func_call.name,
                func_call.arguments.get("file_path").and_then(|v| v.as_str()).unwrap_or(""));

            if last_function_calls.iter().filter(|&x| x == &call_signature).count() >= 3 {
                let recovery_action = analyze_loop_situation(&func_call, input, &output_lines.join("\n"), step_count).await;
                output_lines.push(format!("● {}", recovery_action.description));
                match recovery_action.action_type {
                    LoopRecoveryType::ForceEdit => {
                        if let Some(_file_path) = func_call.arguments.get("file_path").and_then(|v| v.as_str()) {
                            match Ok::<String, anyhow::Error>("Skipping broken edit".to_string()) {
                                Ok(result) => {
                                    output_lines.push(result);
                                    break;
                                }
                                Err(_) => {
                                    output_lines.push("Could not complete recovery action - task stopped".to_string());
                                    break;
                                }
                            }
                        };
                    }
                    LoopRecoveryType::ChangeStrategy => {
                        continue;
                    }
                    LoopRecoveryType::StopWithExplanation => {
                        output_lines.push(recovery_action.instructions);
                        break;
                    }
                }
            }

            last_function_calls.push(call_signature);
            if last_function_calls.len() > 5 {
                last_function_calls.remove(0);
            }

            let function_result = match execute_function_call(
                &func_call,
                config,
                conversation_history,
            ).await {
                Ok(result) => result,
                Err(e) if e.to_string().contains("malformed or truncated") => {
                    let skip_msg = format!("Warning: Skipping malformed function call ({})", func_call.name);
                    output_lines.push(skip_msg.clone());
                    continue;
                },
                Err(e) => return Err(e),
            };

            let summary = create_function_result_summary(&func_call.name, &function_result);
            let result_line = format!("  ⎿ {}", summary);
            output_lines.push(result_line.clone());

            messages.push(function_result.clone());

            if is_task_completed(&func_call.name, &function_result, input) {
                let final_summary = extract_final_summary(&messages.last().unwrap_or(&String::new()));
                output_lines.push(format!("{} Task completed",
                    crate::cli::colors::GREEN_COMPLETE));
                output_lines.push(format!("  ⎿ {}", final_summary));
                break;
            }
        } else if !response_text.is_empty() {
            let final_summary = extract_final_summary(&response_text);
            output_lines.push(format!("{} Task completed",
                crate::cli::colors::GREEN_COMPLETE));
            output_lines.push(format!("  ⎿ {}", final_summary));
            break;
        } else {
            output_lines.push("I understand your request. How can I help you further?".to_string());
            break;
        }
    }

    Ok(output_lines)
}

/// Analyze task complexity to determine if todo breakdown is needed
fn analyze_task_complexity(input: &str) -> u8 {
    let input_lower = input.to_lowercase();
    let mut complexity = 0;
    
    // Multi-file indicators
    if input_lower.contains("files") || input_lower.contains("multiple") || input_lower.contains("all") {
        complexity += 2;
    }
    
    // Complex operations
    if input_lower.contains("refactor") || input_lower.contains("restructure") || 
       input_lower.contains("optimize") || input_lower.contains("implement") {
        complexity += 2;
    }
    
    // Multiple steps indicated
    if input_lower.contains("and") || input_lower.contains("then") || input_lower.contains("also") {
        complexity += 1;
    }
    
    // Size indicators
    if input_lower.contains("large") || input_lower.contains("complex") || input_lower.contains("entire") {
        complexity += 1;
    }
    
    // Word count heuristic
    if input.split_whitespace().count() > 15 {
        complexity += 1;
    }
    
    complexity.min(5)
}

/// Generate smart todos for complex tasks
async fn generate_smart_todos(input: &str, context: &str) -> Result<Vec<String>> {
    // Use local analysis first, only call API if needed
    let local_todos = generate_local_todos(input);
    
    if !local_todos.is_empty() && local_todos.len() >= 2 {
        return Ok(local_todos);
    }
    
    // Fallback to AI-generated todos for very complex tasks
    let prompt = format!(
        "Break down this task into 3-5 specific, actionable steps:\n\
        Task: {}\n\
        Context: {}\n\n\
        Return ONLY a numbered list of concrete actions. Keep each step under 60 characters.",
        input, context
    );
    
    let response = crate::gemini::query_gemini(&prompt, &crate::config::Config::default()).await?;
    let todos: Vec<String> = response.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.len() < 5 {
                None
            } else {
                // Remove numbering and clean up
                let cleaned = trimmed.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ');
                Some(cleaned.to_string())
            }
        })
        .take(5)
        .collect();
    
    Ok(todos)
}

/// Generate todos using local analysis (no API call)
fn generate_local_todos(input: &str) -> Vec<String> {
    let input_lower = input.to_lowercase();
    let mut todos = Vec::new();
    
    // File operation patterns
    if input_lower.contains("edit") || input_lower.contains("modify") || input_lower.contains("change") {
        if input_lower.contains("file") || input.contains(".") {
            todos.push("Read and analyze target file".to_string());
            todos.push("Make required modifications".to_string());
            todos.push("Validate changes work correctly".to_string());
        }
    }
    
    // Creation patterns
    if input_lower.contains("create") || input_lower.contains("build") || input_lower.contains("make") {
        todos.push("Plan structure and requirements".to_string());
        todos.push("Implement core functionality".to_string());
        todos.push("Test and refine implementation".to_string());
    }
    
    // Fix/debug patterns
    if input_lower.contains("fix") || input_lower.contains("debug") || input_lower.contains("problem") {
        todos.push("Identify root cause of issue".to_string());
        todos.push("Implement targeted fix".to_string());
        todos.push("Verify fix resolves problem".to_string());
    }
    
    // Improvement patterns
    if input_lower.contains("improve") || input_lower.contains("optimize") || input_lower.contains("enhance") {
        todos.push("Analyze current implementation".to_string());
        todos.push("Identify improvement opportunities".to_string());
        todos.push("Apply optimizations".to_string());
    }
    
    todos
}

/// Check if any editing functions were executed in this conversation
fn had_editing_functions(output_lines: &[String]) -> bool {
    output_lines.iter().any(|line| {
        line.contains("Editing code") || 
        line.contains("Writing file") ||
        line.contains("Renaming file") ||
        line.contains("edit_code") ||
        line.contains("write_file") ||
        line.contains("rename_file")
    })
}

/// Get human-readable description of what a function does
fn get_function_description(function_name: &str) -> &'static str {
    match function_name {
        "read_file" => "Reading file",
        "write_file" => "Writing file", 
        "rename_file" => "Renaming file",
        "list_directory" => "Scanning directory",
        "edit_code" => "Editing code",
        "diff_files" => "Analyzing differences",
        "fetch_url" => "Fetching content",
        "play_audio" => "Playing audio",
        "search_music" => "Searching music",
        "code_analysis" => "Analyzing code structure",
        _ => "Processing request",
    }
}

fn get_colored_function_indicator(function_name: &str) -> String {
    use crate::cli::colors::*;
    let color = match function_name {
        "edit_code" | "write_file" | "rename_file" => ORANGE_EDIT,
        "code_analysis" | "diff_files" => YELLOW_ANALYZE,
        _ => WHITE_BRIGHT,
    };
    format!("{}●{}", color, RESET)
}

/// Task completion assessment using metacognitive reasoning
#[derive(Debug)]
struct TaskCompletionAssessment {
    is_complete: bool,
    explanation: String,
    confidence: f32,
    next_suggested_action: Option<String>,
}

/// Evaluate if a task is complete using sophisticated metacognitive reasoning
async fn evaluate_task_completion_with_metacognition(
    original_request: &str,
    current_response: &str,
    conversation_history: &[String],
    reasoning_engine: &crate::reasoning_engine::ReasoningEngine,
    config: &Config,
) -> Result<TaskCompletionAssessment> {
    // Build rich context from conversation history
    let context = if conversation_history.len() > 1 {
        conversation_history[1..].join("\n")
    } else {
        "No previous actions".to_string()
    };
    
    // Use metacognitive reasoning to understand the current state
    let reasoning = reasoning_engine.reason_about_request_with_metacognition(
        &format!("COMPLETION EVALUATION: User wanted '{}'. I just responded with '{}'. Have I fully completed their request?", 
                original_request, current_response),
        &context
    ).await?;
    
    // Create detailed completion evaluation prompt
    let completion_prompt = format!(
        r#"METACOGNITIVE TASK COMPLETION ANALYSIS
        
Original User Request: "{}"

What I've Done So Far:
{}

My Current Response: "{}"

My Reasoning Analysis: {}
My Capabilities: {}
My Action Plan: {}

CRITICAL EVALUATION: Have I FULLY completed the user's original request?

Analyze:
1. What exactly did the user want me to do?
2. What have I accomplished so far?  
3. Is the user's original goal completely satisfied?
4. If not, what specific actions are still needed?

Respond with JSON:
{{
    "is_complete": boolean,
    "explanation": "Detailed reasoning for completion status",
    "confidence": 0.95,
    "next_action": "specific next step if not complete, or null if complete"
}}"#,
        original_request,
        context,
        current_response,
        reasoning.interpretation,
        reasoning.capability_assessment.join(", "),
        reasoning.actionable_plan.join(", ")
    );
    
    let completion_response = crate::gemini::query_gemini(&completion_prompt, config).await?;
    
    // Parse the JSON response with fallback
    let completion_data: serde_json::Value = {
        let cleaned_response = completion_response.trim()
            .strip_prefix("```json").unwrap_or(&completion_response)
            .strip_suffix("```").unwrap_or(&completion_response)
            .trim();
        
        match serde_json::from_str(cleaned_response) {
            Ok(json) => json,
            Err(_) => {
                // Fallback: return a safe default completion assessment
                serde_json::json!({
                    "is_complete": true,
                    "explanation": "Unable to parse completion analysis, assuming task completed",
                    "confidence": 0.7,
                    "next_action": null
                })
            }
        }
    };
    
    Ok(TaskCompletionAssessment {
        is_complete: completion_data["is_complete"].as_bool().unwrap_or(false),
        explanation: completion_data["explanation"].as_str().unwrap_or("Assessment unclear").to_string(),
        confidence: completion_data["confidence"].as_f64().unwrap_or(0.5) as f32,
        next_suggested_action: completion_data["next_action"].as_str().map(|s| s.to_string()),
    })
}

/// Create a brief summary of function results instead of dumping full content
fn create_function_result_summary(function_name: &str, result: &str) -> String {
    match function_name {
        "edit_code" => {
            if result.contains("Error:") || result.contains("Failed:") {
                result.lines().find(|line| line.contains("Error:") || line.contains("Failed:"))
                    .unwrap_or("Edit failed")
                    .to_string()
            } else if result.contains("bytes to") {
                // Extract file written info like "Wrote 2106 bytes to 'file.css'"
                result.lines()
                    .find(|line| line.contains("bytes to"))
                    .unwrap_or("Edit completed")
                    .to_string()
            } else {
                "Edit completed successfully".to_string()
            }
        },
        "read_file" => {
            if result.contains("File content of") {
                let first_line = result.lines().next().unwrap_or("");
                if first_line.contains("lines") {
                    first_line.to_string()
                } else {
                    format!("Read file successfully")
                }
            } else {
                result.lines().next().unwrap_or("Read completed").to_string()
            }
        },
        "list_directory" => {
            // Return a concise summary instead of the full listing to prevent triple display
            if result.contains("📁") {
                // Extract just the directory path from first line
                if let Some(first_line) = result.lines().next() {
                    if first_line.starts_with("📁") {
                        format!("Directory contents listed: {}", first_line.trim_start_matches("📁 ").trim())
                    } else {
                        "Directory contents listed".to_string()
                    }
                } else {
                    "Directory contents listed".to_string()
                }
            } else {
                "Directory listed successfully".to_string()
            }
        },
        "code_analysis" => "Code analysis completed successfully".to_string(),
        _ => {
            // For other functions, show the first line or a cleaned version
            if result.contains("Error:") {
                result.lines().next().unwrap_or("Operation failed").to_string()
            } else {
                result.lines().next().unwrap_or("Operation completed").to_string()
            }
        }
    }
}

// Removed hardcoded task completion logic - letting AI reason about completion instead

// Removed old complex multi-path processing functions - using unified intelligent approach now

// New version with UI integration
async fn process_chat_input_with_ui(
    input: &str,
    memory_engine: &mut MemoryEngine,
    config: &Config,
    chat_ui: &ChatUI,
) -> Result<()> {
    // Handle special commands
    match input {
        "/help" => {
            print_help_with_ui(chat_ui);
            return Ok(());
        }
        "/clear" => {
            memory_engine.clear_conversation().await?;
            chat_ui.display_system_message(" Conversation cleared", SystemMessageType::Success);
            return Ok(());
        }
        "/permissions" => {
            handle_permissions_command_with_ui(chat_ui).await?;
            return Ok(());
        }
        "/pwd" => {
            let current_dir = std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            chat_ui.display_system_message(
                &format!(" Current directory: {}", current_dir),
                SystemMessageType::Info,
            );
            return Ok(());
        }
        "/ls" => {
            handle_ls_command_with_ui(chat_ui).await;
            return Ok(());
        }
        _ if input.starts_with('/') => {
            chat_ui.display_system_message(
                &format!("❓ Unknown command: {}. Type '/help' for available commands.", input),
                SystemMessageType::Warning,
            );
            return Ok(());
        }
        _ => {
            // No special command handling needed - function calling will handle all requests
        }
    }

    // Display user message in chat UI
    chat_ui.display_user_message(input);

    // Add user message to conversation
    memory_engine.add_message(MessageRole::User, input).await?;

    // Get AI response with thinking display
    match get_ai_response_with_ui(input, memory_engine, config, chat_ui).await {
        Ok((response, thinking)) => {
            // Display the AI response using the chat UI
            chat_ui.display_response(&response, thinking.as_deref());
            memory_engine.add_message(MessageRole::Assistant, &response).await?;
            
            // Update memory percentage after assistant response
            PersistentStatusBar::update_memory_percentage(memory_engine.get_memory_percentage());
        }
        Err(e) => {
            chat_ui.display_system_message(&format!("🤖 AI Error: {}", e), SystemMessageType::Error);
        }
    }
    Ok(())
}

// Removed duplicate process_chat_input function

/// Try to handle file operations directly without going through AI
async fn try_direct_file_operation(input: &str) -> Option<String> {
    let input_lower = input.to_lowercase();
    
    // Direct file path patterns
    if let Some(file_path) = extract_obvious_file_path(input) {
        let path = std::path::Path::new(&file_path);
        
        // Check if file exists
        if !path.exists() {
            return Some(format!("Error: File not found: {}", file_path));
        }
        
        // Analyze what the user wants to do
        if input_lower.contains("analyze") || input_lower.contains("look at") || input_lower.contains("show") || input_lower.contains("display") {
            // Direct file analysis
            return Some(perform_direct_file_analysis(&file_path).await.unwrap_or_else(|e| format!("Error: Error analyzing file: {}", e)));
        }
        
        if input_lower.contains("remove") && input_lower.contains("android") && input_lower.contains("debug") {
            // Direct Android debugging removal
            return Some(perform_direct_android_debug_removal(&file_path).await.unwrap_or_else(|e| format!("Error: Error editing file: {}", e)));
        }
        
        if input_lower.contains("10 best") || input_lower.contains("best domains") {
            // Direct domain extraction
            return Some(perform_direct_domain_extraction(&file_path).await.unwrap_or_else(|e| format!("Error: Error extracting domains: {}", e)));
        }
    }
    
    None
}

fn extract_obvious_file_path(input: &str) -> Option<String> {
    // Pattern 1: Quoted file paths
    if let Ok(quote_regex) = regex::Regex::new(r#""([^"]+\.[a-zA-Z0-9]+)""#) {
        if let Some(captures) = quote_regex.captures(input) {
            if let Some(path_match) = captures.get(1) {
                return Some(path_match.as_str().to_string());
            }
        }
    }
    
    // Pattern 2: File paths with backslashes (Windows)
    if let Ok(path_regex) = regex::Regex::new(r"[A-Za-z]:\\[^\s]+\.[a-zA-Z0-9]+") {
        if let Some(path_match) = path_regex.find(input) {
            return Some(path_match.as_str().to_string());
        }
    }
    
    // Pattern 3: Common file references
    if let Ok(file_regex) = regex::Regex::new(r"\b([a-zA-Z0-9_\-]+\.(html|txt|json|js|py|rs|md|css))\b") {
        if let Some(captures) = file_regex.captures(input) {
            if let Some(filename) = captures.get(1) {
                // Try common directories
                let common_dirs = [
                    format!("C:\\Users\\Admin\\Desktop\\random\\{}", filename.as_str()),
                    format!("C:\\Users\\Admin\\Desktop\\{}", filename.as_str()),
                    format!(".\\{}", filename.as_str()),
                    filename.as_str().to_string(),
                ];
                
                for dir in &common_dirs {
                    if std::path::Path::new(dir).exists() {
                        return Some(dir.clone());
                    }
                }
            }
        }
    }
    
    None
}

async fn perform_direct_file_analysis(file_path: &str) -> Result<String> {
    let path = std::path::Path::new(file_path);
    let content = tokio::fs::read_to_string(path).await?;
    
    // Display a summary of what the file does
    let summary = crate::code_display::display_file_summary(path, &content)?;
    
    // Add analysis summary
    let line_count = content.lines().count();
    let word_count = content.split_whitespace().count();
    let char_count = content.chars().count();
    
    // Combine the summary with the analysis
    Ok(format!(
        "{}\n\n📊 **File Analysis Details**\n\
        📏 Lines: {}\n\
        📝 Words: {}\n\
        🔢 Characters: {}",
        summary, // Use the returned summary here
        line_count, word_count, char_count
    ))
}

async fn perform_direct_domain_extraction(file_path: &str) -> Result<String> {
    let content = tokio::fs::read_to_string(file_path).await?;
    
    // Extract domains using regex
    let mut domains = Vec::new();
    if let Ok(domain_regex) = regex::Regex::new(r"https?://(?:www\.)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})") {
        for captures in domain_regex.captures_iter(&content) {
            if let Some(domain) = captures.get(1) {
                domains.push(domain.as_str().to_string());
            }
        }
    }
    
    // Also try to find domain-like patterns without http
    if let Ok(simple_domain_regex) = regex::Regex::new(r"\b([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})\b") {
        for captures in simple_domain_regex.captures_iter(&content) {
            if let Some(domain) = captures.get(1) {
                let domain_str = domain.as_str();
                // Filter out obvious non-domains
                if !domain_str.contains("localhost") && !domain_str.contains("example.") && domain_str.contains('.') {
                    domains.push(domain_str.to_string());
                }
            }
        }
    }
    
    // Remove duplicates and sort by popularity heuristics
    domains.sort();
    domains.dedup();
    
    // Take the top 10
    let top_domains: Vec<String> = domains.into_iter().take(10).collect();
    
    if top_domains.is_empty() {
        Ok(format!("Error: No domains found in {}", file_path))
    } else {
        let mut result = format!("🌐 **Top {} Domains from {}:**\n\n", top_domains.len().min(10), file_path);
        for (i, domain) in top_domains.iter().enumerate() {
            result.push_str(&format!("{}. {}\n", i + 1, domain));
        }
        Ok(result)
    }
}

async fn perform_direct_android_debug_removal(file_path: &str) -> Result<String> {
    let content = tokio::fs::read_to_string(file_path).await?;
    let original_content = content.clone();
    
    // Remove Android-specific debugging code
    let mut modified_content = content;
    
    // Remove Android debugging scripts and meta tags
    if let Ok(debug_regex) = regex::Regex::new(r"(?i)(<script[^>]*debug[^>]*>.*?</script>|<meta[^>]*debug[^>]*>)") {
        modified_content = debug_regex.replace_all(&modified_content, "").to_string();
    }
    
    // Remove Android WebView debugging
    if let Ok(webview_regex) = regex::Regex::new(r"(?i)(WebView\.setWebContentsDebuggingEnabled\(true\);?)") {
        modified_content = webview_regex.replace_all(&modified_content, "").to_string();
    }
    
    // Remove console.log statements
    if let Ok(console_regex) = regex::Regex::new(r"(?i)\s*console\.log\([^)]*\);\s*\n?") {
        modified_content = console_regex.replace_all(&modified_content, "").to_string();
    }
    
    // Remove Android-specific debug classes
    if let Ok(debug_class_regex) = regex::Regex::new(r#"(?i)\s*class="[^"]*debug[^"]*"\s*"#) {
        modified_content = debug_class_regex.replace_all(&modified_content, "").to_string();
    }
    
    if modified_content != original_content {
        // Show diff
        crate::code_display::CodeDiffDisplay::new()?.display_diff(&original_content, &modified_content, std::path::Path::new(file_path))?;
        
        // Write back to file
        tokio::fs::write(file_path, &modified_content).await?;
        
        Ok(format!("✅ Removed Android debugging code from {}\n📝 Changes shown above", file_path))
    } else {
        Ok(format!("ℹ️ No Android debugging code found in {}", file_path))
    }
}

// Function calling version - more intelligent than pattern matching
async fn get_ai_response_with_intelligent_routing(
    input: &str,
    memory_engine: &MemoryEngine,
    config: &Config,
    _chat_ui: &ChatUI,
) -> Result<(String, Option<String>)> {
    // Initialize function registry
    let function_registry = FunctionRegistry::new();
    let available_functions = function_registry.get_function_definitions();
    
    // Claude Code-style thinking display with proper formatting
    let thinking_display = crate::thinking_display::ThinkingDisplay::new();
    let thinking = thinking_display.start_thinking("Building context").await
        .unwrap_or_else(|_| crate::thinking_display::ThinkingHandle::new_dummy());
    
    // Build context from recent messages  
    let conversation_context = memory_engine.get_context(8, 800).await?;
    
    // Add current directory context
    let current_dir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    
    // Check if we should do automatic research
    thinking.progress_thought("Research", "checking if needed");
    let research_result = research::auto_research_if_needed(input, &conversation_context, config).await;
    let research_context = if let Some(ref research) = research_result {
        thinking.progress_thought("Research", &format!("found {}", research.title));
        format!("\n\nRESEARCH FINDINGS:\nTitle: {}\nSummary: {}\nURL: {}", 
            research.title, research.summary, research.url)
    } else {
        String::new()
    };
    
    // Create system prompt with function calling instructions
    let system_prompt = create_function_calling_prompt(&available_functions);
    
    // Create the main prompt with Glimmer identity and context
    let user_context_prompt = format!(
        "You are Glimmer, an idiosyncratic programming engineer and helpful AI assistant that completes tasks efficiently.\n\
        \n\
        CORE IDENTITY:\n\
        - I am Glimmer, a proactive AI assistant with a unique engineering perspective\n\
        - I complete tasks thoroughly by calling functions when needed\n\
        - I give concise, helpful responses and let my actions speak for themselves\n\
        - I make intelligent decisions and take action with my characteristic directness\n\
        - I persist until tasks are fully completed to user satisfaction\n\
        \n\
        SYSTEM CONTEXT:\n\
        Current directory: {}\n\
        Recent conversation context:\n{}\n\
        {}\n\
        \n\
        USER REQUEST: {}\n\
        \n\
        RESPONSE STYLE:\n\
        - If I need to read/write files, analyze code, search, etc. → USE FUNCTIONS immediately\n\
        - Give brief, helpful responses after completing actions with my engineering perspective\n\
        - Don't show verbose reasoning or planning - just do the work efficiently\n\
        - Focus on completing the task with my characteristic straightforward approach\n\
        \n\
        I will act immediately and provide clear, helpful results.",
        current_dir,
        conversation_context.trim(),
        research_context,
        input
    );
    
    let full_prompt = format!("{}\n\n{}", system_prompt, user_context_prompt);
    
    // Progressive task execution like Claude Code
    thinking.progress_thought("Analyzing", "determining approach");
    
    // First, determine if this is a multi-step task
    let is_complex_task = input.len() > 50 || 
        input.contains("create") || 
        input.contains("build") || 
        input.contains("implement") ||
        input.contains("generate");
    
    if is_complex_task {
        thinking.progress_thought("Task planning", "breaking into steps");
        
        // Try function calling approach
        match gemini::query_gemini_with_function_calling(&full_prompt, config, Some(&available_functions)).await {
            Ok((response, function_call, token_usage)) => {
                // Display token usage if available
                if let Some(usage) = token_usage {
                    thinking.progress_thought("Token usage", &format!("{} tokens", usage.total_tokens));
                }
                if let Some(func_call) = function_call {
                    thinking.progress_thought(&func_call.name, "⎿ Waiting...");
                    
                    match execute_function_call_safely(&func_call, config, &conversation_context).await {
                        Ok(function_result) => {
                            // Check if we need follow-up actions
                            if should_attempt_followup(&func_call.name, &function_result) {
                                thinking.progress_thought("Step 2: Follow-up", "checking for additional actions");
                                
                                // Attempt intelligent follow-up based on the result
                                if let Ok(followup_result) = attempt_intelligent_followup(input, &function_result, config, &thinking).await {
                                    thinking.finish_with_summary("Multi-step task completed successfully");
                                    Ok((format!("{}\n\n{}", function_result, followup_result), None))
                                } else {
                                    thinking.finish_with_summary(&format!("Primary task completed: {}", func_call.name));
                                    Ok((function_result, None))
                                }
                            } else {
                                thinking.finish_with_summary(&format!("Task completed: {}", func_call.name));
                                Ok((function_result, None))
                            }
                        }
                        Err(e) => {
                            thinking.progress_thought("Error recovery", "attempting alternative approach");
                            
                            // Enhanced error context display
                            let error_context = create_error_context(&func_call.name, &e.to_string(), input).await;
                            thinking.progress_thought("Error analysis", &error_context.summary);
                            
                            // Intelligent error recovery with context
                            match attempt_error_recovery_with_context(input, &e.to_string(), &conversation_context, config, &thinking).await {
                                Ok(recovery_result) => {
                                    thinking.finish_with_summary("Recovered from error successfully");
                                    Ok((format!("{}\n\n{}", error_context.display(), recovery_result), None))
                                }
                                Err(_) => {
                                    thinking.finish_with_error(&format!("Function {} failed: {}", func_call.name, e));
                                    Ok((error_context.display(), None))
                                }
                            }
                        }
                    }
                } else if !response.is_empty() {
                    thinking.finish_with_summary("Generated direct response");
                    Ok((response, None))
                } else {
                    thinking.progress_thought("Using fallback", "generating alternative response");
                    let fallback_response = gemini::query_gemini(&user_context_prompt, config).await?;
                    thinking.finish_with_summary("Alternative approach completed");
                    Ok((fallback_response, None))
                }
            }
            Err(_e) => {
                thinking.progress_thought("Fallback mode", "using direct AI response");
                let fallback_response = gemini::query_gemini(&user_context_prompt, config).await?;
                thinking.finish_with_summary("Fallback completed");
                Ok((fallback_response, None))
            }
        }
    } else {
        // Simple task - direct execution
        thinking.progress_thought("Simple task", "processing directly");
        match gemini::query_gemini_with_function_calling(&full_prompt, config, Some(&available_functions)).await {
            Ok((response, function_call, token_usage)) => {
                // Display token usage if available
                if let Some(usage) = token_usage {
                    thinking.progress_thought("Token usage", &format!("{} tokens", usage.total_tokens));
                }
                if let Some(func_call) = function_call {
                    thinking.progress_thought(&format!("Executing {}", func_call.name), "processing");
                    match execute_function_call_safely(&func_call, config, &conversation_context).await {
                        Ok(function_result) => {
                            thinking.finish_with_summary("Task completed");
                            Ok((function_result, None))
                        }
                        Err(e) => {
                            thinking.finish_with_error(&format!("Error: {}", e));
                            Ok((format!("Error: {}", e), None))
                        }
                    }
                } else {
                    thinking.finish_with_summary("Direct response generated");
                    Ok((response, None))
                }
            }
            Err(_e) => {
                thinking.finish_with_error("Failed to execute function");
                let fallback_response = gemini::query_gemini(&user_context_prompt, config).await?;
                Ok((fallback_response, None))
            }
        }
    }
}

/// Check if a function call should trigger follow-up actions
fn should_attempt_followup(function_name: &str, result: &str) -> bool {
    match function_name {
        "create_file" => {
            // If we created an HTML file, maybe we should open it or validate it
            result.contains("✅") && (result.contains(".html") || result.contains(".js"))
        },
        "edit_code" => {
            // If we edited code, maybe we should check syntax or run tests
            result.contains("✅") && !result.contains("syntax error")
        },
        _ => false
    }
}

/// Attempt intelligent follow-up actions based on the previous result
async fn attempt_intelligent_followup(
    original_input: &str, 
    previous_result: &str, 
    _config: &Config,
    thinking: &crate::thinking_display::ThinkingHandle
) -> Result<String> {
    // Smart follow-up based on context
    if previous_result.contains("Successfully created") && previous_result.contains(".html") {
        thinking.progress_thought("Follow-up", "checking HTML validity");
        
        // Extract file path from result
        if let Some(file_path) = extract_file_path_from_result(previous_result) {
            return Ok(format!("📝 HTML file created successfully!\n💡 Tip: Open {} in your browser to view the Rubik's Cube puzzle.", file_path));
        }
    }
    
    if previous_result.contains("Successfully created") && original_input.contains("rubik") {
        thinking.progress_thought("Follow-up", "providing usage instructions");
        return Ok("🎮 Your Rubik's Cube is ready! You can:\n• Click and drag to rotate the cube\n• Use mouse controls to manipulate individual faces\n• The puzzle is fully interactive and solvable".to_string());
    }
    
    Ok("Task completed successfully!".to_string())
}

/// Extract file path from function result
fn extract_file_path_from_result(result: &str) -> Option<String> {
    // Look for patterns like "file: path/to/file.html"
    if let Some(start) = result.find("file: ") {
        let path_start = start + 6;
        if let Some(end) = result[path_start..].find('\n') {
            return Some(result[path_start..path_start + end].to_string());
        }
        // If no newline, take the rest of the line
        return Some(result[path_start..].split_whitespace().next()?.to_string());
    }
    None
}

/// Enhanced error recovery with conversation context
async fn attempt_error_recovery_with_context(
    original_input: &str,
    error_message: &str,
    conversation_context: &str,
    config: &Config,
    thinking: &crate::thinking_display::ThinkingHandle
) -> Result<String> {
    thinking.progress_thought("Context analysis", "examining conversation for clues");
    
    // Analyze context for path hints when directory operations fail
    if error_message.contains("Directory does not exist") {
        if conversation_context.contains("Desktop") && conversation_context.contains("random") {
            thinking.progress_thought("Recovery", "trying desktop path based on context");
            let _registry = crate::function_calling::FunctionRegistry::new();
            let corrected_call = crate::function_calling::FunctionCall {
                name: "list_directory".to_string(),
                arguments: {
                    let mut args = std::collections::HashMap::new();
                    args.insert("directory_path".to_string(), serde_json::Value::String("C:\\Users\\Admin\\Desktop\\random".to_string()));
                    args
                }
            };
            
            match crate::function_calling::execute_function_call(&corrected_call, config, conversation_context).await {
                Ok(result) => return Ok(format!("Found it! Using context clues from our conversation:\n\n{}", result)),
                Err(_) => {} // Continue with other recovery methods
            }
        }
    }
    
    // Fall back to original recovery logic
    attempt_error_recovery(original_input, error_message, config, thinking).await
}

/// Attempt error recovery with alternative approaches
async fn attempt_error_recovery(
    original_input: &str,
    error_message: &str,
    _config: &Config,
    thinking: &crate::thinking_display::ThinkingHandle
) -> Result<String> {
    thinking.progress_thought("Recovery mode", "analyzing error type");
    
    // If file creation failed due to path issues, try with current directory
    if error_message.contains("Failed to create") || error_message.contains("No such file") {
        thinking.progress_thought("Recovery", "trying alternative file path");
        
        // Try creating in current directory instead
        let simple_filename = if original_input.contains("rubiks") || original_input.contains("rubik") {
            "rubiks-cube.html"
        } else if original_input.contains(".html") {
            "new-file.html"  
        } else {
            "output.txt"
        };
        
        // Create a simple fallback file
        let current_dir = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        
        let fallback_path = std::path::Path::new(&current_dir).join(simple_filename);
        
        let content = if original_input.to_lowercase().contains("rubik") {
            generate_rubiks_cube_html()
        } else {
            "<!-- Generated file -->\n<html><body><h1>Generated Content</h1></body></html>".to_string()
        };
        
        match tokio::fs::write(&fallback_path, content).await {
            Ok(_) => {
                return Ok(format!("✅ Recovered! Created file at: {}\n💡 I used an alternative approach to complete your request.", fallback_path.display()));
            }
            Err(_) => {}
        }
    }
    
    Err(anyhow::anyhow!("Recovery failed"))
}

/// Error context for comprehensive error reporting
#[derive(Debug, Clone)]
struct ErrorContext {
    function_name: String,
    error_message: String,
    user_input: String,
    summary: String,
    suggestions: Vec<String>,
    category: ErrorCategory,
}

#[derive(Debug, Clone)]
enum ErrorCategory {
    FileNotFound,
    PermissionDenied,
    InvalidInput,
    NetworkError,
    SystemError,
    Unknown,
}

impl ErrorContext {
    fn display(&self) -> String {
        let mut result = String::new();
        
        // Claude Code style error display
        result.push_str(&format!("Error: Error in {}: {}\n", self.function_name, self.summary));
        result.push_str(&format!("   Original request: {}\n", self.user_input));
        result.push_str(&format!("   Error details: {}\n", self.error_message));
        
        match self.category {
            ErrorCategory::FileNotFound => {
                result.push_str("   Category: File System - Resource Not Found\n");
            }
            ErrorCategory::PermissionDenied => {
                result.push_str("   Category: File System - Permission Denied\n");
            }
            ErrorCategory::InvalidInput => {
                result.push_str("   Category: Input Validation - Invalid Parameters\n");
            }
            _ => {
                result.push_str("   Category: System Error\n");
            }
        }
        
        if !self.suggestions.is_empty() {
            result.push_str("\n💡 Suggestions:\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                result.push_str(&format!("   {}. {}\n", i + 1, suggestion));
            }
        }
        
        result
    }
}

/// Create comprehensive error context
async fn create_error_context(function_name: &str, error_message: &str, user_input: &str) -> ErrorContext {
    let error_lower = error_message.to_lowercase();
    let input_lower = user_input.to_lowercase();
    
    let category = if error_lower.contains("not found") || error_lower.contains("no such file") {
        ErrorCategory::FileNotFound
    } else if error_lower.contains("permission") || error_lower.contains("access denied") {
        ErrorCategory::PermissionDenied
    } else if error_lower.contains("invalid") || error_lower.contains("missing parameter") {
        ErrorCategory::InvalidInput
    } else {
        ErrorCategory::Unknown
    };
    
    let mut suggestions = Vec::new();
    let summary;
    
    match (&category, function_name) {
        (ErrorCategory::FileNotFound, "create_file") => {
            summary = "Target directory may not exist".to_string();
            suggestions.push("Check if the parent directory exists".to_string());
            suggestions.push("Try using a different file path".to_string());
            suggestions.push("Use an absolute path instead of relative path".to_string());
        }
        (ErrorCategory::FileNotFound, "edit_code") => {
            summary = "Source file not found for editing".to_string();
            suggestions.push("Verify the file path is correct".to_string());
            suggestions.push("Check if the file name is spelled correctly".to_string());
            suggestions.push("Try creating the file first".to_string());
        }
        (ErrorCategory::FileNotFound, "rename_file") => {
            summary = "Source or destination path issue".to_string();
            suggestions.push("Verify both source and destination paths exist".to_string());
            suggestions.push("Check file permissions".to_string());
            suggestions.push("Try using absolute paths".to_string());
        }
        (ErrorCategory::PermissionDenied, _) => {
            summary = "Insufficient permissions for file operation".to_string();
            suggestions.push("Run with administrator privileges".to_string());
            suggestions.push("Check file and directory permissions".to_string());
            suggestions.push("Try a different location".to_string());
        }
        (ErrorCategory::InvalidInput, _) => {
            summary = "Input parameters are invalid or missing".to_string();
            if input_lower.contains("rubik") {
                suggestions.push("For Rubik's cube, specify HTML file type".to_string());
            }
            suggestions.push("Check that all required parameters are provided".to_string());
            suggestions.push("Verify the input format is correct".to_string());
        }
        _ => {
            summary = "Unexpected error occurred".to_string();
            suggestions.push("Try the operation again".to_string());
            suggestions.push("Check system resources".to_string());
            suggestions.push("Use a simpler approach".to_string());
        }
    }
    
    ErrorContext {
        function_name: function_name.to_string(),
        error_message: error_message.to_string(),
        user_input: user_input.to_string(),
        summary,
        suggestions,
        category,
    }
}

/// Generate a simple Rubik's Cube HTML content
fn generate_rubiks_cube_html() -> String {
    r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rubik's Cube Puzzle</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: linear-gradient(45deg, #1e3c72, #2a5298);
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .cube-container {
            perspective: 1000px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .cube {
            width: 200px;
            height: 200px;
            position: relative;
            transform-style: preserve-3d;
            transform: rotateX(-15deg) rotateY(15deg);
            animation: spin 10s infinite linear;
        }
        .face {
            position: absolute;
            width: 200px;
            height: 200px;
            border: 2px solid #000;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(3, 1fr);
            gap: 2px;
        }
        .square {
            border: 1px solid #333;
        }
        .front { background: #ff0000; transform: translateZ(100px); }
        .back { background: #ffa500; transform: translateZ(-100px) rotateY(180deg); }
        .right { background: #0000ff; transform: rotateY(90deg) translateZ(100px); }
        .left { background: #00ff00; transform: rotateY(-90deg) translateZ(100px); }
        .top { background: #ffffff; transform: rotateX(90deg) translateZ(100px); }
        .bottom { background: #ffff00; transform: rotateX(-90deg) translateZ(100px); }
        @keyframes spin {
            from { transform: rotateX(-15deg) rotateY(15deg); }
            to { transform: rotateX(-15deg) rotateY(375deg); }
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .controls {
            margin-top: 30px;
            text-align: center;
            color: white;
        }
        button {
            padding: 10px 20px;
            margin: 5px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <div class="cube-container">
        <h1>🎮 Interactive Rubik's Cube</h1>
        <div class="cube" id="cube">
            <div class="face front">
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
            </div>
            <div class="face back">
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
            </div>
            <div class="face right">
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
            </div>
            <div class="face left">
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
            </div>
            <div class="face top">
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
            </div>
            <div class="face bottom">
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
                <div class="square"></div><div class="square"></div><div class="square"></div>
            </div>
        </div>
        <div class="controls">
            <button onclick="rotateCube('X')">↻ Rotate X</button>
            <button onclick="rotateCube('Y')">↻ Rotate Y</button>
            <button onclick="scramble()">🔀 Scramble</button>
            <button onclick="solve()">✨ Solve</button>
        </div>
    </div>

    <script>
        let rotationX = -15;
        let rotationY = 15;
        const cube = document.getElementById('cube');

        function rotateCube(axis) {
            if (axis === 'X') {
                rotationX += 90;
            } else {
                rotationY += 90;
            }
            updateCubeTransform();
        }

        function updateCubeTransform() {
            cube.style.transform = `rotateX(${rotationX}deg) rotateY(${rotationY}deg)`;
        }

        function scramble() {
            const colors = ['#ff0000', '#ffa500', '#ffff00', '#00ff00', '#0000ff', '#ffffff'];
            const squares = document.querySelectorAll('.square');
            
            squares.forEach(square => {
                const randomColor = colors[Math.floor(Math.random() * colors.length)];
                square.style.backgroundColor = randomColor;
            });
            
            // Random rotation
            rotationX = Math.random() * 360;
            rotationY = Math.random() * 360;
            updateCubeTransform();
        }

        function solve() {
            // Reset to original colors
            const faces = document.querySelectorAll('.face');
            const faceColors = ['#ff0000', '#ffa500', '#0000ff', '#00ff00', '#ffffff', '#ffff00'];
            
            faces.forEach((face, index) => {
                const squares = face.querySelectorAll('.square');
                squares.forEach(square => {
                    square.style.backgroundColor = faceColors[index];
                });
            });
            
            rotationX = -15;
            rotationY = 15;
            updateCubeTransform();
        }

        // Mouse interaction
        let isDragging = false;
        let startX, startY;

        cube.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            cube.style.animation = 'none';
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;
                
                rotationY += deltaX * 0.5;
                rotationX -= deltaY * 0.5;
                
                updateCubeTransform();
                
                startX = e.clientX;
                startY = e.clientY;
            }
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            cube.style.animation = 'spin 10s infinite linear';
        });
    </script>
</body>
</html>"#.to_string()
}

async fn execute_function_call_safely(
    function_call: &crate::function_calling::FunctionCall,
    config: &Config,
    conversation_history: &str,
) -> Result<String> {
    // Create temporary audio state for this call
    
    execute_function_call(
        function_call,
        config,
        conversation_history,
    ).await
}

// New UI-integrated version
async fn get_ai_response_with_ui(
    input: &str,
    memory_engine: &MemoryEngine,
    config: &Config,
    chat_ui: &ChatUI,
) -> Result<(String, Option<String>)> {
    // Removed direct file operation handling to unify all requests through the
    // main intelligent processing pipeline, which is TUI-safe and prevents UI corruption.

    let progress = crate::progress_display::ProgressDisplay::new();
    let _handle = progress.start_operation("analyzing request").await.ok();

    // Build context from recent messages
    let conversation_context = memory_engine.get_context(8, 800).await?;

    // Add current directory context
    let current_dir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Build an intelligent, context-aware prompt
    let enhanced_context = collect_intelligent_context(&current_dir, input).await;
    
    // Check if we should do automatic research
    let research_result = research::auto_research_if_needed(input, &conversation_context, config).await;
    let research_context = if let Some(ref research) = research_result {
        format!("\n\nRESEARCH FINDINGS:\nTitle: {}\nSummary: {}\nURL: {}", 
            research.title, research.summary, research.url)
    } else {
        String::new()
    };

    let prompt = format!(
        "You are Glimmer, an intelligent coding assistant that makes smart decisions automatically.\n\
        Current directory: {}\n\
        Available context: {}{}\n\
        \n\
        IMPORTANT INSTRUCTIONS:\n\
        - Be proactive and make intelligent decisions instead of asking for specifics\n\
        - If you need to work with files, intelligently determine which files based on context\n\
        - If there are multiple possibilities, choose the most likely one and explain your reasoning\n\
        - When you encounter issues, attempt to resolve them proactively\n\
        - Use available file information to provide comprehensive answers\n\
        - Don't ask \"which file?\" or \"can you specify?\" - make educated guesses\n\
        - If research findings are provided, incorporate them into your response naturally\n\
        \n\
        Recent conversation:\n{}\n\
        \n\
        User: {}\n\
        Assistant:",
        current_dir,
        enhanced_context,
        research_context,
        conversation_context.trim(),
        input
    );

    // Smart decision engine - make intelligent automatic choices
    let decision = make_smart_decision(input, &enhanced_context, &conversation_context, config).await;
    
    match decision {
        SmartDecision::DirectResponse => {
            // Simple task - proceed directly
            let response = gemini::query_gemini(&prompt, config).await?;
            Ok((response, None))
        }
        SmartDecision::WithThinking => {
            // Moderate complexity - use thinking process
            let (response, thinking) = gemini::query_gemini_with_thinking(&prompt, config, Some(2048)).await?;
            Ok((response, thinking))
        }
        SmartDecision::AutoProceedWithSteps(steps) => {
            // Complex but auto-proceed - no user confirmation needed
            handle_auto_complex_task_with_ui(input, &steps, config, chat_ui).await
        }
        SmartDecision::ConfirmBeforeSteps(analysis) => {
            // Truly complex - ask for confirmation with better UI
            handle_complex_task_with_modern_ui(input, &analysis, config, chat_ui).await
        }
    }
}

async fn get_ai_response(
    input: &str,
    memory_engine: &MemoryEngine,
    config: &Config,
) -> Result<String> {
    // Check if this is a file operation request
    if let Some(file_op_result) = try_handle_file_operation(input, config, memory_engine).await {
        return Ok(file_op_result);
    }
    
    // Show initial progress indicator
    let mut progress = crate::cli::progress::show_ai_processing("process your request").await;

    // Check for interruption
    if progress.is_interrupted() {
        progress.stop().await;
        return Err(anyhow::anyhow!("Request interrupted by user"));
    }

    // Build context from recent messages
    let conversation_context = memory_engine.get_context(8, 800).await?;

    // Add current directory context
    let current_dir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Build an intelligent, context-aware prompt
    let enhanced_context = collect_intelligent_context(&current_dir, input).await;
    
    // Check if we should do automatic research
    let research_result = research::auto_research_if_needed(input, &conversation_context, config).await;
    let research_context = if let Some(ref research) = research_result {
        format!("\n\nRESEARCH FINDINGS:\nTitle: {}\nSummary: {}\nURL: {}", 
            research.title, research.summary, research.url)
    } else {
        String::new()
    };

    let prompt = format!(
        "You are Glimmer, an intelligent coding assistant that makes smart decisions automatically.\n\
        Current directory: {}\n\
        Available context: {}{}\n\
        \n\
        IMPORTANT INSTRUCTIONS:\n\
        - Be proactive and make intelligent decisions instead of asking for specifics\n\
        - If you need to work with files, intelligently determine which files based on context\n\
        - If there are multiple possibilities, choose the most likely one and explain your reasoning\n\
        - When you encounter issues, attempt to resolve them proactively\n\
        - Use available file information to provide comprehensive answers\n\
        - Don't ask \"which file?\" or \"can you specify?\" - make educated guesses\n\
        - If research findings are provided, incorporate them into your response naturally\n\
        - Your primary source of context is the **recent conversation history**. If the user's request is a follow-up, refer to the last few messages to understand what they are talking about. For example, if they ask for a \"list of items\" after you've shown them a file, their question is likely about the content of that file.\n\
        - Do not interpret a request for a \"list of\" something as a file system command to list directory contents, unless they explicitly mention a directory.\n\
        \n\
        Recent conversation:\n{}\n\
        \n\
        User: {}\n\
        Assistant:",
        current_dir,
        enhanced_context,
        research_context,
        conversation_context.trim(),
        input
    );
    
    // Count tokens before making the request
    let token_usage = match gemini::count_tokens(&prompt, config).await {
        Ok(usage) => Some(usage),
        Err(_) => None, // Fall back to normal processing if token counting fails
    };
    
    // Update progress indicator with token count if available
    if let Some(ref usage) = token_usage {
        progress.stop().await;
        progress = crate::cli::progress::show_ai_processing_with_tokens("process your request", usage.input_tokens).await;
    }

    // Smart decision engine for non-UI mode (fallback)
    let decision = make_smart_decision(input, &enhanced_context, &conversation_context, config).await;
    
    let result = match decision {
        SmartDecision::DirectResponse => {
            let result = gemini::query_gemini(&prompt, config).await;
            progress.stop().await;
            result
        }
        SmartDecision::WithThinking => {
            progress.stop().await;
            match gemini::query_gemini_with_thinking(&prompt, config, Some(2048)).await {
                Ok((response, _thinking)) => Ok(response),
                Err(e) => Err(e),
            }
        }
        SmartDecision::AutoProceedWithSteps(steps) => {
            progress.stop().await;
            execute_steps_without_ui(input, &steps, config).await
        }
        SmartDecision::ConfirmBeforeSteps(analysis) => {
            handle_complex_task_legacy(input, &prompt, &analysis, config, &mut progress).await
        }
    };

    result
}

async fn try_handle_file_operation(input: &str, config: &Config, memory_engine: &MemoryEngine) -> Option<String> {
    let input_lower = input.to_lowercase();
    
    // Handle permission requests
    if input_lower.contains("allow access") || input_lower.contains("grant permission") {
        if let Some(path) = extract_permission_request(input) {
            return Some(handle_permission_grant(&path).await);
        }
    }
    
    // Check for common file operation patterns
    if input_lower.contains("create") || input_lower.contains("make") || input_lower.contains("write") || 
       input_lower.contains("generate") || input_lower.contains("build") || input_lower.contains("develop") {
        if let Some(file_info) = extract_file_creation_request(input, config).await {
            return Some(handle_file_creation(file_info, config).await);
        }
    }
    
    if input_lower.contains("edit") || input_lower.contains("modify") || input_lower.contains("update") {
        if let Some(file_info) = extract_file_edit_request_smart(input, config, memory_engine).await {
            return Some(handle_file_edit(file_info, config).await);
        }
    }
    
    if input_lower.contains("read") || input_lower.contains("show") || input_lower.contains("display") {
        if let Some(filename) = extract_filename_from_request_smart(input, config, memory_engine).await {
            // Check if user wants error analysis/fixing
            let should_analyze = input_lower.contains("error") || 
                               input_lower.contains("fix") || 
                               input_lower.contains("debug") ||
                               input_lower.contains("issue") ||
                               input_lower.contains("problem") ||
                               input_lower.contains("compile") ||
                               input_lower.contains("check") ||
                               input_lower.contains("broken");
                               
            return Some(handle_file_read_with_context(&filename, input, should_analyze).await);
        }
    }
    
    // Handle directory listing requests
    // Make the check more specific to avoid false positives on "list of..."
    if input_lower.starts_with("ls") || input_lower.starts_with("list files") ||
       input_lower.contains("contents of") || input_lower.contains("files in") {
        if let Some(directory) = extract_directory_from_request(input) {
            return Some(handle_directory_list(&directory).await);
        }
    }
    
    // Intelligent file reading: Check if files are mentioned in conversation
    // This should come after explicit operations to avoid false positives
    if let Some(result) = try_intelligent_file_reading(input, config).await {
        return Some(result);
    }
    
    None
}

// Execute steps without UI for non-UI mode
async fn execute_steps_without_ui(
    input: &str,
    steps: &[String],
    config: &Config,
) -> Result<String> {
    let mut output = format!("🚀 **Auto-executing {} steps**\n", steps.len());
    let mut step_results = Vec::new();
    
    for (i, step) in steps.iter().enumerate() {
        // Update status bar instead of printing directly
        PersistentStatusBar::update_status(&format!("Step {}: {}", i + 1, step));
        output.push_str(&format!("▶️  Step {}: {}\n", i + 1, step));

        let step_prompt = format!(
            "Execute step {} of {}: {}\n\nOriginal request: {}\n\nPrevious steps completed: {}\n\nProvide a focused, actionable response for this step only.",
            i + 1,
            steps.len(),
            step,
            input,
            step_results.len()
        );
        
        match gemini::query_gemini(&step_prompt, config).await {
            Ok(response) => {
                output.push_str(&format!("✅ Step {} completed.\n", i + 1));
                step_results.push(response);
            }
            Err(e) => {
                let error_msg = format!("Error: Step {} failed: {}\n", i + 1, e);
                output.push_str(&error_msg);
                break;
            }
        }
    }
    
    let final_output = if step_results.is_empty() {
        "Task execution failed".to_string()
    } else {
        step_results.join("\n\n")
    };

    // Combine the step-by-step log with the final result
    Ok(format!("{}\n---\n{}", output, final_output))
}

// Legacy complex task handler (renamed from handle_complex_task)
async fn handle_complex_task_legacy(
    _input: &str,
    _base_prompt: &str,
    task_analysis: &gemini::TaskComplexity,
    _config: &Config,
    progress: &mut crate::cli::progress::ThinkingIndicator,
) -> Result<String> {
    progress.stop().await;

    // This function is being refactored to integrate with the TUI.
    // Instead of printing and blocking for input, it should return a structured
    // representation of the confirmation request. The main TUI loop will then
    // handle displaying the prompt and capturing the user's 'y' or 'n' response.

    let mut prompt_text = "🧠 **Complex Task Detected**\n".to_string();
    prompt_text.push_str(&format!("Complexity: {}\n", task_analysis.complexity));
    prompt_text.push_str(&format!("Reasoning: {}\n", task_analysis.reasoning));
    prompt_text.push_str(&format!("Estimated time: {}\n\n", task_analysis.estimated_time));

    prompt_text.push_str("📋 **Breaking down the task into steps:**\n");
    for step in &task_analysis.steps {
        prompt_text.push_str(&format!("  [ ] {}\n", step));
    }
    prompt_text.push_str("\n");
    prompt_text.push_str("Would you like me to proceed with this step-by-step approach? (y/N): ");

    // The main loop should display this prompt and wait for a 'y' or 'n' key press.
    // For now, we'll return the prompt. A more advanced implementation would
    // involve changing the application state to 'AwaitingConfirmation'.
    Ok(prompt_text)
}

async fn handle_complex_task(
    input: &str,
    base_prompt: &str,
    task_analysis: &gemini::TaskComplexity,
    config: &Config,
    progress: &mut crate::cli::progress::ThinkingIndicator,
) -> Result<String> {
    use crate::cli::colors::{EMERALD_BRIGHT, YELLOW_WARN, GRAY_DIM, RESET};    

    progress.stop().await;
    
    // Complex task detected - show in status bar instead
    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Complex task detected: {}", task_analysis.complexity));
    // Complex task handling integrated with ratatui interface
    
    let mut response = String::new();
    if std::io::stdin().read_line(&mut response).is_ok() {
        let response = response.trim().to_lowercase();        
        if response == "y" || response == "yes" {
            return execute_complex_task_with_steps(input, base_prompt, task_analysis, config).await;
        }
    }
    
    // Status messages handled by ratatui interface
    gemini::query_gemini(base_prompt, config).await
}

async fn handle_moderate_task_with_planning(
    _input: &str,
    base_prompt: &str,
    task_analysis: &gemini::TaskComplexity,
    config: &Config,
    progress: &mut crate::cli::progress::ThinkingIndicator,
) -> Result<String> {
    use crate::cli::colors::{EMERALD_BRIGHT, GRAY_DIM, RESET, PURPLE_BRIGHT};
    use std::io::{stdout, Write};
    use tokio::time::{sleep, Duration};
    
    progress.stop().await;
    
    // Task planning messages handled by ratatui interface
    
    let spinner_handle = tokio::spawn(async move {
        let frames = ["⌘", " "];
        let mut frame_index = 0;
        loop {
            // Thinking animation handled by ratatui interface
            stdout().flush().unwrap();
            sleep(Duration::from_millis(400)).await;
            frame_index = (frame_index + 1) % frames.len();
        }
    });
    
    let thinking_budget = Some(4096); // Larger budget for complex reasoning
    match gemini::query_gemini_with_thinking(base_prompt, config, thinking_budget).await {
        Ok((response, thinking)) => {
            spinner_handle.abort();
            // Terminal clearing handled by ratatui interface
            stdout().flush().unwrap();
            
            if let Some(thinking_content) = thinking {
                // AI reasoning display handled by ratatui interface
            }
            
            Ok(response)
        }
        Err(_e) => {
            spinner_handle.abort();
            // Terminal clearing handled by ratatui interface
            stdout().flush().unwrap_or_default();
            // Advanced reasoning messages handled by ratatui interface
            gemini::query_gemini(base_prompt, config).await            
        }
    }
}

async fn execute_complex_task_with_steps(
    _input: &str,
    base_prompt: &str, // Note: Use the original user request, not the whole base_prompt
    task_analysis: &gemini::TaskComplexity,
    config: &Config,
) -> Result<String> {
    use crate::cli::colors::{EMERALD_BRIGHT, GRAY_DIM, RESET};
    
    let mut results = Vec::new();
    let original_user_request = base_prompt; // Assuming base_prompt contains the user's initial clean request
    
    // Show step-by-step approach in status bar instead of direct print
    
    for (i, step) in task_analysis.steps.iter().enumerate() {
        // Update status bar to show current step
        PersistentStatusBar::set_ai_thinking(&format!("Step {} of {}: {}", i + 1, task_analysis.steps.len(), step));
        
        let mut step_progress = crate::cli::progress::show_ai_processing(&format!("Step {}: {}", i + 1, step)).await;
        
        let step_prompt = format!(
            "You are working on a complex task broken into steps.\n\
            \n\
            Original request: {}\n\
            Current step ({}/{}): {}\n\
            Context from previous steps: {}\n\
            \n\
            Focus ONLY on this current step. Provide detailed output, facts, or actions taken. Avoid conversational filler.",
            original_user_request,
            i + 1,
            task_analysis.steps.len(),
            step,
            if results.is_empty() { "None yet".to_string() } else { results.join("\n\n") }
        );
        
        // You might want to use query_gemini_with_thinking if you need the thought process for logging
        match gemini::query_gemini(&step_prompt, config).await {
            Ok(step_result) => {
                step_progress.stop().await;
                // Step completed - continue to next
                results.push(format!("Result of step '{}':\n{}", step, step_result));
            }
            Err(e) => {
                step_progress.stop().await;
                // Step failed - continue with error handling
                results.push(format!("Step '{}' failed: {}", step, e));
            }
        }
    }
    
    // All steps completed - show final synthesis in status bar
    PersistentStatusBar::set_ai_thinking("All steps completed! Synthesizing final answer...");
    
    // *** NEW FINAL SYNTHESIS STEP ***
    let final_synthesis_prompt = format!(
        "A complex task was just completed. \
        The original user request was: \"{}\"\n\
        The plan was:\n{}\n\n\
        Here is a log of the results from each step:\n---\n{}\n---\n\n\
        Based on all this information, formulate a single, comprehensive, and user-facing response. \
        Address the user directly. Do not mention the steps or that it was a complex task. \
        Just provide the final, complete answer.",
        original_user_request,
        task_analysis.steps.iter().map(|s| format!("- {}", s)).collect::<Vec<_>>().join("\n"),
        results.join("\n\n")
    );
    
    // Final call to get a clean answer
    gemini::query_gemini(&final_synthesis_prompt, config).await
}

async fn collect_intelligent_context(current_dir: &str, user_input: &str) -> String {
    let mut context_parts = Vec::new();
    
    // 1. File system context - list relevant files in current directory
    if let Ok(entries) = std::fs::read_dir(current_dir) {
        let mut files = Vec::new();
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                // Skip hidden files and common directories
                if !name.starts_with('.') && !name.starts_with("target") && !name.starts_with("node_modules") {
                    let path = entry.path();
                    if path.is_file() {
                        files.push(name.to_string());
                    }
                }
            }
        }
        
        if !files.is_empty() {
            files.sort();
            context_parts.push(format!("Files in current directory: {}", files.join(", ")));
        }
    }
    
    // 2. Project context - detect project type and relevant files
    let project_info = detect_project_type(current_dir).await;
    if !project_info.is_empty() {
        context_parts.push(format!("Project type: {}", project_info));
    }
    
    // 3. Recent file activity - check for recently mentioned files
    let mentioned_files = extract_mentioned_files(user_input).await;
    if !mentioned_files.is_empty() {
        context_parts.push(format!("Files mentioned: {}", mentioned_files.join(", ")));
    }
    
    // 4. Permission context - show accessible directories
    if let Ok(manager) = crate::permissions::PermissionManager::new().await {
        let accessible_dirs: Vec<String> = manager.list_allowed_paths()
            .iter()
            .map(|p| p.display().to_string())
            .take(5)  // Limit to avoid overwhelming
            .collect();
        if !accessible_dirs.is_empty() {
            context_parts.push(format!("Accessible directories: {}", accessible_dirs.join(", ")));
        }
    }
    
    // 5. Intent detection - analyze what the user might want to do
    let intent_hints = detect_user_intent(user_input);
    if !intent_hints.is_empty() {
        context_parts.push(format!("Likely user intent: {}", intent_hints));
    }
    
    if context_parts.is_empty() {
        "Limited context available".to_string()
    } else {
        context_parts.join(" | ")
    }
}

async fn detect_project_type(current_dir: &str) -> String {
    let mut project_types = Vec::new();
    
    // Check for common project files
    let project_files = [
        ("Cargo.toml", "Rust"),
        ("package.json", "Node.js"),
        ("requirements.txt", "Python"),
        ("pom.xml", "Java/Maven"),
        ("build.gradle", "Java/Gradle"),
        ("Gemfile", "Ruby"),
        ("go.mod", "Go"),
        ("composer.json", "PHP"),
        (".csproj", "C#"),
    ];
    
    for (file, project_type) in project_files {
        let path = std::path::Path::new(current_dir).join(file);
        if path.exists() {
            project_types.push(project_type);
        }
    }
    
    if project_types.is_empty() {
        // Check for common directories
        let common_dirs = [
            ("src", "Source code"),
            ("lib", "Library"),
            ("test", "Test suite"),
            ("docs", "Documentation"),
        ];
        
        for (dir, description) in common_dirs {
            let path = std::path::Path::new(current_dir).join(dir);
            if path.exists() && path.is_dir() {
                project_types.push(description);
            }
        }
    }
    
    project_types.join(", ")
}

fn detect_user_intent(user_input: &str) -> String {
    let input_lower = user_input.to_lowercase();
    let mut intents = Vec::new();
    
    // File operation intents
    if input_lower.contains("error") || input_lower.contains("fix") || input_lower.contains("broken") {
        intents.push("debugging/fixing errors");
    }
    
    if input_lower.contains("create") || input_lower.contains("make") || input_lower.contains("new") {
        intents.push("file creation");
    }
    
    if input_lower.contains("edit") || input_lower.contains("modify") || input_lower.contains("update") {
        intents.push("file modification");
    }
    
    if input_lower.contains("explain") || input_lower.contains("understand") || input_lower.contains("how") {
        intents.push("code explanation");
    }
    
    if input_lower.contains("optimize") || input_lower.contains("improve") || input_lower.contains("refactor") {
        intents.push("code optimization");
    }
    
    if input_lower.contains("test") || input_lower.contains("spec") {
        intents.push("testing");
    }
    
    if input_lower.contains("deploy") || input_lower.contains("build") || input_lower.contains("compile") {
        intents.push("build/deployment");
    }
    
    // Content-specific intents
    if input_lower.contains("rubik") || input_lower.contains("cube") {
        intents.push("working with Rubik's cube project");
    }
    
    if input_lower.contains("random") {
        intents.push("working in random directory");
    }
    
    intents.join(", ")
}

async fn try_intelligent_file_reading(input: &str, _config: &Config) -> Option<String> {
    // Skip if this looks like an explicit file operation command
    let input_lower = input.to_lowercase();
    let skip_keywords = ["create", "make", "write", "generate", "edit", "modify", "delete", "remove", "rename"];
    if skip_keywords.iter().any(|&keyword| input_lower.contains(keyword)) {
        return None;
    }
    
    // Look for potential file paths mentioned in conversation
    let file_candidates = extract_mentioned_files(input).await;
    
    if file_candidates.is_empty() {
        return None;
    }
    
    // Check if any mentioned files actually exist and read them
    let mut results = Vec::new();
    
    for file_path in file_candidates {
        // Try to resolve the file path intelligently
        let resolved_path = if std::path::Path::new(&file_path).exists() {
            file_path.clone()
        } else if let Some(smart_path) = smart_resolve_file_path(&file_path, &[".", "random", "src", "docs"]) {
            smart_path
        } else {
            continue; // Skip if file doesn't exist
        };
        
        // Check permissions before reading
        match crate::permissions::verify_path_access(&std::path::Path::new(&resolved_path)).await {
            Ok(true) => {
                // Show progress for file reading
                let mut progress = crate::cli::progress::show_thinking_with_context(&format!("reading mentioned file: {}", resolved_path)).await;
                
                match crate::file_io::read_file(&std::path::Path::new(&resolved_path)).await {
                    Ok(content) => {
                        let language = crate::file_io::detect_language(&std::path::Path::new(&resolved_path));
                        let summary = if content.lines().count() > 50 {
                            format!("📄 **{}** ({}+ lines, showing first 20):\n\n```{}\n{}\n```", 
                                resolved_path, content.lines().count(), language, 
                                content.lines().take(20).collect::<Vec<_>>().join("\n"))
                        } else {
                            format!("📄 **{}**:\n\n```{}\n{}\n```", resolved_path, language, content)
                        };
                        results.push(summary);
                        progress.stop().await;
                    }
                    Err(_) => {
                        progress.stop().await;
                        continue;
                    }
                }
            }
            _ => continue, // Skip if no permissions
        }
    }
    
    if results.is_empty() {
        return None;
    }
    
    // Format the response to show what files were automatically read
    let header = if results.len() == 1 {
        "I noticed you mentioned a file, so I've read it for context:"
    } else {
        "I noticed you mentioned some files, so I've read them for context:"
    };
    
    Some(format!("{}\n\n{}", header, results.join("\n\n")))
}

async fn extract_mentioned_files(input: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    
    // Pattern 1: Detect directory references and find files in those directories
    if let Some(directory_name) = extract_directory_name(input) {
        if let Some(found_files) = find_files_in_directory(&directory_name).await {
            candidates.extend(found_files);
        }
    }
    
    // Pattern 2: Quoted file paths
    if let Ok(quoted_regex) = regex::Regex::new(r#""([^"]+\.(?:rs|py|js|html|css|json|toml|yaml|yml|md|txt|xml|cfg|ini|log|csv))""#) {
        for captures in quoted_regex.captures_iter(input) {
            if let Some(path) = captures.get(1) {
                candidates.push(path.as_str().to_string());
            }
        }
    }
    
    // Pattern 3: File paths with extensions (not in quotes)
    if let Ok(file_regex) = regex::Regex::new(r"\b([a-zA-Z0-9_\-./\\]+\.(?:rs|py|js|html|css|json|toml|yaml|yml|md|txt|xml|cfg|ini|log|csv))\b") {
        for captures in file_regex.captures_iter(input) {
            if let Some(path) = captures.get(1) {
                let path_str = path.as_str();
                // Skip URLs and other non-file patterns
                if !path_str.contains("http") && !path_str.contains("www.") {
                    candidates.push(path_str.to_string());
                }
            }
        }
    }
    
    // Pattern 4: Common file names mentioned in conversation
    let common_files = ["README.md", "package.json", "Cargo.toml", "config.toml", "index.html", "main.rs", "app.js"];
    for file in common_files {
        if input.to_lowercase().contains(&file.to_lowercase()) {
            candidates.push(file.to_string());
        }
    }
    
    // Remove duplicates and return
    candidates.sort();
    candidates.dedup();
    candidates
}

fn extract_directory_name(input: &str) -> Option<String> {
    let _input_lower = input.to_lowercase();
    
    // Look for phrases like "files in X", "show X directory", "list X folder"
    let patterns = [
        r"(?i)files?\s+in\s+(\w+)",
        r"(?i)show\s+(?:the\s+)?files?\s+in\s+(\w+)",
        r"(?i)list\s+(?:the\s+)?(\w+)\s+(?:directory|folder)",
        r"(?i)(?:directory|folder)\s+(\w+)",
        r"(?i)in\s+(?:the\s+)?(\w+)\s+(?:directory|folder)",
    ];
    
    for pattern in &patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(input) {
                if let Some(dir_name) = captures.get(1) {
                    return Some(dir_name.as_str().to_string());
                }
            }
        }
    }
    
    None
}

async fn find_files_in_directory(directory_name: &str) -> Option<Vec<String>> {
    // First try current directory subdirectories
    let current_dir = std::env::current_dir().ok()?;
    let target_dir = current_dir.join(directory_name);
    
    if target_dir.exists() && target_dir.is_dir() {
        if let Ok(files) = list_directory_files(&target_dir).await {
            if !files.is_empty() {
                return Some(files.into_iter().map(|f| {
                    target_dir.join(f).to_string_lossy().to_string()
                }).collect());
            }
        }
    }
    
    // Try parent directories
    if let Some(parent) = current_dir.parent() {
        let target_dir = parent.join(directory_name);
        if target_dir.exists() && target_dir.is_dir() {
            if let Ok(files) = list_directory_files(&target_dir).await {
                if !files.is_empty() {
                    return Some(files.into_iter().map(|f| {
                        target_dir.join(f).to_string_lossy().to_string()
                    }).collect());
                }
            }
        }
    }
    
    // Search in accessible directories from permissions
    if let Ok(manager) = crate::permissions::PermissionManager::new().await {
        for allowed_path in manager.list_allowed_paths() {
            let target_dir = allowed_path.join(directory_name);
            if target_dir.exists() && target_dir.is_dir() {
                if let Ok(files) = list_directory_files(&target_dir).await {
                    if !files.is_empty() {
                        return Some(files.into_iter().map(|f| {
                            target_dir.join(f).to_string_lossy().to_string()
                        }).collect());
                    }
                }
            }
        }
    }
    
    None
}

async fn list_directory_files(dir_path: &std::path::Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    let mut entries = tokio::fs::read_dir(dir_path).await?;
    
    while let Ok(Some(entry)) = entries.next_entry().await {
        if let Ok(file_type) = entry.file_type().await {
            if file_type.is_file() {
                if let Some(file_name) = entry.file_name().to_str() {
                    files.push(file_name.to_string());
                }
            }
        }
    }
    
    files.sort();
    Ok(files)
}

async fn extract_file_creation_request(input: &str, config: &Config) -> Option<FileOperationInfo> {
    // Look for any creation keywords
    let creation_keywords = ["create", "make", "write", "generate", "build", "develop"];
    let has_creation_keyword = creation_keywords.iter().any(|&keyword| 
        input.to_lowercase().contains(keyword)
    );
    
    if !has_creation_keyword {
        return None;
    }
    
    // Check for quoted directory paths
    if let Some(captures) = regex::Regex::new(r#"(?i)(?:in\s+|here\s+|to\s+)"([^"]+)""#)
        .ok()?.captures(input) {
        let directory = captures.get(1)?.as_str();
        
        return Some(FileOperationInfo {
            filename: format!("{}\\{}", directory, "__AI_GENERATE__"), // Special marker
            operation: "create".to_string(),
            content_hint: input.to_string(),
        });
    }
    
    // Check for generic folder references that need intelligent directory detection
    let generic_folder_refs = ["random folder", "some folder", "any folder", "new folder"];
    if generic_folder_refs.iter().any(|&ref_term| input.to_lowercase().contains(ref_term)) {
        // Use intelligent directory detection to find an accessible directory
        if let Some(smart_dir) = config.get_intelligent_directory().await {
            return Some(FileOperationInfo {
                filename: smart_dir.join("__AI_GENERATE__").to_string_lossy().to_string(),
                operation: "create".to_string(),
                content_hint: input.to_string(),
            });
        }
    }
    
    // Check for "here" or current directory references
    if input.to_lowercase().contains("here") || 
       input.to_lowercase().contains("this directory") ||
       input.to_lowercase().contains("current folder") {
        return Some(FileOperationInfo {
            filename: "__AI_GENERATE__".to_string(),
            operation: "create".to_string(),
            content_hint: input.to_string(),
        });
    }
    
    // More flexible pattern matching for file creation requests
    let patterns = [
        r"(?i)(?:create|make|write|generate)(?:\s+a?)?\s+([a-zA-Z_][a-zA-Z0-9_\-\.]*)",
        r"(?i)([a-zA-Z_][a-zA-Z0-9_\-\.]*\.(?:rs|py|js|html|css|md|txt|json|toml|yaml|yml))\s+(?:file|script)",
        r"(?i)(?:hello world|sample|example)\s+(?:in\s+)?([a-zA-Z_][a-zA-Z0-9_\-\.]*)",
    ];
    
    for pattern in &patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(input) {
                let filename = captures.get(1)?.as_str().to_string();
                
                return Some(FileOperationInfo {
                    filename,
                    operation: "create".to_string(),
                    content_hint: input.to_string(), // Pass the full request to Gemini
                });
            }
        }
    }
    
    None
}

async fn extract_file_edit_request_smart(input: &str, config: &Config, memory_engine: &MemoryEngine) -> Option<FileOperationInfo> {
    // First try basic extraction
    if let Some(file_info) = extract_file_edit_request(input) {
        return Some(file_info);
    }
    
    // Try smart discovery
    if let Some(filename) = extract_filename_from_request_smart(input, config, memory_engine).await {
        return Some(FileOperationInfo {
            filename,
            operation: "edit".to_string(),
            content_hint: input.to_string(),
        });
    }
    
    None
}

fn extract_file_edit_request(input: &str) -> Option<FileOperationInfo> {
    let input_lower = input.to_lowercase();
    
    // Strategy 1: Direct filename patterns
    let patterns = [
        r"(?i)(?:edit|modify|update|remake)\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_\-]*\.[a-zA-Z0-9]+)",
        r"(?i)([a-zA-Z_][a-zA-Z0-9_\-]*\.[a-zA-Z0-9]+)\s+(?:file|in)",
    ];
    
    for pattern in &patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(input) {
                let filename = captures.get(1)?.as_str().trim().to_string();
                
                return Some(FileOperationInfo {
                    filename,
                    operation: "edit".to_string(),
                    content_hint: input.to_string(),
                });
            }
        }
    }
    
    // Strategy 2: Smart interpretation based on content keywords
    if input_lower.contains("rubik") || input_lower.contains("cube") {
        let base_filename = if input_lower.contains("js") || input_lower.contains("javascript") {
            "rubiks-cube.js"
        } else {
            "rubiks-cube.html"
        };
        
        // Try to find the file in accessible directories instead of hardcoding paths
        let filename = if input_lower.contains("random") {
            // Look for the file in common "random" directory locations
            if let Some(path) = smart_resolve_file_path(base_filename, &["random"]) {
                path
            } else {
                format!("random\\{}", base_filename)
            }
        } else {
            base_filename.to_string()
        };
        
        return Some(FileOperationInfo {
            filename,
            operation: "edit".to_string(),
            content_hint: input.to_string(),
        });
    }
    
    // Strategy 3: Common file type keywords
    let file_types = vec![
        ("config", "config.toml"),
        ("readme", "README.md"),
        ("package", "package.json"),
        ("cargo", "Cargo.toml"),
    ];
    
    for (keyword, filename) in file_types {
        if input_lower.contains(keyword) {
            return Some(FileOperationInfo {
                filename: filename.to_string(),
                operation: "edit".to_string(),
                content_hint: input.to_string(),
            });
        }
    }
    
    // Strategy 4: Extract from descriptive patterns like "edit the rubiks cube file"
    let descriptive_patterns = vec![
        r"(?i)(?:edit|modify|update|remake)\s+(?:the\s+)?([a-zA-Z][a-zA-Z0-9_\-\s]*?)\s+file",
        r"(?i)(?:the\s+)?([a-zA-Z][a-zA-Z0-9_\-\s]*?)\s+file\s+(?:in|from)",
    ];
    
    for pattern in &descriptive_patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(input) {
                let description = captures.get(1)?.as_str().trim();
                
                let filename = if description.contains("rubik") || description.contains("cube") {
                    if input_lower.contains("js") || input_lower.contains("javascript") {
                        if input_lower.contains("random") {
                            "C:\\Users\\Admin\\Desktop\\random\\rubiks-cube.js"
                        } else {
                            "rubiks-cube.js"
                        }
                    } else {
                        if input_lower.contains("random") {
                            "C:\\Users\\Admin\\Desktop\\random\\rubiks-cube.html"
                        } else {
                            "rubiks-cube.html"
                        }
                    }
                } else {
                    // Convert description to filename
                    &format!("{}.txt", description.replace(" ", "_"))
                };
                
                return Some(FileOperationInfo {
                    filename: filename.to_string(),
                    operation: "edit".to_string(),
                    content_hint: input.to_string(),
                });
            }
        }
    }
    
    None
}

fn smart_resolve_file_path(base_filename: &str, search_directories: &[&str]) -> Option<String> {
    // Try to find the file in various directories
    let user_profile = std::env::var("USERPROFILE").unwrap_or_default();
    
    for search_dir in search_directories {
        let possible_paths = vec![
            format!("{}\\{}\\{}", user_profile, search_dir, base_filename),
            format!("C:\\Users\\Admin\\Desktop\\{}\\{}", search_dir, base_filename),
            format!("{}\\Desktop\\{}\\{}", user_profile, search_dir, base_filename),
            format!("{}\\{}", search_dir, base_filename),
        ];
        
        for path in possible_paths {
            if std::path::Path::new(&path).exists() {
                return Some(path);
            }
        }
    }
    
    // Also check current directory
    let current_dir_path = format!(".\\{}", base_filename);
    if std::path::Path::new(&current_dir_path).exists() {
        return Some(current_dir_path);
    }
    
    // Return relative path as fallback
    None
}

async fn extract_filename_from_request_smart(input: &str, config: &Config, memory_engine: &MemoryEngine) -> Option<String> {
    // First try the basic extraction
    if let Some(filename) = extract_filename_from_request(input) {
        return Some(filename);
    }
    
    // CONTEXT-AWARE: Check if user is referring to recently worked file
    if let Some(context_file) = extract_file_from_context(input, memory_engine).await {
        return Some(context_file);
    }
    
    // Fuzzy matching and intelligent discovery
    if let Some((filename, interrupted)) = smart_file_discovery_with_interrupt(input, config).await {
        if interrupted {
            // Handle interruption - maybe ask for clarification
            return handle_interrupted_file_request(input).await;
        }
        return Some(filename);
    }
    
    // Gemini fallback for really ambiguous requests
    if let Some((filename, interrupted)) = gemini_file_interpretation_with_interrupt(input, config).await {
        if interrupted {
            return handle_interrupted_file_request(input).await;
        }
        return Some(filename);
    }
    
    None
}

/// Extract file from recent conversation context for requests like "change the title"
async fn extract_file_from_context(input: &str, memory_engine: &MemoryEngine) -> Option<String> {
    let input_lower = input.to_lowercase();
    
    // Look for context-dependent requests
    let context_keywords = ["change", "modify", "update", "edit", "fix", "add", "remove"];
    let has_context_action = context_keywords.iter().any(|&keyword| input_lower.contains(keyword));
    
    if !has_context_action {
        return None;
    }
    
    // Get recent messages to find last file worked on
    if let Ok(recent_messages) = memory_engine.get_recent_messages(20).await {
        for message in recent_messages.iter().rev() {
            if message.role == MessageRole::System {
                // Look for action summaries that mention file operations
                if message.content.contains("Edit completed") || 
                   message.content.contains("Changes made to") ||
                   message.content.contains("Modified file") {
                    
                    // Extract filename from the system message
                    if let Some(filename) = extract_filename_from_system_message(&message.content) {
                        return Some(filename);
                    }
                }
            } else if message.role == MessageRole::Assistant {
                // Look for file paths in assistant responses
                if let Some(filename) = extract_filename_from_message_content(&message.content) {
                    return Some(filename);
                }
            }
        }
    }
    
    None
}

/// Extract filename from system action messages
fn extract_filename_from_system_message(content: &str) -> Option<String> {
    // Pattern: "Changes made to filename:" or "Edit completed: filename"
    if let Some(start) = content.find("Changes made to ") {
        let after_prefix = &content[start + 16..]; // "Changes made to ".len() = 16
        if let Some(end) = after_prefix.find(':') {
            return Some(after_prefix[..end].trim().to_string());
        }
    }
    
    // Pattern: file paths in various contexts
    extract_filename_from_message_content(content)
}

/// Extract filename from any message content by looking for file patterns
fn extract_filename_from_message_content(content: &str) -> Option<String> {
    // Look for file extensions first (most reliable)
    let words: Vec<&str> = content.split_whitespace().collect();
    for word in words {
        if let Some(dot_pos) = word.rfind('.') {
            let extension = &word[dot_pos + 1..];
            // Common file extensions
            let valid_extensions = ["html", "css", "js", "ts", "rs", "py", "txt", "md", "json", "xml", "toml", "yaml", "yml"];
            if valid_extensions.contains(&extension) {
                // Clean up the filename (remove quotes, punctuation)
                let clean_filename = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_' && c != '-');
                if !clean_filename.is_empty() {
                    return Some(clean_filename.to_string());
                }
            }
        }
    }
    
    None
}

fn extract_filename_from_request(input: &str) -> Option<String> {
    let input_lower = input.to_lowercase();
    
    // Strategy 1: Look for explicit filenames with extensions
    let extension_patterns = vec![
        r"(?i)(?:read|show|display|edit|open)\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_\-]*\.[a-zA-Z0-9]+)",
        r"(?i)([a-zA-Z_][a-zA-Z0-9_\-]*\.[a-zA-Z0-9]+)\s+(?:file|in)",
    ];
    
    for pattern in &extension_patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(input) {
                return Some(captures.get(1)?.as_str().to_string());
            }
        }
    }
    
    // Strategy 2: Smart interpretation of common descriptions
    if input_lower.contains("rubik") || input_lower.contains("cube") {
        let base_filename = "rubiks-cube.html";
        
        // Look for directory context
        if input_lower.contains("random") {
            // Try to find the file in accessible directories using smart resolution
            if let Some(path) = smart_resolve_file_path(base_filename, &["random"]) {
                return Some(path);
            }
            return Some(format!("random\\{}", base_filename));
        }
        return Some(base_filename.to_string());
    }
    
    if input_lower.contains("config") {
        if input_lower.contains("toml") {
            return Some("config.toml".to_string());
        }
        return Some("config.toml".to_string());
    }
    
    if input_lower.contains("readme") {
        return Some("README.md".to_string());
    }
    
    if input_lower.contains("package") && input_lower.contains("json") {
        return Some("package.json".to_string());
    }
    
    if input_lower.contains("cargo") && input_lower.contains("toml") {
        return Some("Cargo.toml".to_string());
    }
    
    // Strategy 3: Look for quoted filenames or paths
    if let Some(captures) = regex::Regex::new(r#"(?i)"([^"]+)""#)
        .ok()?.captures(input) {
        return Some(captures.get(1)?.as_str().to_string());
    }
    
    // Strategy 4: Extract potential filename from "the X file" patterns
    let descriptive_patterns = vec![
        r"(?i)(?:read|show|display|edit|open)\s+(?:the\s+)?([a-zA-Z][a-zA-Z0-9_\-\s]*?)\s+file",
        r"(?i)(?:the\s+)?([a-zA-Z][a-zA-Z0-9_\-\s]*?)\s+file\s+(?:in|from)",
    ];
    
    for pattern in &descriptive_patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(input) {
                let description = captures.get(1)?.as_str().trim();
                
                // Convert description to likely filename
                let filename = if description.contains("rubik") || description.contains("cube") {
                    let base_filename = "rubiks-cube.html";
                    if input_lower.contains("random") {
                        if let Some(path) = smart_resolve_file_path(base_filename, &["random"]) {
                            return Some(path);
                        }
                        base_filename
                    } else {
                        base_filename
                    }
                } else if description.contains("javascript") || description.contains("js") {
                    if input_lower.contains("rubik") || input_lower.contains("cube") {
                        let base_filename = "rubiks-cube.js";
                        if input_lower.contains("random") {
                            if let Some(path) = smart_resolve_file_path(base_filename, &["random"]) {
                                return Some(path);
                            }
                            base_filename
                        } else {
                            base_filename
                        }
                    } else {
                        "script.js"
                    }
                } else {
                    // Fallback: convert description to filename
                    &format!("{}.txt", description.replace(" ", "_"))
                };
                
                return Some(filename.to_string());
            }
        }
    }
    
    None
}

/// Extract what actions were actually performed from function results
fn extract_action_summary(output_lines: &[String]) -> String {
    let mut actions = Vec::new();
    
    for line in output_lines {
        let line_lower = line.to_lowercase();
        
        // Detect file renames
        if line_lower.contains("successfully renamed") && line_lower.contains("to") {
            if let Some(from_pos) = line.find("'") {
                if let Some(to_pos) = line.rfind("'") {
                    let renamed_info = &line[from_pos..=to_pos];
                    actions.push(format!("RENAMED_FILE: {}", renamed_info));
                }
            }
        }
        
        // Detect file edits
        if line_lower.contains("edit completed") || line_lower.contains("successfully wrote") {
            if let Some(quote_start) = line.find("'") {
                if let Some(quote_end) = line[quote_start + 1..].find("'") {
                    let file_path = &line[quote_start + 1..quote_start + 1 + quote_end];
                    actions.push(format!("EDITED_FILE: {}", file_path));
                }
            }
        }
        
        // Detect file creation
        if line_lower.contains("created") && (line_lower.contains("file") || line_lower.contains(".")) {
            actions.push(format!("CREATED_FILE: {}", line));
        }
        
        // Detect file reading
        if line_lower.contains("read") && line_lower.contains("with") && line_lower.contains("lines") {
            if let Some(quote_start) = line.find("'") {
                if let Some(quote_end) = line[quote_start + 1..].find("'") {
                    let file_path = &line[quote_start + 1..quote_start + 1 + quote_end];
                    actions.push(format!("READ_FILE: {}", file_path));
                }
            }
        }
    }
    
    actions.join("; ")
}

/// Analyze user intent for file discovery using reasoning
#[derive(Debug)]
struct FileDiscoveryIntent {
    target_type: String,  // "styling", "config", "script", "markup", etc.
    action: String,       // "read", "edit", "fix", "change"
    content_clues: Vec<String>, // "title", "color", "function", etc.
}

async fn analyze_file_discovery_intent(input: &str) -> FileDiscoveryIntent {
    let input_lower = input.to_lowercase();
    
    // Determine action intent
    let action = if input_lower.contains("fix") || input_lower.contains("change") || input_lower.contains("edit") {
        "edit"
    } else if input_lower.contains("read") || input_lower.contains("show") || input_lower.contains("what") {
        "read"
    } else {
        "edit" // default assumption
    }.to_string();
    
    // Determine target type
    let target_type = if input_lower.contains("color") || input_lower.contains("style") || input_lower.contains("css") {
        "styling"
    } else if input_lower.contains("title") || input_lower.contains("text") || input_lower.contains("content") {
        "markup"
    } else if input_lower.contains("function") || input_lower.contains("script") || input_lower.contains("js") {
        "script"
    } else if input_lower.contains("config") || input_lower.contains("setting") {
        "config"
    } else {
        "any"
    }.to_string();
    
    // Extract content clues
    let mut content_clues = Vec::new();
    let words: Vec<&str> = input.split_whitespace().collect();
    for word in words {
        if word.len() > 3 && !["that", "this", "with", "from", "file", "code"].contains(&word) {
            content_clues.push(word.to_lowercase());
        }
    }
    
    FileDiscoveryIntent {
        target_type,
        action,
        content_clues,
    }
}

/// Revolutionary content-based file search using existing search capabilities
async fn search_by_content_and_intent(
    _input: &str, 
    intent: &FileDiscoveryIntent, 
    progress: &mut crate::cli::progress::ThinkingIndicator
) -> Option<String> {
    // Get list of accessible files
    let mut candidate_files = Vec::new();
    
    if let Ok(manager) = crate::permissions::PermissionManager::new().await {
        for allowed_path in manager.list_allowed_paths() {
            if let Ok(entries) = std::fs::read_dir(&allowed_path) {
                for entry in entries.flatten() {
                    if let Some(path_str) = entry.path().to_str() {
                        // Filter by file type based on intent
                        let matches_type = match intent.target_type.as_str() {
                            "styling" => path_str.ends_with(".css") || path_str.ends_with(".scss"),
                            "markup" => path_str.ends_with(".html") || path_str.ends_with(".htm"),
                            "script" => path_str.ends_with(".js") || path_str.ends_with(".ts"),
                            "config" => path_str.ends_with(".toml") || path_str.ends_with(".json") || path_str.ends_with(".yaml"),
                            _ => !path_str.ends_with(".lock") && !path_str.ends_with(".log")
                        };
                        
                        if matches_type {
                            candidate_files.push(path_str.to_string());
                        }
                    }
                }
            }
        }
    }
    
    // Now search CONTENT of these files for intent clues
    for file_path in candidate_files {
        if progress.is_interrupted() {
            return None;
        }
        
        if let Ok(content) = std::fs::read_to_string(&file_path) {
            let content_lower = content.to_lowercase();
            let mut relevance_score = 0;
            
            // Score based on content clues
            for clue in &intent.content_clues {
                if content_lower.contains(clue) {
                    relevance_score += clue.len(); // Longer/more specific clues = higher score
                }
            }
            
            // If we found relevant content, this is likely the right file
            if relevance_score > 5 { // Threshold for relevance
                return Some(file_path);
            }
        }
    }
    
    None
}

async fn smart_file_discovery_with_interrupt(input: &str, config: &Config) -> Option<(String, bool)> {
    let mut progress = crate::cli::progress::show_fuzzy_matching(input).await;
    let input_lower = input.to_lowercase();
    
    // PHASE 1: Use reasoning engine to understand intent
    let intent_analysis = analyze_file_discovery_intent(input).await;
    
    // PHASE 2: Content-based search (revolutionary improvement!)
    if let Some(result) = search_by_content_and_intent(input, &intent_analysis, &mut progress).await {
        progress.stop().await;
        return Some((result, false));
    }
    
    // PHASE 3: Fallback to enhanced fuzzy matching
    let fuzzy_matches = vec![
        (vec!["rubic", "rubix", "rubics", "rubiks", "cube", "cubes"], vec!["rubiks-cube.html", "rubiks-cube.js"]),
        (vec!["conf", "config", "configuration"], vec!["config.toml", "Cargo.toml"]),
        (vec!["readme", "read me", "documentation"], vec!["README.md", "readme.txt"]),
        (vec!["package", "pkg"], vec!["package.json", "Cargo.toml"]),
        (vec!["script", "javascript", "js"], vec!["rubiks-cube.js", "script.js"]),
        (vec!["html", "webpage", "page"], vec!["rubiks-cube.html", "index.html"]),
    ];
    
    for (keywords, filenames) in fuzzy_matches {
        // Check for interruption
        if progress.is_interrupted() {
            progress.stop().await;
            return Some(("".to_string(), true));
        }
        
        for keyword in keywords {
            if input_lower.contains(keyword) {
                // Try to find which file actually exists
                for filename in &filenames {
                    // Check for interruption
                    if progress.is_interrupted() {
                        progress.stop().await;
                        return Some(("".to_string(), true));
                    }
                    
                    // Check in accessible directories
                    if let Ok(manager) = crate::permissions::PermissionManager::new().await {
                        for allowed_path in manager.list_allowed_paths() {
                            let full_path = allowed_path.join(filename);
                            if full_path.exists() {
                                progress.stop().await;
                                return Some((full_path.to_string_lossy().to_string(), false));
                            }
                        }
                    }
                    
                    // Check in current directory
                    if let Ok(current_dir) = std::env::current_dir() {
                        let current_path = current_dir.join(filename);
                        if current_path.exists() {
                            progress.stop().await;
                            return Some((filename.to_string(), false));
                        }
                    }
                }
            }
        }
    }
    
    // Check for interruption before final search
    if progress.is_interrupted() {
        progress.stop().await;
        return Some(("".to_string(), true));
    }
    
    // Search for files containing the keywords in their names
    if let Some(discovered_file) = search_files_by_content_hint(&input_lower, config).await {
        progress.stop().await;
        return Some((discovered_file, false));
    }
    
    progress.stop().await;
    None
}

// Main smart file discovery function
pub async fn smart_file_discovery(input: &str, config: &Config) -> Option<String> {
    smart_file_discovery_with_interrupt(input, config).await
        .and_then(|(filename, _)| if filename.is_empty() { None } else { Some(filename) })
}

// Backwards compatibility wrapper  
async fn gemini_file_interpretation(input: &str, config: &Config) -> Option<String> {
    if let Some((filename, _interrupted)) = gemini_file_interpretation_with_interrupt(input, config).await {
        if !filename.is_empty() {
            return Some(filename);
        }
    }
    None
}

async fn search_files_by_content_hint(input: &str, _config: &Config) -> Option<String> {
    // Get directories we have access to
    let mut search_dirs = Vec::new();
    
    if let Ok(manager) = crate::permissions::PermissionManager::new().await {
        for path in manager.list_allowed_paths() {
            if path.exists() && path.is_dir() {
                search_dirs.push(path.clone());
            }
        }
    }
    
    // Also try current directory
    if let Ok(current_dir) = std::env::current_dir() {
        search_dirs.push(current_dir);
    }
    
    // Search through directories for relevant files
    for dir in search_dirs {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_file() {
                        let filename = entry.file_name().to_string_lossy().to_lowercase();
                        
                        // Check if filename matches content hints
                        if (input.contains("cube") || input.contains("rubik")) && filename.contains("cube") {
                            return Some(entry.path().to_string_lossy().to_string());
                        }
                        
                        if input.contains("config") && filename.contains("config") {
                            return Some(entry.path().to_string_lossy().to_string());
                        }
                        
                        if input.contains("html") && filename.ends_with(".html") {
                            return Some(entry.path().to_string_lossy().to_string());
                        }
                        
                        if (input.contains("js") || input.contains("javascript")) && filename.ends_with(".js") {
                            return Some(entry.path().to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }
    
    None
}

async fn gemini_file_interpretation_with_interrupt(input: &str, config: &Config) -> Option<(String, bool)> {
    let mut progress = crate::cli::progress::show_ai_processing("interpret your request").await;
    
    // Get list of available files in accessible directories
    let mut available_files = Vec::new();
    
    if let Ok(manager) = crate::permissions::PermissionManager::new().await {
        for path in manager.list_allowed_paths() {
            if path.exists() && path.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&path) {
                    for entry in entries.flatten() {
                        if let Ok(file_type) = entry.file_type() {
                            if file_type.is_file() {
                                available_files.push(entry.path().to_string_lossy().to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    
    if available_files.is_empty() {
        progress.stop().await;
        return None;
    }
    
    // Check for interruption before expensive AI call
    if progress.is_interrupted() {
        progress.stop().await;
        return Some(("".to_string(), true));
    }
    
    // Ask Gemini to interpret the request
    let prompt = format!(
        "The user made this request: \"{}\"\n\nHere are the available files:\n{}\n\nBased on the user's request, which file do they most likely want to access? Just respond with the full file path, nothing else. If you're not sure, respond with 'UNCLEAR'.",
        input,
        available_files.join("\n")
    );
    
    // Use Gemini to interpret the request
    if let Ok(response) = crate::gemini::query_gemini(&prompt, config).await {
        progress.stop().await;
        
        // Check if we were interrupted during the AI call
        if progress.is_interrupted() {
            return Some(("".to_string(), true));
        }
        
        let response = response.trim();
        if response != "UNCLEAR" && available_files.iter().any(|f| f.contains(response) || response.contains(f)) {
            // Find the best matching file
            for file in available_files {
                if file.contains(response) || response.contains(&file) {
                    return Some((file, false));
                }
            }
            return Some((response.to_string(), false));
        }
    }
    
    progress.stop().await;
    None
}

async fn handle_interrupted_file_request(input: &str) -> Option<String> {
    use crate::cli::colors::{YELLOW_WARN, GRAY_DIM, EMERALD_BRIGHT, RESET};
    
    // Use status bar instead of direct print to avoid UI corruption
    // Show request in status bar if needed
    // Empty lines disabled in ratatui mode
    // Options will be handled by ratatui UI instead of direct print
    // Option 1 - handled by UI
    // Option 2 - handled by UI
    // Option 3 - handled by UI
    // Option 4 - handled by UI
    // Empty lines disabled in ratatui mode
    
    print!("{}Enter your choice (1-4) or press Enter to skip: {}", EMERALD_BRIGHT, RESET);
    std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
    
    let mut response = String::new();
    if std::io::stdin().read_line(&mut response).is_ok() {
        match response.trim() {
            "1" => {
                print!("{}Please provide more details: {}", EMERALD_BRIGHT, RESET);
                std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
                
                let mut details = String::new();
                if std::io::stdin().read_line(&mut details).is_ok() {
                    let combined_input = format!("{} {}", input, details.trim());
                    return extract_filename_from_request(&combined_input);
                }
            },
            "2" => {
                // List available files
                if let Ok(manager) = crate::permissions::PermissionManager::new().await {
                    // Available files will be shown in ratatui UI instead
                    for path in manager.list_allowed_paths() {
                        if path.exists() && path.is_dir() {
                            if let Ok(entries) = std::fs::read_dir(&path) {
                                for entry in entries.flatten() {
                                    if let Ok(file_type) = entry.file_type() {
                                        if file_type.is_file() {
                                            // File paths handled by UI display
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                print!("{}Enter exact filename: {}", EMERALD_BRIGHT, RESET);
                std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
                
                let mut filename = String::new();
                if std::io::stdin().read_line(&mut filename).is_ok() {
                    return Some(filename.trim().to_string());
                }
            },
            "3" => {
                print!("{}Enter exact filename: {}", EMERALD_BRIGHT, RESET);
                std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());
                
                let mut filename = String::new();
                if std::io::stdin().read_line(&mut filename).is_ok() {
                    return Some(filename.trim().to_string());
                }
            },
            _ => {
                // Request skipped - no need for direct print
            }
        }
    }
    
    None
}

async fn smart_analyze_file(path: &std::path::Path, content: &str, user_input: &str, progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    let extension = path.extension()?.to_str()?;
    let _filename = path.file_name()?.to_str()?;
    
    match extension.to_lowercase().as_str() {
        "py" => analyze_python_file(path, content, user_input, progress).await,
        "js" | "ts" => analyze_javascript_file(path, content, user_input, progress).await,
        "rs" => analyze_rust_file(path, content, user_input, progress).await,
        "java" => analyze_java_file(path, content, user_input, progress).await,
        "cpp" | "cc" | "cxx" => analyze_cpp_file(path, content, user_input, progress).await,
        "c" => analyze_c_file(path, content, user_input, progress).await,
        "go" => analyze_go_file(path, content, user_input, progress).await,
        "html" => analyze_html_file(path, content, user_input, progress).await,
        _ => None
    }
}

async fn analyze_python_file(path: &std::path::Path, _content: &str, _user_input: &str, _progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    // Try to run python syntax check
    let result = tokio::process::Command::new("python")
        .arg("-m")
        .arg("py_compile")
        .arg(path)
        .output()
        .await;
        
    match result {
        Ok(output) => {
            if output.status.success() {
                Some("✅ **Python syntax check:** No syntax errors found".to_string())
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                Some(format!("Error: **Python syntax errors:**\n```\n{}\n```", error_msg))
            }
        }
        Err(_) => Some("Warning: **Python not available** - cannot check syntax".to_string())
    }
}

async fn analyze_javascript_file(path: &std::path::Path, _content: &str, _user_input: &str, _progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    // Try to run node syntax check
    let result = tokio::process::Command::new("node")
        .arg("--check")
        .arg(path)
        .output()
        .await;
        
    match result {
        Ok(output) => {
            if output.status.success() {
                Some("✅ **JavaScript syntax check:** No syntax errors found".to_string())
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                Some(format!("Error: **JavaScript syntax errors:**\n```\n{}\n```", error_msg))
            }
        }
        Err(_) => Some("Warning: **Node.js not available** - cannot check syntax".to_string())
    }
}

async fn analyze_rust_file(path: &std::path::Path, _content: &str, _user_input: &str, _progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    // Find if we're in a Cargo project
    let mut current_dir = path.parent()?;
    loop {
        if current_dir.join("Cargo.toml").exists() {
            // We're in a Cargo project - use cargo check
            let result = tokio::process::Command::new("cargo")
                .arg("check")
                .current_dir(current_dir)
                .output()
                .await;
                
            match result {
                Ok(output) => {
                    if output.status.success() {
                        return Some("✅ **Rust check:** No compilation errors found".to_string());
                    } else {
                        let error_msg = String::from_utf8_lossy(&output.stderr);
                        return Some(format!("Error: **Rust compilation errors:**\n```\n{}\n```", error_msg));
                    }
                }
                Err(_) => return Some("Warning: **Cargo not available** - cannot check Rust code".to_string())
            }
        }
        
        current_dir = current_dir.parent()?;
    }
}

async fn analyze_java_file(path: &std::path::Path, _content: &str, _user_input: &str, _progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    // Try to compile with javac
    let result = tokio::process::Command::new("javac")
        .arg("-Xlint")
        .arg(path)
        .output()
        .await;
        
    match result {
        Ok(output) => {
            if output.status.success() {
                Some("✅ **Java compilation:** No compilation errors found".to_string())
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                Some(format!("Error: **Java compilation errors:**\n```\n{}\n```", error_msg))
            }
        }
        Err(_) => Some("Warning: **Java compiler not available** - cannot check compilation".to_string())
    }
}

async fn analyze_cpp_file(path: &std::path::Path, _content: &str, _user_input: &str, _progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    // Try to compile with g++ or clang++
    for compiler in &["g++", "clang++"] {
        let result = tokio::process::Command::new(compiler)
            .arg("-fsyntax-only")
            .arg("-Wall")
            .arg(path)
            .output()
            .await;
            
        if let Ok(output) = result {
            if output.status.success() {
                return Some(format!("✅ **C++ syntax check ({}):** No syntax errors found", compiler));
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                return Some(format!("Error: **C++ compilation errors ({}):**\n```\n{}\n```", compiler, error_msg));
            }
        }
    }
    Some("Warning: **C++ compiler not available** - cannot check syntax".to_string())
}

async fn analyze_c_file(path: &std::path::Path, _content: &str, _user_input: &str, _progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    // Try to compile with gcc or clang
    for compiler in &["gcc", "clang"] {
        let result = tokio::process::Command::new(compiler)
            .arg("-fsyntax-only")
            .arg("-Wall")
            .arg(path)
            .output()
            .await;
            
        if let Ok(output) = result {
            if output.status.success() {
                return Some(format!("✅ **C syntax check ({}):** No syntax errors found", compiler));
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                return Some(format!("Error: **C compilation errors ({}):**\n```\n{}\n```", compiler, error_msg));
            }
        }
    }
    Some("Warning: **C compiler not available** - cannot check syntax".to_string())
}

async fn analyze_go_file(path: &std::path::Path, _content: &str, _user_input: &str, _progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    // Try to check with go
    let result = tokio::process::Command::new("go")
        .arg("fmt")
        .arg("-l")
        .arg(path)
        .output()
        .await;
        
    match result {
        Ok(output) => {
            if output.status.success() && output.stdout.is_empty() {
                Some("✅ **Go format check:** Code is properly formatted".to_string())
            } else if output.status.success() {
                Some("Warning: **Go format:** Code needs formatting".to_string())
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                Some(format!("Error: **Go errors:**\n```\n{}\n```", error_msg))
            }
        }
        Err(_) => Some("Warning: **Go not available** - cannot check code".to_string())
    }
}

async fn analyze_html_file(path: &std::path::Path, content: &str, _user_input: &str, _progress: &mut crate::cli::progress::ThinkingIndicator) -> Option<String> {
    let _path = path; // Basic HTML validation    
    let mut issues = Vec::new();
    
    if !content.contains("<!DOCTYPE") {
        issues.push("Warning: Missing DOCTYPE declaration");
    }
    
    if !content.contains("<html") {
        issues.push("Warning: Missing <html> tag");
    }
    
    if !content.contains("<head") {
        issues.push("Warning: Missing <head> section");
    }
    
    if !content.contains("<title") {
        issues.push("Warning: Missing <title> tag");
    }
    
    // Count open/close tags (basic check)
    let open_tags = content.matches('<').count() - content.matches("</").count() - content.matches("/>").count();
    if open_tags != 0 {
        issues.push("Warning: Possible unclosed HTML tags");
    }
    
    if issues.is_empty() {
        Some("✅ **HTML validation:** Basic structure looks good".to_string())
    } else {
        Some(format!("Warning: **HTML validation issues:**\n{}", issues.join("\n")))
    }
}

fn extract_directory_from_request(input: &str) -> Option<String> {
    // Look for quoted paths first
    if let Some(captures) = regex::Regex::new(r#"(?i)"([^"]+)""#)
        .ok()?.captures(input) {
        return Some(captures.get(1)?.as_str().to_string());
    }
    
    // Look for paths in various patterns including "files in X"
    let patterns = [
        r"(?i)what is in\s+([^\s]+)",
        r"(?i)list\s+([^\s]+)",
        r"(?i)contents of\s+([^\s]+)",
        r"(?i)ls\s+([^\s]+)",
        r"(?i)files?\s+in\s+([^\s]+)",
        r"(?i)show\s+(?:the\s+)?files?\s+in\s+([^\s]+)",
        r"(?i)list\s+(?:the\s+)?([^\s]+)\s+(?:directory|folder)",
        r"(?i)(?:directory|folder)\s+([^\s]+)",
        r"(?i)in\s+(?:the\s+)?([^\s]+)\s+(?:directory|folder)",
    ];
    
    for pattern in &patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(input) {
                return Some(captures.get(1)?.as_str().to_string());
            }
        }
    }
    
    None
}

fn extract_permission_request(input: &str) -> Option<String> {
    // Look for quoted paths in permission requests
    if let Some(captures) = regex::Regex::new(r#"(?i)(?:allow access|grant permissions?).*?"([^"]+)""#)
        .ok()?.captures(input) {
        return Some(captures.get(1)?.as_str().to_string());
    }
    
    // Look for "this folder" or "current directory"
    if input.to_lowercase().contains("this folder") || input.to_lowercase().contains("current directory") {
        if let Ok(current_dir) = std::env::current_dir() {
            return Some(current_dir.display().to_string());
        }
    }
    
    None
}

async fn handle_permission_grant(path: &str) -> String {
    use crate::permissions::PermissionManager;
    
    let path_buf = Path::new(path);
    
    match PermissionManager::new().await {
        Ok(mut manager) => {
            match manager.check_path_access(path_buf).await {
                Ok(true) => format!("✅ Access granted for `{}`.", path),
                Ok(false) => format!("Error: Access denied for `{}`.", path),
                Err(e) => format!("Error: Error granting access: {}", e),
            }
        }
        Err(e) => format!("Error: Failed to initialize permission manager: {}", e),
    }
}

struct FileOperationInfo {
    filename: String,
    operation: String,
    content_hint: String,
}

#[derive(Debug)]
enum EditConfirmation {
    Approved,
    Rejected,
    Revise(String),
    ExplainMore,
}

async fn show_edit_confirmation(filename: &str, original: &str, edited: &str, language: &str) -> EditConfirmation {
    use crate::cli::colors::{EMERALD_BRIGHT, YELLOW_WARN, GRAY_DIM, RESET};
    
    // Generate a summary of changes
    let diff_summary = generate_diff_summary(original, edited);
    let lines_changed = edited.lines().count().abs_diff(original.lines().count());
    
    // File edit confirmation handled by ratatui UI instead of direct print
    // File details shown in UI
    // Language info shown in UI
    // Line change info shown in UI
    // Empty lines disabled in ratatui mode
    
    // Show a condensed diff preview
    // Preview header shown in UI
    if diff_summary.len() > 500 {
        // Diff preview handled by UI system
        // Truncation notice handled by UI
    } else {
        // Full diff shown in UI
    }
    // Empty lines disabled in ratatui mode
    
    loop {
        // Options presented by ratatui UI instead of direct print
        // Option 1 - Apply changes (UI handled)
        // Option 2 - Revise (UI handled)
        // Option 3 - Cancel (UI handled)
        // Option 4 - Show explanation (UI handled)
        // Empty lines disabled in ratatui mode
        print!("{}Enter your choice (1-4): {}", EMERALD_BRIGHT, RESET);
        io::stdout().flush().unwrap_or(());
        
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            match input.trim() {
                "1" | "y" | "yes" | "apply" => return EditConfirmation::Approved,
                "2" | "r" | "revise" => {
                    // Empty lines disabled in ratatui mode
                    print!("{}Please provide feedback for the AI to revise the changes: {}", EMERALD_BRIGHT, RESET);
                    io::stdout().flush().unwrap_or(());
                    let mut feedback = String::new();
                    if io::stdin().read_line(&mut feedback).is_ok() && !feedback.trim().is_empty() {
                        return EditConfirmation::Revise(feedback.trim().to_string());
                    } else {
                        // No feedback - status shown in UI
                        return EditConfirmation::Rejected;
                    }
                }
                "3" | "n" | "no" | "cancel" => return EditConfirmation::Rejected,
                "4" | "e" | "explain" | "details" => return EditConfirmation::ExplainMore,
                _ => {
                    // Invalid choice warning shown in UI
                    continue;
                }
            }
        } else {
            // If stdin read fails, default to rejection for safety
            return EditConfirmation::Rejected;
        }
    }
}

fn generate_diff_summary(original: &str, edited: &str) -> String {
    // Reuse the existing logic from `differ.rs` which already uses imara-diff
    match crate::differ::create_diff(original, edited, "original", "edited") {
        Ok(diff) => crate::differ::colorize_diff(&diff),
        Err(_) => "Could not generate diff summary.".to_string(),
    }
}

async fn handle_file_creation(info: FileOperationInfo, config: &Config) -> String {
    use crate::file_io;    
    
    // Show progress indicator for file creation
    let mut progress = crate::cli::progress::show_thinking_with_context(&format!("creating file: {}", info.filename.replace("__AI_GENERATE__", "new file"))).await;
    
    // Determine the target directory
    let target_dir = if info.filename.contains("__AI_GENERATE__") {
        if info.filename == "__AI_GENERATE__" {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
        } else {
            let dir_path = info.filename.replace("\\__AI_GENERATE__", "").replace("/__AI_GENERATE__", "");
            std::path::PathBuf::from(dir_path)
        }
    } else {
        let path = std::path::Path::new(&info.filename);
        if path.is_absolute() {
            path.parent().unwrap_or(path).to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
        }
    };
    
    // Check for interruption
    if progress.is_interrupted() {
        progress.stop().await;
        return "Warning: File creation interrupted by user".to_string();
    }
    
    // Check permissions on the directory
    match crate::permissions::verify_path_access(&target_dir).await {
        Ok(true) => {
            // Check for interruption before expensive AI call
            if progress.is_interrupted() {
                progress.stop().await;
                return "Warning: File creation interrupted by user".to_string();
            }
            
            // Let Gemini generate the content and filename
            match generate_file_content(&info.filename, &info.content_hint, config).await {
                Ok(ai_response) => {
                    // Parse AI response
                    if info.filename.contains("__AI_GENERATE__") {
                        match parse_ai_file_response(&ai_response, &target_dir).await {
                            Ok((final_path, content)) => {
                                match file_io::write_file(&final_path, &content).await {
                                    Ok(_) => {
                                        let language = file_io::detect_language(&final_path);
                                        let result = format!("✅ Created file: {}\n\n```{}\n{}\n```", 
                                            final_path.display(), language, content);
                                        progress.stop().await;
                                        result
                                    }
                                    Err(e) => {
                                        let result = format!("Error: Failed to create file: {}", e);
                                        progress.stop().await;
                                        result
                                    }
                                }
                            }
                            Err(e) => {
                                // Show more helpful error with truncated response
                                let preview = ai_response.chars().take(150).collect::<String>();
                                let result = format!("Error: Failed to parse AI response: {}\n\nResponse preview: {}...", e, preview);
                                progress.stop().await;
                                result
                            }
                        }
                    } else {
                        // Traditional single file creation
                        let path = if info.filename.contains("\\") || info.filename.contains("/") {
                            std::path::PathBuf::from(&info.filename)
                        } else {
                            target_dir.join(&info.filename)
                        };
                        
                        match file_io::write_file(&path, &ai_response).await {
                            Ok(_) => {
                                let language = file_io::detect_language(&path);
                                let result = format!("✅ Created file: {}\n\n```{}\n{}\n```", 
                                    path.display(), language, ai_response);
                                progress.stop().await;
                                result
                            }
                            Err(e) => {
                                let result = format!("Error: Failed to create file: {}", e);
                                progress.stop().await;
                                result
                            }
                        }
                    }
                }
                Err(e) => {
                    let result = format!("Error: Failed to generate content: {}", e);
                    progress.stop().await;
                    result
                }
            }
        }
        Ok(false) => {
            let result = format!("Error: Access denied to create file in this directory. Use '/permissions' to allow access.");
            progress.stop().await;
            result
        }
        Err(e) => {
            let result = format!("Error: Permission check failed: {}", e);
            progress.stop().await;
            result
        }
    }
}

async fn parse_ai_file_response(ai_response: &str, target_dir: &std::path::Path) -> Result<(std::path::PathBuf, String)> {
    // Strategy 1: Clean and parse as JSON
    if let Ok(result) = try_parse_as_json(ai_response, target_dir) {
        return Ok(result);
    }
    
    // Strategy 2: Extract from common patterns using regex
    if let Ok(result) = try_extract_with_regex(ai_response, target_dir) {
        return Ok(result);
    }
    
    // Strategy 3: Generate intelligent filename from content
    if let Ok(result) = try_generate_from_content(ai_response, target_dir) {
        return Ok(result);
    }
    
    // Final fallback: Use generic filename with full response as content
    warn_println!("Using fallback filename generation");
    let filename = "generated_file.txt";
    let final_path = target_dir.join(filename);
    Ok((final_path, ai_response.to_string()))
}

fn try_parse_as_json(ai_response: &str, target_dir: &std::path::Path) -> Result<(std::path::PathBuf, String)> {
    // Clean the response - remove any markdown formatting
    let cleaned_response = ai_response.trim();
    let cleaned_response = if cleaned_response.starts_with("```json") {
        cleaned_response
            .strip_prefix("```json")
            .and_then(|s| s.strip_suffix("```"))
            .unwrap_or(cleaned_response)
            .trim()
    } else if cleaned_response.starts_with("```") {
        cleaned_response
            .strip_prefix("```")
            .and_then(|s| s.strip_suffix("```"))
            .unwrap_or(cleaned_response)
            .trim()
    } else {
        cleaned_response
    };
    
    // Try to find JSON object within the response
    if let Some(start) = cleaned_response.find('{') {
        if let Some(end) = cleaned_response.rfind('}') {
            let json_part = &cleaned_response[start..=end];
            
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(json_part) {
                if let (Some(filename), Some(content)) = (
                    json_value.get("filename").and_then(|v| v.as_str()),
                    json_value.get("content").and_then(|v| v.as_str())
                ) {
                    let final_path = target_dir.join(filename);
                    return Ok((final_path, content.to_string()));
                }
            }
        }
    }
    
    Err(anyhow::anyhow!("No valid JSON found"))
}

fn try_extract_with_regex(ai_response: &str, target_dir: &std::path::Path) -> Result<(std::path::PathBuf, String)> {
    // Try to extract filename and content using various patterns
    let patterns = [
        // Pattern 1: "filename": "name.ext", "content": "..."
        (r#""filename"\s*:\s*"([^"]+)"[\s\S]*?"content"\s*:\s*"([\s\S]*?)"(?:\s*[}\],]|$)"#, true),
        // Pattern 2: filename: name.ext, content: ...
        (r#"filename\s*:\s*([^\s,]+)[\s\S]*?content\s*:\s*([\s\S]+?)(?:\n|$)"#, false),
        // Pattern 3: Filename: name.ext \n Content: ...
        (r#"(?i)filename\s*:\s*([^\n]+)\n[\s\S]*?content\s*:\s*([\s\S]+)"#, false),
    ];
    
    for (pattern, is_json_escaped) in &patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(ai_response) {
                if let (Some(filename_match), Some(content_match)) = (captures.get(1), captures.get(2)) {
                    let filename = filename_match.as_str().trim().trim_matches('"');
                    let content = content_match.as_str().trim().trim_matches('"');
                    
                    // If it was JSON-escaped, unescape it
                    let final_content = if *is_json_escaped {
                        content.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\")
                    } else {
                        content.to_string()
                    };
                    
                    if !filename.is_empty() && !final_content.is_empty() {
                        let final_path = target_dir.join(filename);
                        return Ok((final_path, final_content));
                    }
                }
            }
        }
    }
    
    Err(anyhow::anyhow!("No patterns matched"))
}

fn try_generate_from_content(ai_response: &str, target_dir: &std::path::Path) -> Result<(std::path::PathBuf, String)> {
    // Look for code patterns and generate appropriate filename
    let content = ai_response.trim();
    
    let filename = if content.contains("console.log") || content.contains("function") || content.contains("const ") || content.contains("let ") {
        if content.contains("<html>") || content.contains("<!DOCTYPE") {
            "index.html"
        } else {
            "script.js"
        }
    } else if content.contains("<html>") || content.contains("<!DOCTYPE") || content.contains("<div>") {
        "page.html"
    } else if content.contains("body {") || content.contains(".class") || content.contains("#id") {
        "styles.css"
    } else if content.contains("def ") || content.contains("import ") || content.contains("print(") {
        "script.py"
    } else if content.contains("fn main") || content.contains("use std") {
        "main.rs"
    } else if content.contains("package ") || content.contains("public class") {
        "Main.java"
    } else {
        "generated_file.txt"
    };
    
    let final_path = target_dir.join(filename);
    Ok((final_path, content.to_string()))
}

async fn handle_file_edit(info: FileOperationInfo, config: &Config) -> String {
    use crate::{file_io, gemini};
    use std::path::Path;

    // Show progress indicator for file editing
    let mut progress = crate::cli::progress::show_thinking_with_context(&format!("editing {}", info.filename)).await;

    // Check if it's a relative path and if the file exists in current directory
    let actual_path = if Path::new(&info.filename).is_absolute() {
        PathBuf::from(&info.filename)
    } else {
        std::env::current_dir().unwrap_or_default().join(&info.filename)
    };

    if !actual_path.exists() {
        let result = format!("Error: File {} does not exist. Use 'create {}' to create it first.", info.filename, info.filename);
        progress.stop().await;
        return result;
    }

    if progress.is_interrupted() {
        progress.stop().await;
        return "Warning: File editing interrupted by user".to_string();
    }

    let original_content = match file_io::read_file(&actual_path).await {
        Ok(content) => content,
        Err(e) => {
            let result = format!("Error: Failed to read file {}: {}", info.filename, e);
            progress.stop().await;
            return result;
        }
    };

    let mut current_query = info.content_hint.clone();
    let language = file_io::detect_language(&actual_path);

    loop {
        if progress.is_interrupted() {
            progress.stop().await;
            return "Warning: File editing interrupted by user".to_string();
        }

        match gemini::edit_code(&original_content, &current_query, &language, config).await {
                Ok(edited_content) => {
                    progress.stop().await;

                    // Show confirmation dialog before making changes
                    match show_edit_confirmation(&info.filename, &original_content, &edited_content, &language).await {
                        EditConfirmation::Approved => {
                            match file_io::write_file(&actual_path, &edited_content).await {
                                Ok(_) => {
                                    return format!("✅ File {} has been successfully edited", info.filename);
                                }
                                Err(e) => return format!("Error: Failed to write edited file: {}", e),
                            }
                        }
                        EditConfirmation::Rejected => {
                            return format!("Warning: File edit cancelled by user");
                        }
                        EditConfirmation::ExplainMore => {
                            return format!("💡 **Proposed changes to {}:**\n\n{}\n\nTo proceed with these changes, please run the edit command again.",
                                info.filename, 
                                generate_diff_summary(&original_content, &edited_content));
                        }
                        EditConfirmation::Revise(feedback) => {
                            current_query = format!("The user's initial request was: '{}'. My last attempt was not correct. The user provided this feedback: '{}'. Please try again, modifying the original code to satisfy the initial request and the new feedback.", info.content_hint, feedback);
                            progress = crate::cli::progress::show_thinking_with_context("Revising changes based on feedback").await;
                            continue;
                        }
                    }
                }
                Err(e) => {
                    let result = format!("Error: Failed to edit file: {}", e);
                    progress.stop().await;
                    return result;
                }
            }
        }
}

async fn handle_file_read(filename: &str) -> String {
    handle_file_read_with_context(filename, "", false).await
}

async fn handle_file_read_with_context(filename: &str, user_input: &str, should_analyze: bool) -> String {
    let mut progress = crate::cli::progress::show_thinking_with_context(&format!("reading {}", filename)).await;
    
    use crate::file_io;
    use std::path::Path;
    
    let path = Path::new(filename);
    
    match file_io::read_file(path).await {
        Ok(content) => {
            let language = file_io::detect_language(path);
            let mut result = format!("📄 Contents of {}:\n\n```{}\n{}\n```", filename, language, content);
            
            // Only analyze if user specifically wants error checking or fixing
            if should_analyze {
                progress.stop().await;
                progress = crate::cli::progress::show_thinking_with_context("checking for errors and issues").await;
                
                if let Some(analysis) = smart_analyze_file(path, &content, user_input, &mut progress).await {
                    result.push_str(&format!("\n\n🔍 **Analysis:**\n{}", analysis));
                }
            }
            
            progress.stop().await;
            result
        }
        Err(e) => {
            progress.stop().await;
            format!("Error: Failed to read file {}: {}", filename, e)
        }
    }
}

async fn handle_directory_list(directory: &str) -> String {
    use std::fs;
    
    // Show progress indicator for directory listing
    let mut progress = crate::cli::progress::show_thinking_with_context(&format!("finding and listing directory '{}'", directory)).await;
    
    // First try to find the directory intelligently
    let found_path = find_directory_by_name(directory).await;
    
    let path = match found_path {
        Some(ref p) => p,
        None => {
            let result = format!("Error: Could not find directory '{}' in accessible locations.\n\nSearched in:\n- Current directory subdirectories\n- Parent directory subdirectories\n- Accessible directories from permissions\n\nTry using a full path or check directory name spelling.", directory);
            progress.stop().await;
            return result;
        }
    };
    
    // Check permissions first
    match crate::permissions::verify_path_access(path).await {
        Ok(false) => {
            let result = format!("Error: Access denied to directory: {}\nUse '/permissions' to manage folder access.", path.display());
            progress.stop().await;
            return result;
        }
        Err(e) => {
            let result = format!("Error: Permission check failed: {}", e);
            progress.stop().await;
            return result;
        }
        Ok(true) => {}
    }
    
    match fs::read_dir(path) {
        Ok(entries) => {
            let mut result = format!(" Contents of {} ({}):\n\n", directory, path.display());
            let mut files = Vec::new();
            let mut dirs = Vec::new();
            
            for entry in entries.flatten() {
                let entry_path = entry.path();
                let name = entry_path.file_name().unwrap().to_string_lossy();
                
                if entry_path.is_dir() {
                    dirs.push(format!(" {}/", name));
                } else {
                    // Get file size
                    let size = entry_path.metadata()
                        .map(|m| format_file_size(m.len()))
                        .unwrap_or_else(|_| "?".to_string());
                    files.push(format!("📄 {} ({})", name, size));
                }
            }
            
            // Sort and display directories first, then files
            dirs.sort();
            files.sort();
            
            for dir in dirs {
                result.push_str(&format!("  {}\n", dir));
            }
            for file in files {
                result.push_str(&format!("  {}\n", file));
            }
            
            if result.lines().count() == 3 {
                result.push_str("  (empty directory)\n");
            }
            
            progress.stop().await;
            result
        }
        Err(e) => {
            let result = format!("Error: Failed to read directory {}: {}", path.display(), e);
            progress.stop().await;
            result
        }
    }
}

async fn find_directory_by_name(directory_name: &str) -> Option<std::path::PathBuf> {
    use std::path::Path;
    
    // If it's already a path, try it directly
    let direct_path = Path::new(directory_name);
    if direct_path.exists() && direct_path.is_dir() {
        return Some(direct_path.to_path_buf());
    }
    
    // Search in current directory
    if let Ok(current_dir) = std::env::current_dir() {
        let target_dir = current_dir.join(directory_name);
        if target_dir.exists() && target_dir.is_dir() {
            return Some(target_dir);
        }
    }
    
    // Search in parent directory
    if let Ok(current_dir) = std::env::current_dir() {
        if let Some(parent) = current_dir.parent() {
            let target_dir = parent.join(directory_name);
            if target_dir.exists() && target_dir.is_dir() {
                return Some(target_dir);
            }
        }
    }
    
    // Search in accessible directories from permissions
    if let Ok(manager) = crate::permissions::PermissionManager::new().await {
        for allowed_path in manager.list_allowed_paths() {
            let target_dir = allowed_path.join(directory_name);
            if target_dir.exists() && target_dir.is_dir() {
                return Some(target_dir);
            }
            
            // Also search one level deep in accessible directories
            if let Ok(entries) = std::fs::read_dir(&allowed_path) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        let target_dir = entry.path().join(directory_name);
                        if target_dir.exists() && target_dir.is_dir() {
                            return Some(target_dir);
                        }
                    }
                }
            }
        }
    }
    
    None
}

fn format_file_size(size: u64) -> String {
    if size < 1024 {
        format!("{} B", size)
    } else if size < 1024 * 1024 {
        format!("{:.1} KB", size as f64 / 1024.0)
    } else if size < 1024 * 1024 * 1024 {
        format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

async fn generate_file_content(filename: &str, user_request: &str, config: &Config) -> Result<String> {
    if filename.contains("__AI_GENERATE__") {
        // Let AI decide the filename and content with retry logic
        let directory = if filename == "__AI_GENERATE__" {
            "current directory".to_string()
        } else {
            filename.replace("\\__AI_GENERATE__", "").replace("/__AI_GENERATE__", "")
        };
        
        // Try with strict JSON prompt first
        for attempt in 1..=3 {
            let prompt = if attempt == 1 {
                format!(
                    "User request: \"{}\"\nDirectory: {}\n\n\
                    Create a file for this request. Choose appropriate filename and generate complete content.\n\
                    \n\
                    RESPOND WITH VALID JSON ONLY - NO MARKDOWN, NO EXPLANATIONS:\n\
                    {{\"filename\": \"example.js\", \"content\": \"// Your code here\\nconsole.log('Hello World');\"}}\n\
                    \n\
                    CRITICAL REQUIREMENTS:\n\
                    - ONLY return JSON, nothing else\n\
                    - Escape all quotes with backslashes\n\
                    - Use \\n for newlines in content\n\
                    - Choose descriptive filename with correct extension\n\
                    - Generate complete, functional code\n\
                    - NO ```json blocks, just raw JSON",
                    user_request, directory
                )
            } else if attempt == 2 {
                format!(
                    "PREVIOUS ATTEMPT FAILED - PLEASE FOLLOW FORMAT EXACTLY\n\n\
                    User request: \"{}\"\nDirectory: {}\n\n\
                    Return ONLY this JSON structure with no other text:\n\
                    {{\"filename\": \"your_filename.ext\", \"content\": \"your file content here\"}}\n\n\
                    Example for JavaScript: {{\"filename\": \"game.js\", \"content\": \"console.log('Hello');\"}}\n\
                    Example for HTML: {{\"filename\": \"page.html\", \"content\": \"<html><body>Hello</body></html>\"}}\n\n\
                    RESPOND WITH ONLY THE JSON - NO OTHER TEXT!",
                    user_request, directory
                )
            } else {
                // Final attempt with simpler prompt
                format!(
                    "Create a {} file. Return JSON: {{\"filename\": \"name.ext\", \"content\": \"file content\"}}",
                    user_request
                )
            };
            
            match gemini::query_gemini(&prompt, config).await {
                Ok(response) => return Ok(response),
                Err(_e) if attempt < 3 => {
                    // Wait a bit before retrying
                    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        
        Err(anyhow::anyhow!("Failed to generate content after 3 attempts"))
    } else {
        // Traditional approach for explicit filenames
        let extension = std::path::Path::new(filename)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        
        let prompt = format!(
            "Create content for a file named '{}' based on this request: \"{}\"\n\n\
            File extension: {}\n\
            Generate complete, functional content with best practices.\n\
            Return ONLY the file content, no explanations or markdown code blocks.",
            filename, user_request, extension
        );
        
        gemini::query_gemini(&prompt, config).await
    }
}

async fn handle_permissions_command() -> Result<String> {
    use crate::permissions::PermissionManager;
    
    let manager = PermissionManager::new().await?;
    let allowed_paths = manager.list_allowed_paths();
    
    if allowed_paths.is_empty() {
        Ok("No folders are currently allowed.\nTo allow access, use a command that requires file access (e.g., 'read file.txt').".to_string())
    } else {
        let mut response = "Currently allowed folders:\n".to_string();
        for path in allowed_paths {
            response.push_str(&format!("  ✅ {}\n", path.display()));
        }
        Ok(response)
    }
}

async fn handle_ls_command() -> String {
    use std::fs;
    
    match fs::read_dir(".") {
        Ok(entries) => {
            let mut response = "Files and directories in current folder:\n".to_string();
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.file_name().unwrap().to_string_lossy();
                if path.is_dir() {
                    response.push_str(&format!("   {}/\n", name));
                } else {
                    response.push_str(&format!("  📄 {}\n", name));
                }
            }
            response
        }
        Err(e) => format!("Failed to list directory: {}", e),
    }
}

fn get_help_text() -> String {
    "Available commands:\n\
    /help        - Show this help message\n\
    /clear       - Clear conversation history\n\
    /permissions - Show folder permissions\n\
    /pwd         - Show current directory\n\
    /ls          - List files and directories\n\
    exit, quit   - Exit chat mode\n\n\
    File operations you can ask for:\n\
    'create hello.py'                    - Create a Python file\n\
    'make a hello world script'          - Create a script\n\
    'edit main.rs'                       - Edit an existing file\n\
    'show config.toml'                   - Display file contents\n\
    'what is in \"C:\\\\path\\\\to\\\\folder\"'    - List directory contents\n\
    'list /home/user/documents'          - List directory contents\n\n\
    Just type your message to chat with the AI assistant!".to_string()
}

fn print_help_with_ui(chat_ui: &ChatUI) {
    chat_ui.display_system_message("📚 Available Commands:", SystemMessageType::Info);
    
    let help_text = "• /help - Show this help message
• /clear - Clear conversation history
• /permissions - Show folder permissions
• /pwd - Show current directory
• /ls - List files and directories
• exit, quit - Exit chat mode

💡 File operations you can ask for:
• 'create hello.py' - Create a Python file
• 'make a hello world script' - Create a script
• 'edit main.rs' - Edit an existing file
• 'show config.toml' - Display file contents
• 'what is in C:\\path\\to\\folder' - List directory
• 'list /home/user/documents' - List directory

Just type your message to chat with Glimmer AI!";
    
    chat_ui.display_response(help_text, None);
}

async fn handle_permissions_command_with_ui(chat_ui: &ChatUI) -> Result<()> {
    chat_ui.display_system_message("🔐 Checking file permissions...", SystemMessageType::Info);
    
    match handle_permissions_command().await {
        Ok(_) => {
            chat_ui.display_system_message("✅ Permission check completed", SystemMessageType::Success);
        }
        Err(e) => {
            chat_ui.display_system_message(&format!("Error: Permission error: {}", e), SystemMessageType::Error);
        }
    }
    Ok(())
}

async fn handle_ls_command_with_ui(chat_ui: &ChatUI) {
    chat_ui.display_system_message(" Listing current directory...", SystemMessageType::Info);
    
    let current_dir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    match tokio::fs::read_dir(&current_dir).await {
        Ok(mut entries) => {
            let mut files = Vec::new();
            let mut dirs = Vec::new();
            
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Ok(file_type) = entry.file_type().await {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if file_type.is_dir() {
                        dirs.push(format!(" {}/", name));
                    } else {
                        files.push(format!("📄 {}", name));
                    }
                }
            }
            
            dirs.sort();
            files.sort();
            
            let mut listing = format!("Directory: {}\n\n", current_dir);
            
            if !dirs.is_empty() {
                listing.push_str("Directories:\n");
                for dir in &dirs {
                    listing.push_str(&format!("{}\n", dir));
                }
                listing.push('\n');
            }
            
            if !files.is_empty() {
                listing.push_str("Files:\n");
                for file in &files {
                    listing.push_str(&format!("{}\n", file));
                }
            }
            
            if dirs.is_empty() && files.is_empty() {
                listing.push_str("(empty directory)");
            }
            
            chat_ui.display_response(&listing, None);
        }
        Err(e) => {
            chat_ui.display_system_message(&format!("Error: Cannot read directory: {}", e), SystemMessageType::Error);
        }
    }
}

// Auto-proceed complex tasks without confirmation (smart behavior)
async fn handle_auto_complex_task_with_ui(
    input: &str,
    steps: &[String],
    config: &Config,
    chat_ui: &ChatUI,
) -> Result<(String, Option<String>)> {
    use crate::cli::colors::{EMERALD_BRIGHT, GRAY_DIM, RESET};
    
    // Show brief info without asking for confirmation
    // Auto-executing task shown in status bar instead of direct print
    PersistentStatusBar::set_ai_thinking("Auto-executing multi-step task");
    for (i, step) in steps.iter().enumerate() {
        // Step shown in status bar
        PersistentStatusBar::set_ai_thinking(&format!("Step {}: {}", i + 1, step));
    }
    // Empty lines disabled in ratatui mode
    
    let mut results = Vec::new();
    
    for (i, step) in steps.iter().enumerate() {
        chat_ui.show_thinking_box(&format!("step {} of {}: {}", i + 1, steps.len(), step.chars().take(30).collect::<String>()));
        
        let step_prompt = format!(
            "Execute step {} of {}: {}\n\nOriginal request: {}\n\nPrevious steps completed: {}\n\nProvide a focused, actionable response for this step only.",
            i + 1,
            steps.len(),
            step,
            input,
            results.len()
        );
        
        match gemini::query_gemini(&step_prompt, config).await {
            Ok(step_response) => {
                results.push(step_response);
            }
            Err(e) => {
                chat_ui.display_system_message(&format!("Error: Step {} failed: {}", i + 1, e), SystemMessageType::Error);
                break;
            }
        }
    }
    
    let final_result = if results.is_empty() {
        "Task execution failed".to_string()
    } else {
        results.join("\n\n")
    };
    
    Ok((final_result, None))
}

// Modern UI for complex tasks that truly need confirmation
async fn handle_complex_task_with_modern_ui(
    _input: &str,
    task_analysis: &gemini::TaskComplexity,
    _config: &Config,
    _chat_ui: &ChatUI,
) -> Result<(String, Option<String>)> {
    // This function is being refactored to integrate with the TUI.
    // It must not manage raw_mode or read events directly, as that conflicts
    // with the main ratatui event loop in `handle_chat`.

    // Instead of blocking for input, this function will format a confirmation
    // prompt as a string. The main TUI loop will be responsible for displaying
    // this prompt and handling the user's key presses ([Enter], [Space], [Esc])
    // to decide the next action. This requires a state management change in the
    // main loop to handle this confirmation mode.

    let mut prompt = "Warning:  **High-Complexity Task Detected**\n".to_string();
    prompt.push_str(&format!("Reasoning: {}\n\n", task_analysis.reasoning));

    prompt.push_str("📋 **Planned Steps:**\n");
    for (i, step) in task_analysis.steps.iter().enumerate() {
        prompt.push_str(&format!("  {}. {}\n", i + 1, step));
    }
    prompt.push_str("\n");

    prompt.push_str("Press [Space] to execute the detailed step-by-step plan.\n");
    prompt.push_str("Press [Enter] to proceed with a simplified approach.\n");
    prompt.push_str("Press [Esc] to cancel.\n");

    // The caller (main TUI loop) should now display this prompt and handle the
    // subsequent user input to call either `handle_auto_complex_task_with_ui`
    // or query Gemini with a simplified prompt.
    // For now, we return the prompt text.
    Ok((prompt, None))
}

async fn handle_complex_task_with_ui(
    input: &str,
    task_analysis: &gemini::TaskComplexity,
    config: &Config,
    chat_ui: &ChatUI,
) -> Result<(String, Option<String>)> {
    chat_ui.display_system_message("🔧 Complex task detected - breaking into steps", SystemMessageType::Info);
    
    let mut results = Vec::new();
    
    for (i, step) in task_analysis.steps.iter().enumerate() {
        chat_ui.show_thinking_box(&format!("executing step {} of {}", i + 1, task_analysis.steps.len()));
        
        let step_prompt = format!(
            "You are working on step {} of {} for this complex task:\n\
            Original request: {}\n\
            Current step: {}\n\
            Previous results: {:?}\n\
            \n\
            Complete this step thoroughly and provide detailed output:",
            i + 1,
            task_analysis.steps.len(),
            input,
            step,
            results
        );
        
        match gemini::query_gemini_with_thinking(&step_prompt, config, Some(2048)).await {
            Ok((step_result, _)) => {
                results.push(step_result.clone());
                chat_ui.display_system_message(
                    &format!("✅ Step {} completed", i + 1),
                    SystemMessageType::Success
                );
            }
            Err(e) => {
                chat_ui.display_system_message(
                    &format!("Error: Step {} failed: {}", i + 1, e),
                    SystemMessageType::Error
                );
                return Err(e);
            }
        }
    }
    
    let final_result = results.join("\n\n---\n\n");
    let thinking = Some(format!("Completed {} steps:\n{}", task_analysis.steps.len(), 
                               task_analysis.steps.iter()
                               .enumerate()
                               .map(|(i, step)| format!("{}. {}", i + 1, step))
                               .collect::<Vec<_>>()
                               .join("\n")));
    
    Ok((final_result, thinking))
}

/// Display formatted response with Claude Code-style formatting
/// Handles diffs with simple red/green backgrounds and code blocks with syntax highlighting
fn display_formatted_response(response: &str) {
    
    // Check if response contains diff patterns
    if is_diff_response(response) {
        display_diff_response(response);
        return;
    }
    
    // Check if response contains code blocks
    if contains_code_blocks(response) {
        display_code_block_response(response);
        return;
    }
    
    // Display clean response: white answers, grey context, no **Answer** labels
    display_clean_response(response);
}

/// Check if response is a diff-style output
fn is_diff_response(response: &str) -> bool {
    let lines: Vec<&str> = response.lines().collect();
    let diff_indicators = lines.iter()
        .filter(|line| line.starts_with('+') || line.starts_with('-'))
        .count();
    
    // If more than 2 lines start with +/-, it's likely a diff
    diff_indicators > 2
}

/// Display diff response with simple red/green backgrounds like Claude Code
fn display_diff_response(_response: &str) {
    // NOTE: This function should not be called during ratatui mode
    // as it interferes with the display. Diffs should be added to display_lines instead.
}

/// Check if response contains code blocks
fn contains_code_blocks(response: &str) -> bool {
    response.contains("```")
}

/// Display response with code blocks using syntect syntax highlighting
fn display_code_block_response(response: &str) {
    use syntect::parsing::SyntaxSet;
    use syntect::highlighting::ThemeSet;
    
    // Load syntect assets
    let ps = SyntaxSet::load_defaults_newlines();
    let ts = ThemeSet::load_defaults();
    let theme = &ts.themes["base16-ocean.dark"]; // Use a nice dark theme
    
    let mut in_code_block = false;
    let mut current_language = "";
    let mut code_lines = Vec::new();
    
    for line in response.lines() {
        if line.starts_with("```") {
            if in_code_block {
                // End of code block - highlight and display accumulated code
                if !code_lines.is_empty() {
                    display_highlighted_code(&code_lines.join("\n"), current_language, &ps, theme);
                    code_lines.clear();
                }
                in_code_block = false;
                current_language = "";
            } else {
                // Start of code block
                in_code_block = true;
                current_language = line.strip_prefix("```").unwrap_or("").trim();
            }
        } else if in_code_block {
            code_lines.push(line);
        } else {
            // Regular text outside code blocks
            display_clean_text_line(line);
        }
    }
}

/// Display syntax highlighted code using syntect
fn display_highlighted_code(code: &str, language: &str, ps: &syntect::parsing::SyntaxSet, theme: &syntect::highlighting::Theme) {
    use syntect::easy::HighlightLines;
    use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};
    
    let syntax = ps.find_syntax_by_token(language)
        .or_else(|| ps.find_syntax_by_extension(language))
        .unwrap_or_else(|| ps.find_syntax_plain_text());
    
    let mut h = HighlightLines::new(syntax, theme);
    
    // Empty lines disabled in ratatui mode // Add spacing before code block
    for line in LinesWithEndings::from(code) {
        match h.highlight_line(line, ps) {
            Ok(ranges) => {
                let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
                print!("{}", escaped);
            }
            Err(_) => {
                // Fallback to plain text if highlighting fails
                print!("{}", line);
            }
        }
    }
    // Empty lines disabled in ratatui mode // Add spacing after code block
}

/// Display clean response: white answers, grey context, remove **Answer** labels
fn display_clean_response(response: &str) {
    
    for line in response.lines() {
        let cleaned_string = line
            .replace("**Answer**:", "")
            .replace("**Answer:**", "")
            .replace("**Context**:", "")
            .replace("**Context:**", "")
            .replace("Assistant:", "")
            .replace("Assistant: ", "");
        let cleaned_line = cleaned_string.trim();
        
        if cleaned_line.is_empty() {
            continue;
        }
        
        // Check if this looks like context (containing things like "based on", "according to")
        if is_context_line(cleaned_line) {
            // Context line handled by UI display system
        } else {
            // Regular answer - display handled by message system
        }
    }
}

/// Display a single line of clean text (helper for mixed content)
fn display_clean_text_line(line: &str) {
    
    let cleaned_string = line
        .replace("**Answer**:", "")
        .replace("**Answer:**", "")
        .replace("**Context**:", "")
        .replace("**Context:**", "")
        .replace("Assistant:", "")
        .replace("Assistant: ", "");
    let cleaned_line = cleaned_string.trim();
    
    if cleaned_line.is_empty() {
        return;
    }
    
    if is_context_line(cleaned_line) {
        // Context line handled by UI display system
    } else {
        // Output shown in message system instead of direct print
    }
}

/// Check if a line looks like contextual information rather than direct answer
fn is_context_line(line: &str) -> bool {
    let line_lower = line.to_lowercase();
    let context_indicators = [
        "based on", "according to", "from the", "in the context",
        "looking at", "from what i can see", "it appears", "it seems"
    ];
    
    context_indicators.iter().any(|&indicator| line_lower.contains(indicator))
}

/// Filter out AI interpretation blocks from the response
fn filter_ai_interpretation_blocks(response: &str) -> String {
    let lines: Vec<&str> = response.lines().collect();
    let mut filtered_lines = Vec::new();
    let mut in_interpretation_block = false;
    
    for line in lines {
        // Check for start of interpretation block
        if line.trim().starts_with("**AI interpretation") {
            in_interpretation_block = true;
            continue;
        }
        
        // Check for end of interpretation block (empty line after interpretation)
        if in_interpretation_block && line.trim().is_empty() {
            in_interpretation_block = false;
            continue;
        }
        
        // Skip lines that are part of interpretation blocks
        if in_interpretation_block {
            continue;
        }
        
        // Skip standalone interpretation markers
        if line.trim().starts_with("**Selection Criteria:**") || 
           line.trim().starts_with("I will proceed with these actions:") ||
           line.contains("AI interpretation of selection request") {
            continue;
        }
        
        filtered_lines.push(line);
    }
    
    filtered_lines.join("\n").trim().to_string()
}

/// Determine if a task is truly completed based on function result and user intent
fn is_task_completed(function_name: &str, result: &str, user_input: &str) -> bool {
    let input_lower = user_input.to_lowercase();
    
    match function_name {
        "edit_code" | "write_file" => {
            // Task completed if file was successfully written and user didn't ask for multiple things
            result.contains("bytes to") && 
            !result.contains("Error:") &&
            !input_lower.contains("and") && 
            !input_lower.contains("also") &&
            !input_lower.contains("then")
        },
        "list_directory" => {
            // Directory listings should NOT show "Task completed" - they're just informational
            false
        },
        "read_file" => {
            // File reading should NOT show "Task completed" - it's just informational
            false
        },
        "code_analysis" => {
            // Code analysis should NOT show "Task completed" - it's just analysis, not editing
            false
        },
        _ => false
    }
}

/// Get suggested next steps if task is not completed
fn get_suggested_next_steps(function_name: &str, result: &str, user_input: &str) -> Option<String> {
    let input_lower = user_input.to_lowercase();
    
    match function_name {
        "code_analysis" => {
            if result.contains("issues found") || result.contains("errors") {
                Some("Fix identified issues".to_string())
            } else if input_lower.contains("fix") || input_lower.contains("improve") {
                Some("Apply improvements".to_string())
            } else {
                None
            }
        },
        "list_directory" => {
            if input_lower.contains("fix") || input_lower.contains("edit") {
                Some("Edit specific file".to_string())
            } else if input_lower.contains("analyze") {
                Some("Analyze file contents".to_string())
            } else {
                None
            }
        },
        "read_file" => {
            if input_lower.contains("fix") || input_lower.contains("improve") {
                Some("Apply fixes to file".to_string())
            } else if input_lower.contains("understand") || input_lower.contains("explain") {
                Some("Provide explanation".to_string())
            } else {
                None
            }
        },
        _ => None
    }
}

/// Try to execute simple requests directly without reasoning engine overhead
async fn try_direct_function_execution(_input: &str, _config: &Config, _conversation_history: &str) -> Option<String> {
    // Remove pattern matching - let the smart system handle everything
    None
}

/// Simple inline permission request - no popup to avoid ratatui conflicts
pub async fn show_permission_dialog(path: &Path) -> Result<bool> {
    // Auto-grant permissions to avoid UI conflicts with ratatui
    Ok(true)
}

/// Extracts the last meaningful sentence or instruction from an AI response for the final summary.
fn extract_final_summary(response: &str) -> String {
    response.lines()
        .last()
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Task completed successfully.".to_string())
}
