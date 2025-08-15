use std::path::Path;
use std::fs;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::{anyhow, Result};
use html2text;
use crate::config::{Config};
use crate::gemini;
use crate::differ::{create_diff};
use crate::progress_display::{start_analyzing, start_editing, complete_operation, task_completed};

/// Edit strategy determined by intelligent analysis
#[derive(Debug, Clone)]
enum EditStrategy {
    Surgical,  // Precise, targeted edits using imara-diff
    Chunked,   // Large file chunked editing
    Direct,    // Direct full-file editing for simple cases
}
use ropey::Rope;
use crate::file_io::{read_file, write_file};
// Audio removed - keeping TrackInfo type alias for compatibility
type TrackInfo = String;
use reqwest::Client;
use std::time::Duration;
/// Track what actions we performed for context memory
fn track_action_performed(function_call: &FunctionCall, _result: &str) {
    let (action_summary, file_path_opt) = match function_call.name.as_str() {
        "edit_code" => {
            if let Some(file_path) = function_call.arguments.get("file_path").and_then(|v| v.as_str()) {
                // Clean action tracking without verbose output
                ("File edited".to_string(), Some(file_path.to_string()))
            } else {
                ("File edited".to_string(), None)
            }
        },
        "write_file" => {
            if let Some(file_path) = function_call.arguments.get("file_path").and_then(|v| v.as_str()) {
                ("File written".to_string(), Some(file_path.to_string()))
            } else {
                ("File written".to_string(), None)
            }
        },
        "rename_file" => {
            if let (Some(old_path), Some(new_path)) = (
                function_call.arguments.get("old_path").and_then(|v| v.as_str()),
                function_call.arguments.get("new_path").and_then(|v| v.as_str())
            ) {
                (format!("RENAMED_FILE: {} → {} (file renamed)", old_path, new_path), Some(new_path.to_string()))
            } else {
                ("RENAMED_FILE: unknown (file renamed)".to_string(), None)
            }
        },
        _ => return // Don't track read-only operations
    };
    
    // Store action for context only - no user display
    
    // Store globally for memory system to pick up
    if let Some(file_path) = file_path_opt {
        LAST_MODIFIED_FILES.lock().unwrap().push(LastAction {
            timestamp: std::time::SystemTime::now(),
            action: function_call.name.clone(),
            file_path,
            description: action_summary.clone(),
        });
    }
    
    // Silent action tracking - no user notifications
}

#[derive(Debug, Clone)]
struct LastAction {
    timestamp: std::time::SystemTime,
    action: String,
    file_path: String,
    description: String,
}

lazy_static::lazy_static! {
    static ref LAST_MODIFIED_FILES: std::sync::Mutex<Vec<LastAction>> = std::sync::Mutex::new(Vec::new());
}

/// Get recently modified files for emergency context
pub fn get_recent_modifications() -> Vec<String> {
    let actions = LAST_MODIFIED_FILES.lock().unwrap();
    let recent_cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(300); // Last 5 minutes
    
    actions.iter()
        .filter(|action| action.timestamp > recent_cutoff)
        .map(|action| format!("{}: {}", action.file_path, action.description))
        .collect()
}

/// Check if we should validate file integrity after this edit
fn should_validate_after_edit(function_call: &FunctionCall) -> bool {
    match function_call.name.as_str() {
        "edit_code" | "write_file" => {
            // Always validate HTML/JS/CSS files that could break display
            if let Some(file_path) = function_call.arguments.get("file_path").and_then(|v| v.as_str()) {
                let extension = std::path::Path::new(file_path)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("");
                matches!(extension, "html" | "htm" | "js" | "css" | "json")
            } else {
                false
            }
        },
        _ => false
    }
}

/// Validate edit integrity in background (non-blocking)
async fn validate_edit_integrity(function_call: FunctionCall, _result: String) {
    if let Some(file_path) = function_call.arguments.get("file_path").and_then(|v| v.as_str()) {
        // Brief delay to let file system settle
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        match tokio::fs::read_to_string(file_path).await {
            Ok(content) => {
                let issues = detect_potential_issues(&content, file_path);
                if !issues.is_empty() {
                    crate::thinking_display::PersistentStatusBar::add_reasoning_step(&format!(
                        "VALIDATION_WARNING: {} may have issues: {}", 
                        file_path, 
                        issues.join(", ")
                    ));
                }
            },
            Err(e) => {
                crate::thinking_display::PersistentStatusBar::add_reasoning_step(&format!(
                    "VALIDATION_ERROR: {} - {}", file_path, e
                ));
            }
        }
    }
}

/// Detect common issues that could break functionality
fn detect_potential_issues(content: &str, file_path: &str) -> Vec<String> {
    let mut issues = Vec::new();
    
    let extension = std::path::Path::new(file_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    match extension {
        "html" | "htm" => {
            // Check for actual structural issues, not implementation choices
            if content.contains("<script") && content.contains("src=") && !content.contains("defer") && !content.contains("async") {
                // Only warn about potentially blocking scripts, not missing canvas
            }
            if content.contains("<script") && content.contains("src=") {
                // Extract script src and check if files exist
                for line in content.lines() {
                    if line.contains("<script") && line.contains("src=") {
                        if let Some(src) = extract_script_src(line) {
                            let script_path = if src.starts_with("./") {
                                src.strip_prefix("./").unwrap_or(&src)
                            } else {
                                &src
                            };
                            
                            let full_path = std::path::Path::new(file_path)
                                .parent()
                                .unwrap_or(std::path::Path::new("."))
                                .join(script_path);
                                
                            if !full_path.exists() {
                                issues.push(format!("Referenced script '{}' not found", src));
                            }
                        }
                    }
                }
            }
        },
        "js" => {
            // Check for common JavaScript issues
            if content.contains("getElementById") && !content.contains("null") && !content.contains("querySelector") {
                // Only flag if no modern DOM methods are used
                issues.push("JavaScript uses getElementById without null checks".to_string());
            }
            // Remove the cube/canvas check - DOM-based cubes are valid
        },
        "css" => {
            // Check for CSS issues
            if content.contains("canvas") && !content.contains("display") {
                issues.push("CSS for canvas without display properties".to_string());
            }
        },
        _ => {}
    }
    
    issues
}

/// Extract script src attribute from HTML line
fn extract_script_src(line: &str) -> Option<String> {
    if let Some(start) = line.find("src=\"") {
        let start = start + 5; // Skip 'src="'
        if let Some(end) = line[start..].find('"') {
            return Some(line[start..start + end].to_string());
        }
    }
    None
}

/// Format file size in human-readable format
fn format_file_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size_f = size as f64;
    let mut unit_index = 0;
    
    while size_f >= 1024.0 && unit_index < UNITS.len() - 1 {
        size_f /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", size, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size_f, UNITS[unit_index])
    }
}

/// Registry for available functions that can be called by the AI
pub struct FunctionRegistry {
    functions: HashMap<String, FunctionDefinition>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
        };
        
        registry.register_core_functions();
        registry.register_advanced_functions();
        
        registry
    }

    fn register_core_functions(&mut self) {
        // File operations
        self.add_function("read_file", FunctionDefinition {
            name: "read_file".to_string(),
            description: "Read and display file contents ONLY when user explicitly asks to 'show', 'read', 'view', or 'display' file contents. Do NOT use when user mentions a file path for context or asks for specific information extraction.".to_string(),
            parameters: vec![
                Parameter {
                    name: "file_path".to_string(),
                    param_type: "string".to_string(),
                    description: "The path to the file to read".to_string(),
                    required: true,
                }
            ],
        });

        self.add_function("write_file", FunctionDefinition {
            name: "write_file".to_string(),
            description: "Create a new file with the specified content. The generated content should be complete, high-quality, and production-ready.".to_string(),
            parameters: vec![
                Parameter {
                    name: "file_path".to_string(),
                    param_type: "string".to_string(),
                    description: "The path to the file to write".to_string(),
                    required: true,
                },
                Parameter {
                    name: "content".to_string(),
                    param_type: "string".to_string(),
                    description: "The content to write to the file".to_string(),
                    required: true,
                }
            ],
        });

        self.add_function("list_directory", FunctionDefinition {
            name: "list_directory".to_string(),
            description: "List files and directories in a given path".to_string(),
            parameters: vec![
                Parameter {
                    name: "directory_path".to_string(),
                    param_type: "string".to_string(),
                    description: "The path to the directory to list".to_string(),
                    required: true,
                }
            ],
        });

        self.add_function("find_files", FunctionDefinition {
            name: "find_files".to_string(),
            description: "Search for files by name pattern in a directory and subdirectories".to_string(),
            parameters: vec![
                Parameter {
                    name: "directory_path".to_string(),
                    param_type: "string".to_string(),
                    description: "The directory path to search in (e.g., 'C:\\Users\\Admin\\Desktop\\random')".to_string(),
                    required: true,
                },
                Parameter {
                    name: "pattern".to_string(),
                    param_type: "string".to_string(),
                    description: "The filename pattern to search for (e.g., 'rubiks' will find files containing 'rubiks')".to_string(),
                    required: true,
                }
            ],
        });

        // Intelligent search within files
        self.add_function("search_in_file", FunctionDefinition {
            name: "search_in_file".to_string(),
            description: "Search for specific patterns, code, or text within a file with context - like Claude Code's search".to_string(),
            parameters: vec![
                Parameter {
                    name: "file_path".to_string(),
                    param_type: "string".to_string(),
                    description: "The path to the file to search in".to_string(),
                    required: true,
                },
                Parameter {
                    name: "query".to_string(),
                    param_type: "string".to_string(),
                    description: "The search query - can be text, patterns like 'android debug', 'console.log', or 'iOS debugging'".to_string(),
                    required: true,
                },
                Parameter {
                    name: "context_lines".to_string(),
                    param_type: "number".to_string(),
                    description: "Number of lines of context to show around matches (default: 3)".to_string(),
                    required: false,
                }
            ],
        });

        // Web operations
        self.add_function("fetch_url", FunctionDefinition {
            name: "fetch_url".to_string(),
            description: "Fetch content from a URL".to_string(),
            parameters: vec![
                Parameter {
                    name: "url".to_string(),
                    param_type: "string".to_string(),
                    description: "The URL to fetch".to_string(),
                    required: true,
                }
            ],
        });

        // File management operations  
        self.add_function("rename_file", FunctionDefinition {
            name: "rename_file".to_string(),
            description: "Rename or move a file to a new name/location".to_string(),
            parameters: vec![
                Parameter {
                    name: "old_path".to_string(),
                    param_type: "string".to_string(),
                    description: "Current path/name of the file".to_string(),
                    required: true,
                },
                Parameter {
                    name: "new_path".to_string(),
                    param_type: "string".to_string(),
                    description: "New path/name for the file".to_string(),
                    required: true,
                }
            ],
        });

        // Audio operations
        // Audio functions removed
    }

    fn register_advanced_functions(&mut self) {
        self.add_function("diff_files", FunctionDefinition {
            name: "diff_files".to_string(),
            description: "Compare two files and show differences".to_string(),
            parameters: vec![
                Parameter {
                    name: "file1_path".to_string(),
                    param_type: "string".to_string(),
                    description: "Path to the first file".to_string(),
                    required: true,
                },
                Parameter {
                    name: "file2_path".to_string(),
                    param_type: "string".to_string(),
                    description: "Path to the second file".to_string(),
                    required: true,
                }
            ],
        });

        self.add_function("code_analysis", FunctionDefinition {
            name: "code_analysis".to_string(),
            description: "Analyze code to understand what needs to be fixed. Use this before edit_code for complex fixes.".to_string(),
            parameters: vec![
                Parameter {
                    name: "file_path".to_string(),
                    param_type: "string".to_string(),
                    description: "Path to the code file to analyze".to_string(),
                    required: true,
                }
            ],
        });

        self.add_function("edit_code", FunctionDefinition {
            name: "edit_code".to_string(),
            description: "Intelligently edit, modify, or refactor code in an existing file. This function reads the file, applies high-quality changes, and writes it back. The edits should be complete and follow best practices.".to_string(),
            parameters: vec![
                Parameter {
                    name: "file_path".to_string(),
                    param_type: "string".to_string(),
                    description: "Path to the code file to edit".to_string(),
                    required: true,
                },
                Parameter {
                    name: "query".to_string(),
                    param_type: "string".to_string(),
                    description: "Instructions for what changes to make to the code".to_string(),
                    required: true,
                }
            ],
        });
    }

    fn add_function(&mut self, name: &str, definition: FunctionDefinition) {
        self.functions.insert(name.to_string(), definition);
    }

    pub fn get_function_definitions(&self) -> Vec<FunctionDefinition> {
        self.functions.values().cloned().collect()
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionDefinition> {
        self.functions.get(name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<Parameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// Execute a function call with automatic error recovery
pub async fn execute_function_call(
    function_call: &FunctionCall,
    config: &Config,
    conversation_history: &str,
) -> Result<String> {
    // Initialize real-time UI
    crate::progress_display::init_realtime_ui();
    
    // Start real-time UI for function calls - show immediately
    let operation_id = match function_call.name.as_str() {
        "edit_code" => {
            // Estimate token count from query length
            let query_tokens = function_call.arguments.get("query")
                .and_then(|v| v.as_str())
                .map(|q| (q.len() / 4) as u32)
                .unwrap_or(1000);
            Some(start_editing(query_tokens))
        },
        "code_analysis" => {
            // Estimate token count
            let analysis_tokens = 1669; // Default estimate
            Some(start_analyzing(analysis_tokens))
        },
        _ => None
    };
    
    // Try the function call with retry logic for transient failures
    let mut attempt = 1;
    const MAX_ATTEMPTS: u8 = 3;
    
    loop {
        let result = match function_call.name.as_str() {
            "read_file" => execute_read_file_with_fallback(function_call).await,
            "write_file" => execute_write_file(function_call).await,
            "rename_file" => execute_rename_file(function_call).await,
            "list_directory" => execute_list_directory(function_call).await,
            "find_files" => execute_find_files(function_call).await,
            "search_in_file" => execute_search_in_file(function_call).await,
            "fetch_url" => execute_fetch_url(function_call).await,
            "diff_files" => execute_diff_files(function_call).await,
            "code_analysis" => execute_code_analysis(function_call, config, conversation_history).await,
            "edit_code" => execute_edit_code_with_recovery(function_call, config, conversation_history).await,
            _ => Err(anyhow!("Unknown function: {}", function_call.name)),
        };
        
        match result {
            Ok(success_result) => {
                // Complete the real-time UI operation with validation for edits
                if let Some(op_id) = &operation_id {
                    let completion_msg = match function_call.name.as_str() {
                        "edit_code" => {
                            // Validate the edit and provide specific feedback
                            validate_edit_completion(function_call, &success_result).await
                        },
                        "code_analysis" => "Code analysis completed successfully".to_string(), 
                        _ => "Operation completed".to_string()
                    };
                    complete_operation(op_id, &completion_msg);
                }
                
                // Track file modifications for context memory
                track_action_performed(function_call, &success_result);
                
                return Ok(success_result);
            }
            Err(e) if attempt < MAX_ATTEMPTS && is_retryable_error(&e) => {
                crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Retry attempt {} for {}: {}", attempt + 1, function_call.name, e));
                crate::thinking_display::PersistentStatusBar::add_reasoning_step(&format!("{} failed, retrying (attempt {})", function_call.name, attempt + 1));
                attempt += 1;
                // Brief delay before retry
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                continue;
            },
            Err(e) => {
                // Function failed - keep it simple
                return Err(e);
            }
        }
    }
}

/// Check if an error is worth retrying
fn is_retryable_error(error: &anyhow::Error) -> bool {
    let error_str = error.to_string().to_lowercase();
    error_str.contains("timeout") ||
    error_str.contains("connection") ||
    error_str.contains("network") ||
    error_str.contains("temporary") ||
    error_str.contains("rate limit")
}

/// Create function calling prompt for Gemini API
pub fn create_function_calling_prompt(functions: &[FunctionDefinition]) -> String {
    let mut prompt = String::from("You are an AI assistant with access to these functions:\n\n");
    
    for function in functions {
        prompt.push_str(&format!("Function: {}\n", function.name));
        prompt.push_str(&format!("Description: {}\n", function.description));
        prompt.push_str("Parameters:\n");
        
        for param in &function.parameters {
            let required_text = if param.required { " (required)" } else { " (optional)" };
            prompt.push_str(&format!("  - {}: {} - {}{}\n", 
                param.name, param.param_type, param.description, required_text));
        }
        
        prompt.push('\n');
    }
    
    prompt.push_str("\nWhen you need to use a function, respond with JSON in this format:\n");
    prompt.push_str("{\n  \"function_call\": {\n    \"name\": \"function_name\",\n    \"arguments\": {\n      \"parameter_name\": \"value\"\n    }\n  },\n  \"reasoning\": \"Brief explanation of why you're calling this function\"\n}\n\n");
    prompt.push_str("EXAMPLES:\n");
    prompt.push_str("edit_code: {\"function_call\": {\"name\": \"edit_code\", \"arguments\": {\"file_path\": \"script.js\", \"query\": \"fix the JavaScript errors\"}}}\n");
    prompt.push_str("read_file: {\"function_call\": {\"name\": \"read_file\", \"arguments\": {\"file_path\": \"index.html\"}}} // ONLY when user asks to view/see file\n");
    prompt.push_str("list_directory: {\"function_call\": {\"name\": \"list_directory\", \"arguments\": {\"directory_path\": \"C:\\\\Users\\\\Admin\\\\Desktop\\\\random\"}}}\n\n");
    prompt.push_str("CRITICAL FUNCTION USAGE RULES:\n");
    prompt.push_str("- User says 'what is in [folder]' → USE list_directory with full path\n");
    prompt.push_str("- User says 'show me/read/view [file]' → USE read_file\n");
    prompt.push_str("- User says 'fix/improve/edit [file]' → For simple fixes: edit_code directly. For complex fixes: code_analysis then edit_code\n");
    prompt.push_str("- User says 'create new file X' → USE write_file to create it from scratch\n");
    prompt.push_str("- NEVER call read_file before editing - edit_code reads the file automatically\n");
    prompt.push_str("- Complex fixes (missing functions, major features): code_analysis → edit_code\n");
    prompt.push_str("- Simple fixes (typos, small changes): edit_code directly\n\n");
    
    prompt.push_str("IMPORTANT: Use exact parameter names as listed above. Be smart about user intent - don't fail because of minor ambiguities.\n\n");
    
    prompt
}

#[derive(Debug, Clone)]
pub struct PromptContext {
    pub recent_errors: Vec<String>,
    pub success_patterns: Vec<String>,
    pub user_preferences: Vec<String>,
    pub complexity_level: ComplexityLevel,
    pub task_type: TaskType,
    pub confidence_feedback: Option<crate::gemini::ConfidenceAssessment>,
}

#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Simple,   // Basic operations, single file changes
    Medium,   // Multi-file changes, moderate logic
    Complex,  // Architecture changes, advanced features
}

#[derive(Debug, Clone)]
pub enum TaskType {
    Reading,      // File viewing, analysis
    Editing,      // Code changes, fixes
    Creating,     // New files, features
    Debugging,    // Error investigation, troubleshooting
    Analysis,     // Code review, system analysis
}

impl Default for PromptContext {
    fn default() -> Self {
        Self {
            recent_errors: vec![],
            success_patterns: vec![],
            user_preferences: vec![],
            complexity_level: ComplexityLevel::Medium,
            task_type: TaskType::Editing,
            confidence_feedback: None,
        }
    }
}

pub fn create_dynamic_function_calling_prompt(functions: &[FunctionDefinition], context: &PromptContext) -> String {
    let mut prompt = String::new();
    
    // Adaptive system role based on context
    match context.task_type {
        TaskType::Reading => prompt.push_str("You are a code analysis expert focused on thorough examination and explanation of code structures.\n\n"),
        TaskType::Editing => prompt.push_str("You are a precise code editor focused on making targeted, effective improvements.\n\n"),
        TaskType::Creating => prompt.push_str("You are a creative developer focused on building robust, well-structured new code.\n\n"),
        TaskType::Debugging => prompt.push_str("You are a systematic debugger focused on identifying and resolving issues methodically.\n\n"),
        TaskType::Analysis => prompt.push_str("You are a system architect focused on analyzing patterns, performance, and architectural decisions.\n\n"),
    }
    
    // Add confidence-based guidance
    if let Some(confidence) = &context.confidence_feedback {
        match confidence.level {
            crate::gemini::ConfidenceLevel::VeryLow | crate::gemini::ConfidenceLevel::Low => {
                prompt.push_str("IMPORTANT: Your recent responses showed low confidence. Take extra care to:\n");
                prompt.push_str("- Analyze the problem thoroughly before acting\n");
                prompt.push_str("- Use code_analysis when unsure about complex changes\n");
                prompt.push_str("- Ask for clarification if the request is ambiguous\n\n");
            },
            _ => {}
        }
    }
    
    // Adaptive complexity guidance
    match context.complexity_level {
        ComplexityLevel::Simple => {
            prompt.push_str("Focus on direct, efficient solutions. For simple tasks, use edit_code directly.\n\n");
        },
        ComplexityLevel::Complex => {
            prompt.push_str("Use systematic analysis for complex tasks. Always use code_analysis before major changes.\n\n");
        },
        _ => {}
    }
    
    // Learn from recent errors
    if !context.recent_errors.is_empty() {
        prompt.push_str("RECENT ERROR PATTERNS TO AVOID:\n");
        for error in &context.recent_errors {
            prompt.push_str(&format!("- {}\n", error));
        }
        prompt.push('\n');
    }
    
    // Reinforce successful patterns
    if !context.success_patterns.is_empty() {
        prompt.push_str("PROVEN SUCCESSFUL APPROACHES:\n");
        for pattern in &context.success_patterns {
            prompt.push_str(&format!("- {}\n", pattern));
        }
        prompt.push('\n');
    }
    
    prompt.push_str("Available functions:\n\n");
    
    for function in functions {
        prompt.push_str(&format!("Function: {}\n", function.name));
        prompt.push_str(&format!("Description: {}\n", function.description));
        prompt.push_str("Parameters:\n");
        
        for param in &function.parameters {
            let required_text = if param.required { " (required)" } else { " (optional)" };
            prompt.push_str(&format!("  - {}: {} - {}{}\n", 
                param.name, param.param_type, param.description, required_text));
        }
        
        prompt.push('\n');
    }
    
    prompt.push_str("\nWhen you need to use a function, respond with JSON in this format:\n");
    prompt.push_str("{\n  \"function_call\": {\n    \"name\": \"function_name\",\n    \"arguments\": {\n      \"parameter_name\": \"value\"\n    }\n  },\n  \"reasoning\": \"Brief explanation of why you're calling this function\"\n}\n\n");
    
    // Context-adaptive examples
    match context.task_type {
        TaskType::Reading | TaskType::Analysis => {
            prompt.push_str("EXAMPLES FOR ANALYSIS:\n");
            prompt.push_str("read_file: {\"function_call\": {\"name\": \"read_file\", \"arguments\": {\"file_path\": \"index.html\"}}}\n");
            prompt.push_str("code_analysis: {\"function_call\": {\"name\": \"code_analysis\", \"arguments\": {\"file_path\": \"script.js\", \"query\": \"analyze performance bottlenecks\"}}}\n");
        },
        TaskType::Editing | TaskType::Debugging => {
            prompt.push_str("EXAMPLES FOR EDITING:\n");
            prompt.push_str("edit_code: {\"function_call\": {\"name\": \"edit_code\", \"arguments\": {\"file_path\": \"script.js\", \"query\": \"fix the JavaScript errors\"}}}\n");
            prompt.push_str("code_analysis: {\"function_call\": {\"name\": \"code_analysis\", \"arguments\": {\"file_path\": \"script.js\", \"query\": \"identify the cause of runtime errors\"}}}\n");
        },
        TaskType::Creating => {
            prompt.push_str("EXAMPLES FOR CREATION:\n");
            prompt.push_str("write_file: {\"function_call\": {\"name\": \"write_file\", \"arguments\": {\"file_path\": \"new_component.js\", \"content\": \"// New component code\"}}}\n");
            prompt.push_str("list_directory: {\"function_call\": {\"name\": \"list_directory\", \"arguments\": {\"directory_path\": \"C:\\\\Users\\\\Admin\\\\Desktop\\\\project\"}}}\n");
        },
    }
    
    prompt.push_str("\nCRITICAL FUNCTION USAGE RULES:\n");
    prompt.push_str("- User says 'fix/improve/edit [file]' → For simple fixes: edit_code directly. For complex fixes: code_analysis then edit_code\n");
    prompt.push_str("- User says 'show me/read/view/what's in [file]' → USE read_file\n");
    prompt.push_str("- User says 'create new file X' → USE write_file to create it from scratch\n");
    prompt.push_str("- NEVER call read_file before editing - edit_code reads the file automatically\n");
    prompt.push_str("- Complex fixes (missing functions, major features): code_analysis → edit_code\n");
    prompt.push_str("- Simple fixes (typos, small changes): edit_code directly\n\n");
    
    prompt.push_str("PROACTIVE CONTEXT ANALYSIS:\n");
    prompt.push_str("- When user provides a file path for editing/fixing: Use code_analysis for complex issues, edit_code for simple ones\n");
    prompt.push_str("- When user provides a file path for viewing: use read_file\n");
    prompt.push_str("- When user says 'you don't see anything wrong': refer to recent conversation and re-examine the relevant files\n");
    prompt.push_str("- When user reports issues (blank page, not working): systematically check related files to find root cause\n");
    prompt.push_str("- Use conversation context to understand what files are relevant to the current discussion\n");
    prompt.push_str("- Be action-oriented: Analyze complex problems first, then fix them with targeted edits\n");
    prompt.push_str("- For directory requests like 'random folder' + 'on desktop': use full paths like C:\\Users\\Admin\\Desktop\\random\n");
    prompt.push_str("- When operations fail, analyze conversation context for missing path information\n\n");
    
    prompt.push_str("IMPORTANT: Use exact parameter names as listed above. Be smart about user intent - don't fail because of minor ambiguities.\n\n");
    
    prompt
}

// Function implementations

/// Smart read file with fallback to creation if file doesn't exist and intent suggests creation
async fn execute_read_file_with_fallback(function_call: &FunctionCall) -> Result<String> {
    match execute_read_file(function_call).await {
        Ok(result) => Ok(result),
        Err(e) => {
            let file_path = function_call.arguments.get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            
            // If file doesn't exist, suggest creation based on context
            if e.to_string().contains("No such file") || e.to_string().contains("cannot find") {
                Ok(format!("File '{}' does not exist. Consider using write_file to create it first, or check if the path is correct.", file_path))
            } else {
                Err(e)
            }
        }
    }
}

async fn execute_read_file(function_call: &FunctionCall) -> Result<String> {
    let file_path = function_call.arguments.get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file_path parameter"))?;
    
    crate::thinking_display::PersistentStatusBar::add_reasoning_step(&format!("Looking for file: {}", file_path));
    
    // Use existing smart file discovery from chat.rs
    let actual_path = if Path::new(file_path).exists() {
        file_path.to_string()
    } else {
        crate::thinking_display::PersistentStatusBar::add_reasoning_step("File not found, searching similar files");
        // Use the existing smart_file_discovery function
        use crate::cli::chat::smart_file_discovery;
        use crate::config::Config;
        let default_config = Config::default();
        smart_file_discovery(file_path, &default_config).await
            .unwrap_or_else(|| file_path.to_string())
    };
    
    let content = read_file(Path::new(&actual_path)).await?;
    
    // For very large files, show a preview instead of dumping everything
    let line_count = content.lines().count();
    let char_count = content.len();
    
    crate::thinking_display::PersistentStatusBar::add_reasoning_step(&format!("Read {} with {} lines, {} chars", actual_path, line_count, char_count));
    
    // Return full content - AI needs complete file to edit properly
    let result = format!("File content of '{}' ({} lines, {} chars):\n\n{}", 
                        file_path, line_count, char_count, content);
    
    Ok(result)
}

async fn execute_write_file(function_call: &FunctionCall) -> Result<String> {
    let file_path = function_call.arguments.get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file_path parameter"))?;
    
    let content = function_call.arguments.get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing content parameter"))?;
    
    write_file(Path::new(file_path), content).await?;
    Ok(format!("Successfully wrote {} bytes to '{}'", content.len(), file_path))
}

async fn execute_rename_file(function_call: &FunctionCall) -> Result<String> {
    let old_path = function_call.arguments.get("old_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing old_path parameter"))?;
        
    let new_path = function_call.arguments.get("new_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing new_path parameter"))?;
    
    let old_path_obj = Path::new(old_path);
    let new_path_obj = Path::new(new_path);
    
    if !old_path_obj.exists() {
        return Err(anyhow!("Source file does not exist: {}", old_path));
    }
    
    // Use std::fs::rename for atomic file rename
    std::fs::rename(old_path_obj, new_path_obj)
        .map_err(|e| anyhow!("Failed to rename '{}' to '{}': {}", old_path, new_path, e))?;
    
    Ok(format!("Successfully renamed '{}' to '{}'", old_path, new_path))
}

async fn execute_list_directory(function_call: &FunctionCall) -> Result<String> {
    let directory_path = function_call.arguments.get("directory_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing directory_path parameter"))?;
    
    let path = Path::new(directory_path);
    if !path.exists() {
        return Err(anyhow!("Directory does not exist: {}", directory_path));
    }
    
    let entries = fs::read_dir(path)?;
    let mut result = format!("📁 {}\n", directory_path);
    
    // Collect entries and sort them (directories first, then files)
    let mut entries_vec = Vec::new();
    for entry in entries {
        let entry = entry?;
        let metadata = entry.metadata()?;
        entries_vec.push((entry, metadata));
    }
    
    // Sort: directories first, then files, both alphabetically
    entries_vec.sort_by(|(a, a_meta), (b, b_meta)| {
        match (a_meta.is_dir(), b_meta.is_dir()) {
            (true, false) => std::cmp::Ordering::Less,  // Directories first
            (false, true) => std::cmp::Ordering::Greater, // Files after directories
            _ => a.file_name().cmp(&b.file_name()),      // Alphabetical within same type
        }
    });
    
    for (i, (entry, metadata)) in entries_vec.iter().enumerate() {
        let is_last = i == entries_vec.len() - 1;
        let prefix = if is_last { "└── " } else { "├── " };
        
        if metadata.is_dir() {
            result.push_str(&format!("{}📂 {}\n", 
                prefix,
                entry.file_name().to_string_lossy()
            ));
        } else {
            let size = metadata.len();
            let size_str = format_file_size(size);
            result.push_str(&format!("{}📄 {} ({})\n", 
                prefix,
                entry.file_name().to_string_lossy(),
                size_str
            ));
        }
    }
    
    Ok(result)
}

async fn execute_find_files(function_call: &FunctionCall) -> Result<String> {
    let directory_path = function_call.arguments.get("directory_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing directory_path parameter"))?;
    
    let pattern = function_call.arguments.get("pattern")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing pattern parameter"))?;
    
    let path = Path::new(directory_path);
    if !path.exists() {
        return Err(anyhow!("Directory does not exist: {}", directory_path));
    }
    
    let pattern_lower = pattern.to_lowercase();
    let mut found_files = Vec::new();
    
    // Search recursively through directories
    fn search_directory(dir: &Path, pattern: &str, found_files: &mut Vec<String>) -> Result<()> {
        let entries = fs::read_dir(dir)?;
        
        for entry in entries {
            let entry = entry?;
            let entry_path = entry.path();
            
            if entry_path.is_file() {
                let file_name = entry.file_name().to_string_lossy().to_lowercase();
                if file_name.contains(pattern) {
                    found_files.push(entry_path.to_string_lossy().to_string());
                }
            } else if entry_path.is_dir() {
                // Recursively search subdirectories
                if let Err(_) = search_directory(&entry_path, pattern, found_files) {
                    // Continue searching other directories even if one fails
                    // Skip failed directories silently to avoid UI corruption
                }
            }
        }
        Ok(())
    }
    
    search_directory(path, &pattern_lower, &mut found_files)?;
    
    if found_files.is_empty() {
        Ok(format!("No files containing '{}' found in '{}'", pattern, directory_path))
    } else {
        let mut result = format!("Found {} file(s) containing '{}':\n\n", found_files.len(), pattern);
        for file in found_files {
            result.push_str(&format!("📁 {}\n", file));
        }
        Ok(result)
    }
}

/// Execute intelligent search within a file - like Claude Code's search capability
async fn execute_search_in_file(function_call: &FunctionCall) -> Result<String> {
    let file_path = function_call.arguments.get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file_path parameter"))?;
    
    let search_query = function_call.arguments.get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing query parameter"))?;
    
    let context_lines = function_call.arguments.get("context_lines")
        .and_then(|v| v.as_u64())
        .unwrap_or(3) as usize;
    
    // Read file content
    let content = match tokio::fs::read_to_string(file_path).await {
        Ok(content) => content,
        Err(e) => return Err(anyhow!("Failed to read file '{}': {}", file_path, e)),
    };
    
    let lines: Vec<&str> = content.lines().collect();
    let mut matches = Vec::new();
    
    // Search for patterns (case-insensitive by default)
    let query_lower = search_query.to_lowercase();
    
    for (line_idx, line) in lines.iter().enumerate() {
        let line_lower = line.to_lowercase();
        
        // Check for direct text matches
        if line_lower.contains(&query_lower) {
            matches.push((line_idx + 1, line, "direct_match"));
            continue;
        }
        
        // Check for pattern matches based on common debugging patterns
        if is_debugging_related(&query_lower) {
            if is_debug_line(line) {
                matches.push((line_idx + 1, line, "debug_pattern"));
                continue;
            }
        }
        
        // Check for Android-specific patterns
        if query_lower.contains("android") {
            if line_lower.contains("android") || 
               line_lower.contains("blink") && (line_lower.contains("fix") || line_lower.contains("render")) {
                matches.push((line_idx + 1, line, "android_pattern"));
                continue;
            }
        }
        
        // Check for iOS patterns
        if query_lower.contains("ios") {
            if line_lower.contains("ios") || 
               line_lower.contains("webkit") || 
               line_lower.contains("safari") {
                matches.push((line_idx + 1, line, "ios_pattern"));
                continue;
            }
        }
    }
    
    if matches.is_empty() {
        return Ok(format!("No matches found for '{}' in '{}'", search_query, file_path));
    }
    
    // Format results with context
    let mut result = format!("Found {} match(es) for '{}' in '{}':\n\n", matches.len(), search_query, file_path);
    
    for (line_num, _matched_line, match_type) in &matches {
        result.push_str(&format!("📍 Line {}: ({})\n", line_num, match_type));
        
        // Add context lines
        let start_idx = if *line_num > context_lines + 1 { line_num - context_lines - 1 } else { 0 };
        let end_idx = std::cmp::min(line_num + context_lines, lines.len());
        
        for i in start_idx..end_idx {
            let prefix = if i + 1 == *line_num { ">>>" } else { "   " };
            result.push_str(&format!("{} {}: {}\n", prefix, i + 1, lines[i]));
        }
        result.push_str("\n");
    }
    
    Ok(result)
}

/// Check if query is related to debugging
fn is_debugging_related(query: &str) -> bool {
    let debug_keywords = [
        "debug", "console", "log", "print", "trace", "error", "warning", 
        "testing", "test", "development", "dev", "temp", "temporary", "todo", "fixme"
    ];
    
    debug_keywords.iter().any(|&keyword| query.contains(keyword))
}

/// Check if a line contains debugging code
fn is_debug_line(line: &str) -> bool {
    let line_lower = line.trim().to_lowercase();
    
    // Console logging
    if line_lower.contains("console.log") || 
       line_lower.contains("console.error") || 
       line_lower.contains("console.warn") || 
       line_lower.contains("console.debug") {
        return true;
    }
    
    // Print statements
    if line_lower.contains("print(") || 
       line_lower.contains("println!") || 
       line_lower.contains("dbg!") {
        return true;
    }
    
    // Comments with debug keywords
    if (line_lower.starts_with("//") || line_lower.starts_with("/*")) &&
       (line_lower.contains("debug") || 
        line_lower.contains("todo") || 
        line_lower.contains("fixme") || 
        line_lower.contains("temp")) {
        return true;
    }
    
    // Development/testing code
    if line_lower.contains("// dev") || 
       line_lower.contains("// test") || 
       line_lower.contains("// debug") {
        return true;
    }
    
    false
}

/// Extract search keywords from user query (like Claude Code does)
fn extract_search_keywords_from_query(query: &str) -> Vec<String> {
    let mut keywords = Vec::new();
    let query_lower = query.to_lowercase();
    
    // Direct platform mentions
    if query_lower.contains("android") {
        keywords.push("android".to_string());
        keywords.push("blink".to_string()); // Android often uses Blink rendering
    }
    if query_lower.contains("ios") {
        keywords.push("ios".to_string());
        keywords.push("webkit".to_string());
        keywords.push("safari".to_string());
    }
    
    // Debug/logging keywords
    if query_lower.contains("debug") || query_lower.contains("debugging") {
        keywords.push("debug".to_string());
        keywords.push("console.log".to_string());
        keywords.push("console.error".to_string());
    }
    if query_lower.contains("console") || query_lower.contains("log") {
        keywords.push("console".to_string());
        keywords.push("log".to_string());
    }
    
    // Development keywords
    if query_lower.contains("testing") || query_lower.contains("test") {
        keywords.push("test".to_string());
        keywords.push("testing".to_string());
    }
    if query_lower.contains("development") || query_lower.contains("dev") {
        keywords.push("dev".to_string());
        keywords.push("development".to_string());
    }
    
    // Specific code patterns mentioned in query
    if query_lower.contains("blinkrenderingfixes") {
        keywords.push("BlinkRenderingFixes".to_string());
    }
    if query_lower.contains("rendering") {
        keywords.push("rendering".to_string());
        keywords.push("render".to_string());
    }
    if query_lower.contains("fixes") || query_lower.contains("fix") {
        keywords.push("fixes".to_string());
        keywords.push("fix".to_string());
    }
    
    // Remove duplicates and return
    keywords.sort();
    keywords.dedup();
    
    // If no specific keywords found, try to extract important words
    if keywords.is_empty() {
        let words: Vec<&str> = query.split_whitespace().collect();
        for word in words {
            let word_clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if word_clean.len() > 3 { // Only include substantial words
                keywords.push(word_clean.to_string());
            }
        }
    }
    
    keywords
}

/// Check if line matches platform-specific debugging patterns
fn matches_platform_debugging(line: &str, keyword: &str) -> bool {
    let line_lower = line.to_lowercase();
    let keyword_lower = keyword.to_lowercase();
    
    if keyword_lower.contains("android") || keyword_lower.contains("blink") {
        return line_lower.contains("android") || 
               (line_lower.contains("blink") && (line_lower.contains("fix") || line_lower.contains("render")));
    }
    
    if keyword_lower.contains("ios") {
        return line_lower.contains("ios") || 
               line_lower.contains("webkit") || 
               line_lower.contains("safari");
    }
    
    false
}

#[derive(Debug, Clone)]
struct EditableSection {
    start_line: usize,
    end_line: usize,
    match_count: usize,
    keywords: Vec<String>,
}

impl EditableSection {
    fn line_count(&self) -> usize {
        if self.end_line >= self.start_line {
            self.end_line - self.start_line + 1
        } else {
            0
        }
    }
}

/// Group nearby matches into contiguous sections that should be edited together
fn group_matches_into_sections(matches: &[(usize, usize, String)]) -> Vec<EditableSection> {
    if matches.is_empty() {
        return Vec::new();
    }
    
    let mut sections = Vec::new();
    let tolerance = 10; // Lines within 10 of each other are grouped together
    
    let mut current_start = matches[0].1;
    let mut current_end = matches[0].1;
    let mut current_keywords = vec![matches[0].2.clone()];
    let mut current_count = 1;
    
    for i in 1..matches.len() {
        let (_, line_idx, keyword) = &matches[i];
        
        if *line_idx <= current_end + tolerance {
            // Extend current section
            current_end = *line_idx;
            current_keywords.push(keyword.clone());
            current_count += 1;
        } else {
            // Start new section
            sections.push(EditableSection {
                start_line: current_start,
                end_line: current_end,
                match_count: current_count,
                keywords: current_keywords.clone(),
            });
            
            current_start = *line_idx;
            current_end = *line_idx;
            current_keywords = vec![keyword.clone()];
            current_count = 1;
        }
    }
    
    // Add final section
    sections.push(EditableSection {
        start_line: current_start,
        end_line: current_end,
        match_count: current_count,
        keywords: current_keywords,
    });
    
    sections
}

/// Choose the most important section to edit when there are multiple options
fn choose_main_target_section(sections: &[EditableSection], query: &str) -> EditableSection {
    if sections.is_empty() {
        return EditableSection {
            start_line: 0,
            end_line: 0,
            match_count: 0,
            keywords: Vec::new(),
        };
    }
    
    // Score each section based on relevance to the query
    let mut best_section = sections[0].clone();
    let mut best_score = score_section_relevance(&sections[0], query);
    
    for section in &sections[1..] {
        let score = score_section_relevance(section, query);
        if score > best_score {
            best_score = score;
            best_section = section.clone();
        }
    }
    
    best_section
}

/// Score how relevant a section is to the user's query
fn score_section_relevance(section: &EditableSection, query: &str) -> f32 {
    let mut score = 0.0;
    let query_lower = query.to_lowercase();
    
    // Higher score for more matches in the section
    score += section.match_count as f32 * 2.0;
    
    // Higher score for platform-specific matches if query mentions platform
    if query_lower.contains("android") || query_lower.contains("debug") {
        for keyword in &section.keywords {
            if keyword.contains("android") || keyword.contains("blink") || keyword.contains("debug") {
                score += 5.0;
            }
        }
    }
    
    // Higher score for sections with more relevant keywords
    for keyword in &section.keywords {
        if query_lower.contains(&keyword.to_lowercase()) {
            score += 3.0;
        }
    }
    
    // Prefer larger sections (more comprehensive removal)
    score += section.line_count() as f32 * 0.1;
    
    score
}

/// Edit a targeted section of the file with intelligent block removal
async fn edit_targeted_section(content: &str, section: &EditableSection, query: &str, config: &Config) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    
    // Expand section to include complete code blocks
    let (block_start, block_end) = find_complete_code_blocks(&lines, section.start_line, section.end_line);
    
    // Add context around the complete blocks
    let context_size = 10;
    let actual_start = if block_start > context_size { 
        block_start - context_size 
    } else { 
        0 
    };
    let actual_end = std::cmp::min(block_end + context_size, lines.len());
    
    let section_content = lines[actual_start..actual_end].join("\n");
    
    // Ask for the full edited content directly
    let edit_prompt = create_intelligent_edit_prompt(query, &section_content, actual_start + 1, actual_end + 1, block_start + 1, block_end + 1);
    
    match crate::gemini::query_gemini(&edit_prompt, config).await {
        Ok(patched_section) => {
            // Reconstruct the full file
            let before_content = if actual_start > 0 {
                lines[..actual_start].join("\n") + "\n"
            } else {
                String::new()
            };

            let after_content = if actual_end < lines.len() {
                "\n".to_string() + &lines[actual_end..].join("\n")
            } else {
                String::new()
            };

            Ok(format!("{}{}{}", before_content, patched_section, after_content))
        }
        Err(e) => Err(anyhow!("Failed to edit section: {}", e))
    }
}

/// Find complete code blocks around the target lines
fn find_complete_code_blocks(lines: &[&str], start_line: usize, end_line: usize) -> (usize, usize) {
    let mut block_start = start_line;
    let mut block_end = end_line;
    
    // Look backwards for block openings
    for i in (0..start_line).rev() {
        let line = lines[i].trim();
        
        // Check for block starts (functions, if statements, objects, etc.)
        if line.contains("if (") && line.contains("{") ||
           line.contains("function") && line.contains("{") ||
           line.contains("const ") && line.contains("= {") ||
           line.starts_with("//") && (line.contains("===") || line.len() > 50) {
            // This might be the start of our block
            block_start = i;
            break;
        }
        
        // Stop if we hit another major block
        if line.ends_with("};") || line == "}" {
            block_start = i + 1;
            break;
        }
    }
    
    // Look forwards for block endings
    for i in end_line..lines.len() {
        let line = lines[i].trim();
        
        // Check for block ends
        if line == "}" || line.ends_with("};") || line == "</script>" {
            block_end = i;
            break;
        }
        
        // Stop if we hit another major block start
        if line.contains("if (") && line.contains("{") ||
           line.contains("function") && line.contains("{") ||
           line.starts_with("//") && (line.contains("===") || line.len() > 50) {
            block_end = i - 1;
            break;
        }
    }
    
    (block_start, block_end)
}

/// Create an intelligent edit prompt based on the type of removal requested
fn create_intelligent_edit_prompt(query: &str, content: &str, start_line: usize, end_line: usize, focus_start: usize, focus_end: usize) -> String {
    let query_lower = query.to_lowercase();
    
    // Determine the type of removal needed
    let removal_type = if query_lower.contains("android") && query_lower.contains("debug") {
        "Android debugging code"
    } else if query_lower.contains("ios") && query_lower.contains("debug") {
        "iOS debugging code"
    } else if query_lower.contains("console") {
        "console logging statements"
    } else if query_lower.contains("debug") {
        "debugging code"
    } else {
        "the specified code"
    };
    
    format!(
        "IMPORTANT: You are removing {} from this JavaScript/HTML file.\n\n\
        TASK: Remove ALL related code completely - not just individual lines.\n\
        This includes:\n\
        - Complete if/else blocks\n\
        - Entire function definitions\n\
        - Full object definitions\n\
        - Associated comments and dividers\n\
        - Any incomplete fragments\n\n\
        FOCUS AREA: Lines {}-{} contain the main target code.\n\
        CONTEXT: Lines {}-{} (you must return this entire section, edited)\n\n\
        CRITICAL: \n\
        - Do NOT leave partial code fragments\n\
        - Do NOT leave orphaned variables or references\n\
        - Do NOT break JavaScript syntax\n\
        - Remove complete logical blocks, not individual scattered lines\n\n\
        Original request: {}\n\n\
        Code section to edit:\n{}\n\n\
        Provide the complete edited section with {} fully removed:",
        removal_type, focus_start, focus_end, start_line, end_line, query, content, removal_type
    )
}

/// Process matches using PARALLEL DIFF COLLECTION (O(log n) instead of O(n²))
async fn process_matches_incrementally(content: &str, query: &str, matches: &[(usize, usize, String)], config: &Config) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    
    // Group matches into logical chunks (Claude Code style)
    let chunks = create_processing_chunks(matches, &lines);
    
    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Collecting {} diff patches in parallel", chunks.len()));
    
    // REVOLUTIONARY: Collect ALL diff patches first, then apply them in reverse order
    let mut diff_patches = Vec::new();
    
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if chunk.matches.is_empty() {
            continue;
        }
        
        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Getting diff {}/{}", chunk_idx + 1, chunks.len()));
        
        // Get diff patch for this chunk (using original content for all chunks)
        match get_diff_patch_for_chunk(content, query, chunk, config).await {
            Ok(patch) => {
                if !patch.is_empty() && patch != "NO_CHANGES_NEEDED" {
                    diff_patches.push((chunk.start_line, patch));
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Diff {}/{} collected", chunk_idx + 1, chunks.len()));
                } else {
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Diff {}/{} - no changes", chunk_idx + 1, chunks.len()));
                }
            },
            Err(e) => {
                crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Diff {}/{} failed: {}", chunk_idx + 1, chunks.len(), e));
            }
        }
    }
    
    // Apply all patches in REVERSE order (bottom-up) to maintain line numbers
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Applying patches bottom-up");
    diff_patches.sort_by(|a, b| b.0.cmp(&a.0)); // Sort by line number, descending
    
    let mut final_content = content.to_string();
    for (base_line, patch) in diff_patches {
        final_content = apply_diff_patch_to_content(&final_content, &patch, base_line)?;
    }
    
    Ok(final_content)
}

/// Get a diff patch for a specific chunk
async fn get_diff_patch_for_chunk(content: &str, query: &str, chunk: &ProcessingChunk, config: &Config) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    let section_content = lines[chunk.start_line..chunk.end_line].join("\n");
    
    let diff_prompt = format!(
        "Analyze this code section and provide ONLY a unified diff patch to remove {}:\n\
        \nCode section (lines {}-{}):\n{}\n\n\
        CRITICAL: Return ONLY the unified diff patch in this exact format:\n\
        @@ -start,count +start,count @@\n\
        -removed line\n\
        +added line\n\
        \n\
        If no changes needed, return: NO_CHANGES_NEEDED",
        query, chunk.start_line + 1, chunk.end_line + 1, section_content
    );
    
    crate::gemini::query_gemini(&diff_prompt, config).await
}

#[derive(Debug)]
struct ProcessingChunk {
    start_line: usize,
    end_line: usize,
    matches: Vec<(usize, usize, String)>,
    content_preview: String,
}

/// Create manageable processing chunks from matches
fn create_processing_chunks(matches: &[(usize, usize, String)], lines: &[&str]) -> Vec<ProcessingChunk> {
    let mut chunks = Vec::new();
    let chunk_size = 300; // Much bigger chunks - more efficient
    
    // Sort matches by line number
    let mut sorted_matches = matches.to_vec();
    sorted_matches.sort_by_key(|m| m.1);
    
    let mut current_chunk_matches = Vec::new();
    let mut chunk_start = if !sorted_matches.is_empty() { sorted_matches[0].1 } else { 0 };
    
    for match_item in sorted_matches {
        let line_idx = match_item.1;
        
        // If this match is too far from our current chunk, start a new chunk
        if line_idx > chunk_start + chunk_size {
            // Finish current chunk
            if !current_chunk_matches.is_empty() {
                let chunk_end = current_chunk_matches.iter().map(|m: &(usize, usize, String)| m.1).max().unwrap_or(chunk_start);
                let preview = create_chunk_preview(lines, chunk_start, chunk_end);
                
                chunks.push(ProcessingChunk {
                    start_line: chunk_start,
                    end_line: chunk_end,
                    matches: current_chunk_matches.clone(),
                    content_preview: preview,
                });
            }
            
            // Start new chunk
            current_chunk_matches.clear();
            chunk_start = line_idx;
        }
        
        current_chunk_matches.push(match_item);
    }
    
    // Add final chunk
    if !current_chunk_matches.is_empty() {
        let chunk_end = current_chunk_matches.iter().map(|m: &(usize, usize, String)| m.1).max().unwrap_or(chunk_start);
        let preview = create_chunk_preview(lines, chunk_start, chunk_end);
        
        chunks.push(ProcessingChunk {
            start_line: chunk_start,
            end_line: chunk_end,
            matches: current_chunk_matches,
            content_preview: preview,
        });
    }
    
    chunks
}

/// Create a preview of the chunk content for processing
fn create_chunk_preview(lines: &[&str], start_line: usize, end_line: usize) -> String {
    let context_size = 5;
    let actual_start = if start_line > context_size { start_line - context_size } else { 0 };
    let actual_end = std::cmp::min(end_line + context_size, lines.len());
    
    lines[actual_start..actual_end].join("\n")
}

/// Process a single chunk of matches
async fn process_single_chunk(content: &str, query: &str, chunk: &ProcessingChunk, config: &Config) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    
    // Find the complete blocks for this chunk
    let (block_start, block_end) = find_complete_code_blocks(&lines, chunk.start_line, chunk.end_line);
    
    // Create a focused section to edit
    let context_size = 8;
    let actual_start = if block_start > context_size { block_start - context_size } else { 0 };
    let actual_end = std::cmp::min(block_end + context_size, lines.len());
    
    // Keep the section manageable - if it's too big, focus on the core area
    let section_size = actual_end - actual_start;
    if section_size > 100 {
        // Too big - focus on just the immediate area around matches
        let focus_start = chunk.start_line.saturating_sub(10);
        let focus_end = std::cmp::min(chunk.end_line + 10, lines.len());
        
        return process_focused_section(content, query, focus_start, focus_end, config).await;
    }
    
    process_focused_section(content, query, actual_start, actual_end, config).await
}

/// Process a focused section using DIFF PATCHES for speed
async fn process_focused_section(content: &str, query: &str, start_line: usize, end_line: usize, config: &Config) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    let section_content = lines[start_line..end_line].join("\n");
    
    // REVOLUTIONARY: Ask for diff patch instead of full content
    let diff_prompt = format!(
        "Analyze this code section and provide ONLY a unified diff patch to remove {}:\n\
        \nCode section (lines {}-{}):\n{}\n\n\
        CRITICAL: Return ONLY the unified diff patch in this exact format:\n\
        @@ -start,count +start,count @@\n\
        -removed line\n\
        +added line\n\
        \n\
        If no changes needed, return: NO_CHANGES_NEEDED",
        query, start_line + 1, end_line + 1, section_content
    );
    
    match crate::gemini::query_gemini(&diff_prompt, config).await {
        Ok(diff_response) => {
            let diff_response = diff_response.trim();
            
            if diff_response == "NO_CHANGES_NEEDED" {
                return Ok(content.to_string());
            }
            
            // Parse and apply the diff patch (MUCH faster than full content replacement)
            apply_diff_patch_to_content(content, &diff_response, start_line)
        },
        Err(e) => Err(anyhow!("Failed to process section: {}", e))
    }
}

/// Apply a diff patch to content efficiently
fn apply_diff_patch_to_content(content: &str, diff_patch: &str, base_line: usize) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut result_lines = lines.clone();
    
    // Simple diff parser - parse the diff and apply changes
    let diff_lines: Vec<&str> = diff_patch.lines().collect();
    let mut line_offset = 0i32;
    
    for line in diff_lines {
        if line.starts_with("@@") {
            continue; // Skip hunk headers
        } else if line.starts_with("-") {
            // Find and remove the line
            let target_line = &line[1..];
            if let Some(pos) = find_matching_line(&result_lines, target_line, base_line) {
                result_lines.remove(pos);
                line_offset -= 1;
            }
        } else if line.starts_with("+") {
            // Add the line at appropriate position
            let new_line = &line[1..];
            let insert_pos = (base_line as i32 + line_offset).max(0) as usize;
            if insert_pos <= result_lines.len() {
                result_lines.insert(insert_pos, new_line);
                line_offset += 1;
            }
        }
    }
    
    Ok(result_lines.join("\n"))
}

/// Find a matching line in the content
fn find_matching_line(lines: &[&str], target: &str, near_line: usize) -> Option<usize> {
    // First try exact match near the expected location
    let search_start = near_line.saturating_sub(5);
    let search_end = std::cmp::min(near_line + 15, lines.len());
    
    for i in search_start..search_end {
        if lines[i].trim() == target.trim() {
            return Some(i);
        }
    }
    
    // Fallback: broader search
    for (i, line) in lines.iter().enumerate() {
        if line.trim() == target.trim() {
            return Some(i);
        }
    }
    
    None
}

/// Extract actual file content from result messages
fn extract_content_from_result(result: &str, original_content: &str) -> String {
    // If the result looks like just a status message, return original content
    if result.starts_with("COMPLETE") || result.starts_with("STATS") || result.starts_with("No changes") {
        return original_content.to_string();
    }
    
    // Otherwise assume the result IS the content
    result.to_string()
}

/// Estimate the number of changes made between two versions
fn estimate_changes(old_content: &str, new_content: &str) -> usize {
    // First check if content is actually different
    if old_content == new_content {
        return 0;
    }
    
    // Count line changes (additions + removals)
    let old_lines = old_content.lines().count();
    let new_lines = new_content.lines().count();
    let line_changes = if old_lines > new_lines {
        old_lines - new_lines // Lines removed
    } else {
        new_lines - old_lines // Lines added
    };
    
    // If no line changes but content is different, count as 1 change
    if line_changes == 0 {
        1
    } else {
        line_changes
    }
}

/// High-performance large file editor using a Rope data structure.
/// This avoids panics and O(n^2) complexity by not re-parsing the file after each edit.
async fn process_large_file_with_rope(file_path_str: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    // 1. Load the content into a Rope once.
    let mut rope = Rope::from_str(content);
    
    let search_keywords = extract_search_keywords_from_query(query);
    if search_keywords.is_empty() {
        return Err(anyhow!("Could not determine what to search for in query: '{}'", query));
    }

    // 2. Find all matches and their character offsets from the original rope.
    let mut matches_with_char_offsets = Vec::new();
    for (line_idx, line) in rope.lines().enumerate() {
        let line_content = line.to_string();
        for keyword in &search_keywords {
            if line_content.to_lowercase().contains(&keyword.to_lowercase()) {
                let line_start_char = rope.line_to_char(line_idx);
                matches_with_char_offsets.push((line_start_char, keyword.clone()));
            }
        }
    }

    if matches_with_char_offsets.is_empty() {
        return Ok(content.to_string()); // No changes needed.
    }

    // 3. Define chunks based on line numbers initially, then convert to char offsets.
    let total_lines = rope.len_lines();
    let num_chunks = 4;
    let chunk_size_lines = (total_lines + num_chunks - 1) / num_chunks;
    
    let mut chunks: Vec<(usize, usize)> = Vec::new(); // (start_char, end_char)
    for i in 0..num_chunks {
        let start_line = i * chunk_size_lines;
        if start_line >= total_lines { break; }
        let end_line = ((i + 1) * chunk_size_lines).min(total_lines);
        
        let start_char = rope.line_to_char(start_line);
        let end_char = if end_line < total_lines {
            rope.line_to_char(end_line)
        } else {
            rope.len_chars()
        };
        chunks.push((start_char, end_char));
    }

    // 4. This offset tracks the total change in character count from previous edits.
    let mut cumulative_char_offset: isize = 0;

    for (chunk_idx, chunk_def) in chunks.iter().enumerate() {
        // 5. Apply the cumulative offset to find the chunk's *current* position.
        let current_start_char = (chunk_def.0 as isize + cumulative_char_offset) as usize;
        let mut current_end_char = (chunk_def.1 as isize + cumulative_char_offset) as usize;

        // Ensure the slice range is still valid.
        if current_start_char >= rope.len_chars() {
            continue; 
        }
        current_end_char = current_end_char.min(rope.len_chars());
        if current_start_char >= current_end_char {
            continue;
        }

        // Check if this chunk contains any matches by comparing against original offsets.
        let has_match_in_chunk = matches_with_char_offsets.iter().any(|&(char_offset, _)| {
            char_offset >= chunk_def.0 && char_offset < chunk_def.1
        });

        if !has_match_in_chunk {
            crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Section {}/4: no matches - skipping", chunk_idx + 1));
            continue;
        }
        
        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Section {}/4: processing...", chunk_idx + 1));

        let chunk_slice = rope.slice(current_start_char..current_end_char);
        let chunk_content = chunk_slice.to_string();

        let edit_prompt = format!(
            "You are an expert programmer editing a section of a large file: '{}'.\n\
            Your task is to: {}.\n\n\
            IMPORTANT INSTRUCTIONS:\n\
            - Write clean, readable, and high-quality code.\n\
            - Follow the existing code style and patterns.\n\
            - Ensure the edited code is complete, functional, and syntactically correct.\n\
            - Keep all non-target functionality in the section intact.\n\n\
            SECTION TO EDIT:\n```\n{}\n```\n\n\
            Return ONLY the complete, edited section of the code. Do not add explanations or markdown formatting.",
            file_path_str, query, chunk_content
        );

        let edited_section = match crate::gemini::query_gemini(&edit_prompt, config).await {
            Ok(edited) => edited,
            Err(e) => {
                crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Section {}/4 failed: {}", chunk_idx + 1, e));
                continue; // Skip this chunk on error.
            }
        };

        let original_len = current_end_char - current_start_char;
        let new_len = edited_section.chars().count();

        // 6. Perform the edit on the rope. This is very fast!
        rope.remove(current_start_char..current_end_char);
        rope.insert(current_start_char, &edited_section);

        // 7. Calculate the delta from this specific edit and update the cumulative offset.
        let delta = new_len as isize - original_len as isize;
        cumulative_char_offset += delta;
        
        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Section {}/4 processed", chunk_idx + 1));
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    
    Ok(rope.to_string())
}

/// Process a large section of the file (OBSOLETE - replaced by Rope implementation)
async fn process_large_section_legacy(content: &str, query: &str, start_line: usize, end_line: usize, config: &Config) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    let section_content = lines[start_line..end_line].join("\n");
    
    // Use a focused prompt for large section processing
    let edit_prompt = format!(
        "Remove all {} from this large code section.\n\n\
        Instructions:\n\
        - Remove complete blocks, functions, and structures\n\
        - Maintain valid JavaScript/HTML syntax\n\
        - Keep all non-target functionality intact\n\n\
        Section to process (lines {}-{}):\n{}\n\n\
        Return the complete section with all {} removed:",
        query, start_line + 1, end_line + 1, section_content, query
    );
    
    match crate::gemini::query_gemini(&edit_prompt, config).await {
        Ok(edited_section) => {
            // Reconstruct the full file
            let before_content = if start_line > 0 {
                lines[..start_line].join("\n") + "\n"
            } else {
                String::new()
            };
            
            let after_content = if end_line < lines.len() {
                "\n".to_string() + &lines[end_line..].join("\n")
            } else {
                String::new()
            };
            
            Ok(format!("{}{}{}", before_content, edited_section, after_content))
        },
        Err(e) => Err(anyhow!("Failed to process large section: {}", e)),
    }
}

/// Edit multiple small sections
async fn edit_multiple_small_sections(content: &str, sections: &[EditableSection], query: &str, config: &Config) -> Result<String> {
    // For now, just edit the most relevant section
    // TODO: Implement true multi-section editing
    let main_section = choose_main_target_section(sections, query);
    edit_targeted_section(content, &main_section, query, config).await
}

async fn execute_fetch_url(function_call: &FunctionCall) -> Result<String> {
    let url_str = function_call.arguments.get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing url parameter"))?;
    
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .build()?;
    
    let response = client.get(url_str).send().await?;
    let content = response.text().await?;
    
    // Convert HTML to text if it looks like HTML
    let text_content = if content.trim_start().starts_with('<') {
        html2text::from_read(content.as_bytes(), 80)
    } else {
        content
    };
    
    Ok(format!("Content from '{}':\n\n{}", url_str, text_content))
}

// Audio functions removed

async fn execute_diff_files(function_call: &FunctionCall) -> Result<String> {
    let file1_path = function_call.arguments.get("file1_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file1_path parameter"))?;
    
    let file2_path = function_call.arguments.get("file2_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file2_path parameter"))?;
    
    let content1 = read_file(Path::new(file1_path)).await?;
    let content2 = read_file(Path::new(file2_path)).await?;
    
    let diff = create_diff(&content1, &content2, file1_path, file2_path)?;
    Ok(format!("Differences between '{}' and '{}':\n\n{}", file1_path, file2_path, diff))
}

async fn execute_code_analysis(
    function_call: &FunctionCall,
    config: &Config,
    conversation_history: &str
) -> Result<String> {
    let file_path = function_call.arguments.get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file_path parameter"))?;
    
    // Check if structured analysis is requested
    let query = function_call.arguments.get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
    if query.contains("json") || query.contains("structured") || query.contains("detailed") {
        return execute_structured_code_analysis(file_path, query, config, conversation_history).await;
    }
    
    let content = read_file(Path::new(file_path)).await?;
    let file_extension = Path::new(file_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    // Create focused analysis prompt based on user's request context
    let user_request = conversation_history.lines().last().unwrap_or("analyze the code");
    let analysis_prompt = format!(
        "USER REQUEST: {}\n\n\
        Quickly analyze this {} file '{}' to identify what's wrong or missing for the user's request.\n\
        Focus ONLY on the specific issue mentioned. Provide a brief, targeted analysis:\n\
        - What is the specific problem?\n\
        - What code is missing or broken?\n\
        - What needs to be added/fixed?\n\n\
        Keep it concise and actionable.\n\n\
        Code content:\n{}",
        user_request, file_extension, file_path, content
    );
    
    // Use shorter timeout for focused analysis (15 seconds max)
    let mut fast_config = config.clone();
    fast_config.gemini.timeout_seconds = 45;
    let analysis = gemini::query_gemini(&analysis_prompt, &fast_config).await?;
    Ok(format!("Code analysis for '{}':\n\n{}", file_path, analysis))
}

async fn execute_structured_code_analysis(
    file_path: &str,
    query: &str,
    config: &Config,
    conversation_history: &str
) -> Result<String> {
    let content = read_file(Path::new(file_path)).await?;
    let file_extension = Path::new(file_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    let structured_prompt = format!(
        "Analyze this {} file and return a JSON response with structured analysis.\n\n\
        USER REQUEST: {}\n\
        CONVERSATION CONTEXT: {}\n\n\
        Please analyze the code and return ONLY a valid JSON object with this structure:\n\
        {{\n\
          \"file_info\": {{\n\
            \"path\": \"{}\",\n\
            \"language\": \"{}\",\n\
            \"lines_of_code\": <number>,\n\
            \"size_bytes\": <number>\n\
          }},\n\
          \"code_quality\": {{\n\
            \"complexity_score\": <1-10>,\n\
            \"maintainability_score\": <1-10>,\n\
            \"readability_score\": <1-10>,\n\
            \"test_coverage_estimate\": <0-100>\n\
          }},\n\
          \"issues_found\": [\n\
            {{\n\
              \"severity\": \"high|medium|low\",\n\
              \"type\": \"bug|performance|security|style\",\n\
              \"description\": \"Brief description\",\n\
              \"location\": \"line 123 or function name\",\n\
              \"suggestion\": \"How to fix\"\n\
            }}\n\
          ],\n\
          \"dependencies\": [\n\
            {{\n\
              \"name\": \"library/module name\",\n\
              \"type\": \"external|internal|builtin\",\n\
              \"usage\": \"how it's used\"\n\
            }}\n\
          ],\n\
          \"functions\": [\n\
            {{\n\
              \"name\": \"function_name\",\n\
              \"complexity\": <1-10>,\n\
              \"parameters\": <number>,\n\
              \"return_type\": \"type or description\",\n\
              \"purpose\": \"brief description\"\n\
            }}\n\
          ],\n\
          \"recommendations\": [\n\
            {{\n\
              \"priority\": \"high|medium|low\",\n\
              \"category\": \"performance|security|maintainability|features\",\n\
              \"description\": \"What to improve\",\n\
              \"impact\": \"Expected benefit\"\n\
            }}\n\
          ],\n\
          \"summary\": {{\n\
            \"overall_score\": <1-10>,\n\
            \"main_purpose\": \"What this code does\",\n\
            \"key_strengths\": [\"strength1\", \"strength2\"],\n\
            \"critical_issues\": [\"issue1\", \"issue2\"],\n\
            \"next_steps\": [\"action1\", \"action2\"]\n\
          }}\n\
        }}\n\n\
        Code to analyze:\n{}",
        file_extension, query, conversation_history, file_path, file_extension, content
    );
    
    let analysis = crate::gemini::query_gemini(&structured_prompt, config).await?;
    
    // Try to validate and format the JSON response
    match serde_json::from_str::<serde_json::Value>(&analysis) {
        Ok(json) => Ok(serde_json::to_string_pretty(&json).unwrap_or(analysis)),
        Err(_) => {
            // If not valid JSON, try to extract JSON from the response
            if let Some(start) = analysis.find('{') {
                if let Some(end) = analysis.rfind('}') {
                    let json_str = &analysis[start..=end];
                    match serde_json::from_str::<serde_json::Value>(json_str) {
                        Ok(json) => Ok(serde_json::to_string_pretty(&json).unwrap_or(analysis)),
                        Err(_) => Ok(format!("Structured Code Analysis for '{}':\n\n{}", file_path, analysis))
                    }
                } else {
                    Ok(format!("Structured Code Analysis for '{}':\n\n{}", file_path, analysis))
                }
            } else {
                Ok(format!("Structured Code Analysis for '{}':\n\n{}", file_path, analysis))
            }
        }
    }
}

/// Execute code editing with intelligent recovery and fallback strategies
async fn execute_edit_code_with_recovery(
    function_call: &FunctionCall,
    config: &Config,
    conversation_history: &str
) -> Result<String> {
    // Attempt the primary, high-quality edit function first.
    match execute_edit_code_with_path_and_query(function_call, config, conversation_history).await {
        Ok(result) => Ok(result),
        Err(e) => {
            // The primary edit failed. Log the failure and attempt a simplified recovery.
            crate::thinking_display::PersistentStatusBar::set_ai_thinking("Primary edit failed. Attempting simplified recovery...");

            // Use the error message from the failed attempt to inform the recovery.
            let failure_reason = e.to_string();
            execute_edit_with_simplified_prompt(function_call, config, &failure_reason).await
        }
    }
}

/// Create new file based on description when edit_code is called on non-existent file
async fn execute_create_new_file_from_description(
    file_path: &str,
    description: &str,
    config: &Config,
    conversation_history: &str
) -> Result<String> {
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Creating new file based on description");
    
    let create_prompt = format!(
        "You are an expert programmer. Create a high-quality, production-ready file: '{}'

        Requirements: {}
        
        Context: {}
        
        IMPORTANT GUIDELINES:
        - Write clean, well-structured, professional code
        - Use modern best practices and patterns  
        - Include proper error handling where appropriate
        - Add meaningful comments for complex logic
        - Ensure code is performant and maintainable
        - Follow language-specific conventions
        
        Return ONLY the complete file content with NO markdown formatting, explanations, or code blocks.",
        file_path, description, conversation_history
    );
    
    let content = crate::gemini::query_gemini(&create_prompt, config).await?;
    let validated_content = validate_and_extract_file_content(&content, "", file_path, description)?;
    
    write_file(std::path::Path::new(file_path), &validated_content).await?;
    Ok(format!("Created new file '{}' with {} characters", file_path, validated_content.len()))
}

/// Simplified edit approach for recovery.
async fn execute_edit_with_simplified_prompt(
    function_call: &FunctionCall,
    config: &Config,
    failure_reason: &str // Now accepts the failure reason
) -> Result<String> {
    let file_path = function_call.arguments.get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file_path parameter"))?;
    
    let query = function_call.arguments.get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing query parameter"))?;
    
    // IMPORTANT: Re-read the original content to avoid editing a broken file.
    let current_content = read_file(std::path::Path::new(file_path)).await?;
    
    // Create a more informative prompt for the recovery attempt.
    let simple_prompt = format!(
        "A previous attempt to edit the file '{}' failed.
Original Request: {}
Previous Error: {}

Please try again to edit the file with the original request. Return only the complete, updated file content.

Current content:
{}",
        file_path, query, failure_reason, current_content
    );
    
    let response = crate::gemini::query_gemini(&simple_prompt, config).await?;
    let new_content = validate_and_extract_file_content(&response, &current_content, file_path, query)?;
    
    // Don't create diff during edit process - it interferes with editing
    // Just do a simple change summary
    let diff_info = if new_content != current_content {
        let original_lines = current_content.lines().count();
        let new_lines = new_content.lines().count();
        let lines_added = new_lines.saturating_sub(original_lines);
        let lines_removed = original_lines.saturating_sub(new_lines);
        format!("\nChanges: +{} lines, -{} lines", lines_added, lines_removed)
    } else {
        "".to_string()
    };
    
    write_file(std::path::Path::new(file_path), &new_content).await?;
    Ok(format!("Updated '{}' using simplified approach{}", file_path, diff_info))
}

/// Analyze edit requirements and choose optimal strategy
async fn analyze_edit_strategy(query: &str, content: &str, _file_path: &str) -> Result<EditStrategy> {
    let file_size = content.len();
    let query_words = query.split_whitespace().count();
    let query_lower = query.to_lowercase();
    
    // Check for surgical edit indicators
    let _is_targeted_change = query_words < 15 && (
        query_lower.contains("rename") ||
        query_lower.contains("change") && (
            query_lower.contains("title") ||
            query_lower.contains("name") ||
            query_lower.contains("color") ||
            query_lower.contains("text")
        ) ||
        query_lower.contains("fix") && query_words < 8 ||
        query_lower.contains("add") && query_words < 10
    );
    
    // Small, targeted changes should use surgical editing
    if _is_targeted_change && file_size < 100_000 { // 100KB limit for surgical
        return Ok(EditStrategy::Surgical);
    }
    
    // Large files need chunked approach
    if file_size > 50_000 { // 50KB threshold
        return Ok(EditStrategy::Chunked);
    }
    
    // Default to direct editing for medium-sized files with complex changes
    Ok(EditStrategy::Direct)
}

/// Execute surgical editing using imara-diff for precise change injection
async fn execute_surgical_edit(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Using Surgical Editor with imara-diff");
    
    // Use fast heuristics to identify the target area (no AI call needed)
    let target_area = identify_target_area_heuristically(content, query)?;
    
    // Extract only the relevant section for AI processing
    let lines: Vec<&str> = content.lines().collect();
    let context_start = target_area.start.saturating_sub(5); // 5 lines of context
    let context_end = (target_area.end + 5).min(lines.len());
    let context_lines = &lines[context_start..context_end];
    let context_content = context_lines.join("\n");
    
    // Create focused prompt for surgical edit
    let focused_prompt = format!(
        "SURGICAL EDIT: Make only the specific change requested in this code section.

TARGET CHANGE: {}
SECTION TO EDIT (lines {}-{}):
{}

Return ONLY the modified section, keeping the same line structure.",
        query, context_start + 1, context_end, context_content
    );
    
    // Get AI response for just the section
    let response = crate::gemini::query_gemini(&focused_prompt, config).await?;
    let modified_section = validate_and_extract_file_content(&response, &context_content, file_path, query)?;
    
    // Use imara-diff to inject precise changes
    inject_changes_precisely(content, &modified_section, context_start, context_end)
}

/// Use AI reasoning to identify target area instead of hardcoded patterns
async fn identify_target_area_intelligently(content: &str, query: &str, config: &Config) -> Result<std::ops::Range<usize>> {
    let lines: Vec<&str> = content.lines().collect();
    
    // Use AI to analyze the code structure and locate the target
    let analysis_prompt = format!(
        r#"LOCATE TARGET: Analyze this code and identify the exact line numbers that need to be modified for this request.

REQUEST: {}

CODE WITH LINE NUMBERS:
{}

Respond with ONLY the line numbers in format: START_LINE-END_LINE (e.g., "15-18" for lines 15 to 18)
If the change affects a single line, use same number twice (e.g., "6-6")."#,
        query,
        lines.iter().enumerate()
            .map(|(i, line)| format!("{:3}: {}", i + 1, line))
            .take(50) // Limit to first 50 lines for analysis
            .collect::<Vec<_>>()
            .join("\n")
    );
    
    // Try AI analysis with error handling
    match crate::gemini::query_gemini(&analysis_prompt, config).await {
        Ok(response) => {
            let response = response.trim();
            
            // Try to extract line numbers from various response formats
            let line_numbers = extract_line_numbers_from_response(response);
            if let Some((start, end)) = line_numbers {
                let start_idx = start.saturating_sub(1).min(lines.len().saturating_sub(1));
                let end_idx = end.min(lines.len());
                if start_idx < end_idx {
                    return Ok(start_idx..end_idx);
                }
            }
        },
        Err(_) => {
            // AI query failed, continue to heuristic fallback
        }
    }
    
    // Fallback: if AI analysis fails, use intelligent heuristics
    identify_target_area_heuristic(lines, query)
}

/// Fast heuristic target identification (no AI calls)
fn identify_target_area_heuristically(content: &str, query: &str) -> Result<std::ops::Range<usize>> {
    let lines: Vec<&str> = content.lines().collect();
    let query_lower = query.to_lowercase();
    
    // For button/event functionality
    if query_lower.contains("button") || query_lower.contains("event") || query_lower.contains("scramble") {
        // Look for event listener section or end of DOMContentLoaded
        for (i, line) in lines.iter().enumerate() {
            if line.contains("addEventListener") || line.contains("click") {
                return Ok(i.saturating_sub(2)..lines.len().min(i + 10));
            }
        }
        // If no event listeners found, target end of main function
        for (i, line) in lines.iter().enumerate().rev() {
            if line.contains("});") && i > lines.len() / 2 {
                return Ok(i.saturating_sub(5)..i + 1);
            }
        }
    }
    
    // For function additions, target the end of file
    if query_lower.contains("function") || query_lower.contains("add") {
        let end = lines.len();
        return Ok(end.saturating_sub(10)..end);
    }
    
    // Default: target middle section
    let start = lines.len() / 3;
    let end = lines.len() * 2 / 3;
    Ok(start..end)
}

/// Extract line numbers from AI response in various formats
fn extract_line_numbers_from_response(response: &str) -> Option<(usize, usize)> {
    // Format 1: "15-18"
    if let Some(dash_pos) = response.find('-') {
        let start_str = response[..dash_pos].trim();
        let end_str = response[dash_pos + 1..].trim();
        
        if let (Ok(start), Ok(end)) = (start_str.parse::<usize>(), end_str.parse::<usize>()) {
            return Some((start, end));
        }
    }
    
    // Format 2: "line 15 to 18" or "lines 15-18"
    if let Ok(regex) = regex::Regex::new(r"line[s]?\s+(\d+)(?:\s+to\s+|\s*-\s*)(\d+)") {
        if let Some(captures) = regex.captures(response) {
            if let (Some(start), Some(end)) = (captures.get(1), captures.get(2)) {
                if let (Ok(start_num), Ok(end_num)) = (start.as_str().parse::<usize>(), end.as_str().parse::<usize>()) {
                    return Some((start_num, end_num));
                }
            }
        }
    }
    
    // Format 3: Just numbers "15 18" or "15,18"
    let numbers: Vec<usize> = response.split(|c: char| !c.is_numeric())
        .filter_map(|s| s.parse().ok())
        .take(2)
        .collect();
    
    if numbers.len() >= 2 {
        return Some((numbers[0], numbers[1]));
    } else if numbers.len() == 1 {
        return Some((numbers[0], numbers[0]));
    }
    
    None
}

/// Intelligent heuristic-based target identification as fallback
fn identify_target_area_heuristic(lines: Vec<&str>, query: &str) -> Result<std::ops::Range<usize>> {
    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    let mut best_score = 0.0;
    let mut best_range = 0..10; // Default small range
    
    // Score each potential section based on relevance
    for start in 0..lines.len() {
        for window_size in [1, 3, 5, 10] {
            let end = (start + window_size).min(lines.len());
            let section_text = lines[start..end].join(" ").to_lowercase();
            
            // Calculate relevance score
            let mut score = 0.0;
            for word in &query_words {
                if section_text.contains(word) {
                    score += word.len() as f64; // Longer words = more specific
                }
            }
            
            // Prefer smaller, more focused sections
            score = score / (window_size as f64).sqrt();
            
            if score > best_score {
                best_score = score;
                best_range = start..end;
            }
        }
    }
    
    // If no good match found, default to reasonable section
    if best_score < 1.0 {
        let default_end = (lines.len() / 5).max(5).min(lines.len());
        best_range = 0..default_end;
    }
    
    Ok(best_range)
}

/// Inject changes precisely using rope-like operations
fn inject_changes_precisely(original: &str, modified_section: &str, start_line: usize, end_line: usize) -> Result<String> {
    let mut lines: Vec<&str> = original.lines().collect();
    let new_lines: Vec<&str> = modified_section.lines().collect();
    
    // Replace the target section
    lines.splice(start_line..end_line, new_lines.iter().cloned());
    
    Ok(lines.join("\n"))
}

/// Execute editing on large files using FAST EDITOR SYSTEM
async fn execute_large_file_edit(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Using Fast Editor System");
    
    // Use the proper intelligent matching system 
    match edit_file_in_chunks(content, query, config).await {
        Ok(mut result) => {
            // Extract the actual content from the result message using proper validation
            let final_content = validate_and_extract_file_content(&result, content, file_path, query)?;
            
            // Don't show diff during edit - it interferes with the process
            // Just track that changes were made
            if final_content != content {
                let original_lines = content.lines().count();
                let new_lines = final_content.lines().count();
                let lines_added = new_lines.saturating_sub(original_lines);
                let lines_removed = original_lines.saturating_sub(new_lines);
                result = format!("{}\nChanges: +{} lines, -{} lines", result, lines_added, lines_removed);
            }
            
            write_file(std::path::Path::new(file_path), &final_content).await?;
            Ok(result)
        },
        Err(e) => Err(e)
    }
}

#[derive(Debug)]
enum EditSection {
    Start,
    End,
    Middle(usize, usize),
}

/// Edit a specific section of a large file
async fn edit_file_section(content: &str, query: &str, config: &Config, section: EditSection) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    
    let (start_idx, end_idx, context_before, context_after) = match section {
        EditSection::Start => {
            (0, 500.min(lines.len()), String::new(), 
             format!("... (file continues for {} more lines)", lines.len().saturating_sub(500)))
        },
        EditSection::End => {
            let start = lines.len().saturating_sub(500);
            (start, lines.len(), 
             format!("(file starts with {} lines) ...", start),
             String::new())
        },
        EditSection::Middle(start, end) => {
            (start, end, 
             format!("(previous {} lines) ...", start),
             format!("... ({} more lines)", lines.len().saturating_sub(end)))
        }
    };
    
    let section_content = lines[start_idx..end_idx].join("\n");
    
    let edit_prompt = format!(
        "Edit this section of a large file:\n\
        Context: {}\n\n\
        SECTION TO EDIT:\n{}\n\n\
        Context: {}\n\n\
        Task: {}\n\n\
        Return only the edited section (not the full file):",
        context_before, section_content, context_after, query
    );
    
    let response = crate::gemini::query_gemini(&edit_prompt, config).await?;
    
    // Reconstruct the full file with the edited section
    let mut result_lines = lines.to_vec();
    let new_section_lines: Vec<&str> = response.lines().collect();
    
    // Replace the section
    result_lines.splice(start_idx..end_idx, new_section_lines.iter().cloned());
    
    Ok(result_lines.join("\n"))
}

/// FAST SINGLE-CALL editor - like Claude Code
pub async fn edit_file_in_chunks(content: &str, query: &str, config: &Config) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    
    // UNIVERSAL APPROACH: Process ANY file with diff patches to avoid MAX_TOKENS
    if total_lines > 100 {
        // The diff patch approach is too restrictive and can lead to low-quality code.
        // Instead, we will use the more robust rope-based editor which handles large files
        // and complex edits better by processing chunks and reassembling the file.
        return process_large_file_with_rope(
            "", // file_path is not available here, but the rope editor can handle it
            content,
            query,
            config
        ).await;
    }
    
    // Fallback: process entire file in chunks (not recommended for very large files)
    let chunk_size = 100; // Smaller chunks for fallback
    let mut processed_content = String::new();
    
    for (chunk_index, chunk_start) in (0..lines.len()).step_by(chunk_size).enumerate() {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, lines.len());
        let chunk_lines = &lines[chunk_start..chunk_end];
        let chunk_content = chunk_lines.join("\n");
        
        let chunk_prompt = format!(
            "You are an expert programmer. Your task is to edit this chunk of code based on the user's query.\n\n\
            User Query: {}\n\n\
            Code Chunk (chunk {} of {}, lines {}-{}):\n```\n{}\n```\n\n\
            IMPORTANT:\n\
            - Return ONLY the complete, edited content for this specific chunk.\n\
            - Ensure the code is syntactically correct and follows best practices.\n\
            - Do not add explanations or markdown formatting.",
            chunk_index + 1,
            (lines.len() + chunk_size - 1) / chunk_size,
            chunk_start + 1,
            chunk_end,
            query,
            chunk_content
        );
        
        match crate::gemini::query_gemini(&chunk_prompt, config).await {
            Ok(edited_chunk) => {
                if !processed_content.is_empty() {
                    processed_content.push('\n');
                }
                processed_content.push_str(&edited_chunk);
            },
            Err(e) => {
                return Err(anyhow!("Failed to edit chunk {} (lines {}-{}): {}", 
                    chunk_index + 1, chunk_start + 1, chunk_end, e));
            }
        }
    }
    
    Ok(processed_content)
}

/// Generates an edited version of the code for small files using a direct prompt.
async fn generate_direct_edit(content: &str, query: &str, file_path: &str, config: &Config) -> Result<String> {
    let edit_prompt = format!(
        "You are an expert programmer. Modify this code to fulfill the request.\n\n\
        Request: {}\n\n\
        Current code:\n{}\n\n\
        Requirements:\n\
        - Write clean, readable code\n\
        - Follow existing code style and patterns\n\
        - Add appropriate comments for new functionality\n\
        - Ensure the code works correctly\n\n\
        Return ONLY the complete updated file content with no markdown formatting.",
        query, content
    );

    let start_time = std::time::Instant::now();
    
    // Use function calling version to get token usage
    let (response, _function_call, token_usage) = crate::gemini::query_gemini_with_function_calling(&edit_prompt, config, None).await?;
    
    let response_time_ms = start_time.elapsed().as_millis() as u32;
    
    // Update status with token information if available
    if let Some(usage) = token_usage {
        crate::thinking_display::PersistentStatusBar::update_status_with_tokens_and_timing(
            "Editing code", 
            usage.input_tokens, 
            usage.output_tokens,
            response_time_ms
        );
    }
    
    validate_and_extract_file_content(&response, content, file_path, query)
}

/// Simple, reliable edit function that doesn't use complex diff logic
async fn execute_simple_edit(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Using simple edit strategy");
    
    let simple_prompt = format!(
        "Edit this file to: {}\n\nFile: {}\n\n{}\n\n\
        Return ONLY the complete updated file content.",
        query, file_path, content
    );

    let response = crate::gemini::query_gemini(&simple_prompt, config).await?;
    
    // Simple validation and extraction
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("Empty response from AI"));
    }
    
    // Extract from code blocks if present
    let content = if trimmed.starts_with("```") {
        // Find the content between code fences
        let lines: Vec<&str> = trimmed.lines().collect();
        if lines.len() >= 3 {
            // Skip first and last lines
            lines[1..lines.len()-1].join("\n")
        } else {
            trimmed.to_string()
        }
    } else {
        trimmed.to_string()
    };
    
    Ok(content)
}

/// Execute code editing with intelligent path resolution and query processing
pub async fn execute_edit_code_with_path_and_query(
    function_call: &FunctionCall,
    config: &Config,
    conversation_history: &str
) -> Result<String> {
    // Enhanced parameter validation with context awareness
    let file_path = function_call.arguments.get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file_path parameter. Available parameters: {:?}. Context: {}", function_call.arguments.keys().collect::<Vec<_>>(), conversation_history.lines().last().unwrap_or("No context")))?;
    
    let query = function_call.arguments.get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing query parameter. Available parameters: {:?}. Context: {}", function_call.arguments.keys().collect::<Vec<_>>(), conversation_history.lines().last().unwrap_or("No context")))?;
    
    // Real-time UI is already shown by the caller
    
    // METACOGNITIVE VALIDATION: Check if query is malformed/truncated
    if is_malformed_query(query) {
        return Err(anyhow!("Query appears to be malformed or truncated. Query: '{}'", query));
    }
    
    // Read current file content
    let current_content = read_file(Path::new(file_path)).await?;
    
    // Use simple heuristic strategy selection (no AI calls)
    let edit_strategy = select_edit_strategy_fast(query, &current_content);
    
    let new_content = match edit_strategy {
        EditStrategy::Surgical => {
            // Use surgical editing for precise, targeted changes
            crate::thinking_display::PersistentStatusBar::set_ai_thinking("Strategy: Surgical editing detected");
            match execute_surgical_edit(file_path, &current_content, query, config).await {
                Ok(content) => content,
                Err(e) => {
                    // Surgical editing failed, try reasoning-based approach
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Surgical editing failed, using reasoning approach");
                    
                    // Log the error through status system, not direct print
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Surgical edit failed: {}", e.to_string().chars().take(80).collect::<String>()));
                    
                    // Use a simpler, more reliable approach
                    match execute_simple_edit(file_path, &current_content, query, config).await {
                        Ok(content) => content,
                        Err(_) => {
                            crate::thinking_display::PersistentStatusBar::set_ai_thinking("All edit methods failed, using basic replacement");
                            // Last resort: basic edit
                            generate_direct_edit(&current_content, query, file_path, config).await?
                        }
                    }
                }
            }
        }
        EditStrategy::Chunked => {
            // For large files, use the robust, memory-efficient Rope-based editor
            crate::thinking_display::PersistentStatusBar::set_ai_thinking("Strategy: Chunked editing for large file");
            process_large_file_with_rope(file_path, &current_content, query, config).await?
        }
        EditStrategy::Direct => {
            // For smaller files, use the direct, single-prompt approach
            crate::thinking_display::PersistentStatusBar::set_ai_thinking("Strategy: Direct editing for medium complexity");
            generate_direct_edit(&current_content, query, file_path, config).await?
        }
    };

    // Simple diff info without breaking functionality
    let _diff_info = if new_content != current_content {
        // Count simple changes instead of complex diff
        let original_lines = current_content.lines().count();
        let new_lines = new_content.lines().count();
        let lines_added = new_lines.saturating_sub(original_lines);
        let lines_removed = original_lines.saturating_sub(new_lines);
        
        if lines_added > 0 || lines_removed > 0 {
            format!("\nChanges: +{} lines, -{} lines", lines_added, lines_removed)
        } else {
            "\nChanges: content modified".to_string()
        }
    } else {
        "".to_string() // No changes, no message
    };

    // Write the final content to the file.
    write_file(Path::new(file_path), &new_content).await?;
    
    // Show completion in modern UI style
    let changes = estimate_changes(&current_content, &new_content);
    let file_display = if file_path.len() > 50 {
        format!("...{}", &file_path[file_path.len() - 47..])
    } else {
        file_path.to_string()
    };
    
    let result = if changes > 0 {
        // UI completion is handled by caller
        format!("Wrote {} bytes to '{}'", new_content.len(), file_display)
    } else {
        "No changes needed".to_string()
    };
    
    Ok(result)
}

/// Metacognitive validation: Detect malformed or truncated queries
fn is_malformed_query(query: &str) -> bool {
    // Check for common signs of truncated/malformed queries
    let indicators = [
        // Ends abruptly without completing sentence
        query.ends_with("1.") || query.ends_with("2.") || query.ends_with("3."),
        // Missing spaces (common in truncated JSON)
        query.contains("theRubik's") || query.contains("fortheRubik"),
        // Ends with incomplete sentence structure
        query.ends_with("cubelets") && !query.contains("?") && !query.contains("."),
        // Very long single-sentence run-on (likely truncated)
        !query.contains(".") && query.len() > 200,
        // Contains formatting artifacts from truncated JSON
        query.contains("\\`") || query.contains("'Add functions") && query.len() < 50,
        // New patterns from recent failures
        query.ends_with("This will") || query.ends_with("This function should:"),
        query.ends_with("scramble.This function should:"),
        // Ends with incomplete list items
        query.contains("1. Define") && query.ends_with("This will"),
        // Missing spaces in compound words (JSON parsing artifacts)
        query.contains("scramble.This") || query.contains("moves.This"),
    ];
    
    indicators.iter().any(|&condition| condition)
}

/// Fast strategy selection using simple heuristics (no AI calls)
fn select_edit_strategy_fast(_query: &str, content: &str) -> EditStrategy {
    let file_size = content.len();
    
    // Use Chunked strategy for large files, otherwise use Direct.
    // Removed fragile Surgical strategy that was causing failures.
    if file_size > 50_000 { // 50KB threshold
        EditStrategy::Chunked
    } else {
        EditStrategy::Direct
    }
}

/// Validate AI response and extract file content, preventing data corruption
fn validate_and_extract_file_content(raw_response: &str, original_content: &str, file_path: &str, query: &str) -> Result<String> {
    let trimmed_response = raw_response.trim();
    
    // Check if this is just a model name or metadata (not actual content)
    if trimmed_response == "gemini-2.5-flash" || 
       trimmed_response.starts_with("gemini-") ||
       trimmed_response.starts_with("claude-") ||
       trimmed_response.starts_with("gpt-") {
        return Err(anyhow::anyhow!(
            "Response parsing error - extracted model name instead of content. File: {}, Response: '{}'", 
            file_path, trimmed_response
        ));
    }
    
    // Check if this looks like a random ID/hash instead of content
    if trimmed_response.len() < 100 && 
       trimmed_response.chars().all(|c| c.is_alphanumeric()) && 
       !trimmed_response.contains(' ') &&
       !trimmed_response.contains('<') {  // Not HTML
        return Err(anyhow::anyhow!(
            "Response parsing error - extracted ID/hash instead of content. File: {}, Response: '{}'", 
            file_path, trimmed_response
        ));
    }
    
    // Check for obvious invalid responses that would corrupt the file (minimal check)
    let invalid_indicators = [
        "I cannot",
        "I don't have access",
        "Error:",
    ];
    
    // Only reject if response is clearly an error message
    if invalid_indicators.iter().any(|indicator| trimmed_response.starts_with(indicator)) {
        return Err(anyhow::anyhow!(
            "Invalid AI response detected - would corrupt file {}. Response: '{}'", 
            file_path, trimmed_response
        ));
    }
    
    // For HTML files, ensure basic structure exists
    if file_path.ends_with(".html") || file_path.ends_with(".htm") {
        if !trimmed_response.contains("<!DOCTYPE") && !trimmed_response.contains("<html") {
            return Err(anyhow::anyhow!(
                "Invalid HTML response - missing basic HTML structure. File: {}", 
                file_path
            ));
        }
    }
    
    // Extract content from code blocks if present - be more aggressive
    let content = if trimmed_response.starts_with("```") {
        // Extract from code block
        let lines: Vec<&str> = trimmed_response.lines().collect();
        if lines.len() < 3 {
            return Err(anyhow::anyhow!("Invalid code block format"));
        }
        
        // Skip first and last lines (```)
        let mut content_lines = Vec::new();
        let mut in_content = false;
        
        for line in lines {
            if line.starts_with("```") {
                if in_content {
                    break; // End of code block
                } else {
                    in_content = true; // Start of code block
                }
            } else if in_content {
                content_lines.push(line);
            }
        }
        content_lines.join("\n")
    } else if trimmed_response.contains("```") {
        // Handle inline code blocks
        let start = trimmed_response.find("```").unwrap();
        let after_start = &trimmed_response[start + 3..];
        if let Some(end) = after_start.find("```") {
            // Find the actual content start (skip language identifier)
            let content_start = after_start.find('\n').map(|i| i + 1).unwrap_or(0);
            after_start[content_start..end].to_string()
        } else {
            trimmed_response.to_string()
        }
    } else {
        trimmed_response.to_string()
    };
    
    // Smart validation for potential data corruption - but allow legitimate edits
    let size_ratio = if original_content.len() > 0 {
        content.len() as f64 / original_content.len() as f64
    } else {
        1.0 // New file creation
    };
    
    let is_removal_task = query.to_lowercase().contains("remove") || 
                         query.to_lowercase().contains("delete") ||
                         query.to_lowercase().contains("clean") ||
                         query.to_lowercase().contains("strip");
    
    // Only flag as corrupted if it's an extreme case AND not a removal task
    let looks_corrupted = (size_ratio < 0.01 && !is_removal_task) || // >99% reduction without explicit removal
                         content.is_empty() && !original_content.is_empty() && !is_removal_task;
    
    if looks_corrupted {
        return Err(anyhow::anyhow!(
            "AI response appears corrupted - extreme size reduction detected. Original: {} chars, New: {} chars (ratio: {:.2}). This may indicate the AI truncated or replaced the file content instead of editing it properly.",
            original_content.len(), content.len(), size_ratio
        ));
    }
    
    Ok(content)
}



// Helper functions for path resolution
pub fn smart_resolve_file_path_for_creation(file_path: &str, _context: Option<&str>) -> String {
    let path = Path::new(file_path);
    
    // If it's already a proper path, use it
    if path.is_absolute() || path.starts_with("./") || path.starts_with("../") {
        return file_path.to_string();
    }
    
    // For relative paths, ensure they're in current directory
    if !file_path.starts_with("./") {
        format!("./{}", file_path)
    } else {
        file_path.to_string()
    }
}

/// Validate edit completion with diff analysis - fast and targeted
async fn validate_edit_completion(function_call: &FunctionCall, result: &str) -> String {
    // Extract file path from the function call
    let file_path = match function_call.arguments.get("file_path").and_then(|v| v.as_str()) {
        Some(path) => path,
        None => return "Edit completed".to_string(), // Fallback if no file path
    };
    
    // Extract the original query/intent
    let query = function_call.arguments.get("query").and_then(|v| v.as_str()).unwrap_or("");
    
    // Check if the result indicates success
    if result.contains("Error:") || result.contains("Failed") {
        return "Edit failed - see details above".to_string();
    }
    
    // Extract diff statistics if available in the result
    let lines_added = count_occurrences(result, "added") + count_occurrences(result, "inserted");
    let lines_removed = count_occurrences(result, "removed") + count_occurrences(result, "deleted");
    let lines_modified = count_occurrences(result, "modified") + count_occurrences(result, "changed");
    
    // Basic validation rules
    let total_changes = lines_added + lines_removed + lines_modified;
    
    if total_changes == 0 {
        return "No changes detected - verify edit was needed".to_string();
    }
    
    // Check for suspicious patterns
    if lines_removed > 50 && lines_added < 5 {
        return format!("Warning: Removed {} lines but added only {} - verify this is correct", lines_removed, lines_added);
    }
    
    if lines_removed > lines_added * 3 && lines_removed > 10 {
        return format!("Warning: Removed significantly more lines ({}) than added ({}) - verify this is intentional", lines_removed, lines_added);
    }
    
    // Check if query suggests adding functionality but we only removed lines
    if query.to_lowercase().contains("add") && lines_removed > lines_added {
        return format!("Warning: Query asked to 'add' but removed {} lines, added {} - review changes", lines_removed, lines_added);
    }
    
    // Success cases
    if total_changes <= 5 {
        return "Edit completed - minor changes".to_string();
    } else if total_changes <= 20 {
        return format!("Edit completed - {} lines changed", total_changes);
    } else {
        return format!("Edit completed - {} lines changed (major update)", total_changes);
    }
}

/// Helper function to count occurrences of a word in text
fn count_occurrences(text: &str, word: &str) -> usize {
    text.to_lowercase().matches(&word.to_lowercase()).count()
}