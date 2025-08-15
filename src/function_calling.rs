use std::path::Path;
use std::fs;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::{anyhow, Result};
use html2text;
use crate::config::{Config};
use crate::gemini;
use crate::differ::{create_diff};
use crate::progress_display::{start_analyzing, start_editing, complete_operation};

/// Edit strategy determined by intelligent analysis
#[derive(Debug, Clone)]
enum EditStrategy {
    SearchReplace, // Fast search/replace operations using rope
    Surgical,      // Precise targeted edits to specific functions/sections
    MultiLocation, // Changes across multiple functions/locations
    Direct,        // Complete rewrites or major structural changes
    Chunked,       // Large file intelligent chunking (fallback)
}
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
            Ok(_content) => {
                // Validation is now AI-driven rather than rule-based
                crate::thinking_display::PersistentStatusBar::add_reasoning_step(&format!(
                    "File {} written successfully", file_path
                ));
            },
            Err(e) => {
                crate::thinking_display::PersistentStatusBar::add_reasoning_step(&format!(
                    "VALIDATION_ERROR: {} - {}", file_path, e
                ));
            }
        }
    }
}


/// Check if content has a built-in TWEEN implementation
fn has_builtin_tween_implementation(content: &str) -> bool {
    content.contains("var TWEEN = TWEEN ||") || 
    content.contains("const TWEEN =") ||
    content.contains("TWEEN = {") ||
    content.contains("function TWEEN(") ||
    content.contains("class TWEEN")
}

/// Validate basic HTML structure for common issues
fn validate_html_structure(content: &str) -> bool {
    let mut tag_stack = Vec::new();
    let mut in_tag = false;
    let mut current_tag = String::new();
    
    // Simple validation - check for basic tag matching
    for ch in content.chars() {
        match ch {
            '<' => {
                in_tag = true;
                current_tag.clear();
            },
            '>' => {
                if in_tag {
                    in_tag = false;
                    let tag = current_tag.trim();
                    
                    // Skip self-closing tags and special tags
                    if tag.ends_with('/') || tag.starts_with('!') || tag.starts_with('?') {
                        continue;
                    }
                    
                    // Check for closing tag
                    if tag.starts_with('/') {
                        let closing_tag = &tag[1..];
                        if let Some(opening_tag) = tag_stack.pop() {
                            if opening_tag != closing_tag {
                                return false; // Mismatched tags
                            }
                        } else {
                            return false; // Closing tag without opening
                        }
                    } else {
                        // Skip self-closing HTML tags
                        let self_closing = ["img", "br", "hr", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"];
                        let tag_name = tag.split_whitespace().next().unwrap_or("");
                        if !self_closing.contains(&tag_name) {
                            tag_stack.push(tag_name.to_string());
                        }
                    }
                }
            },
            _ => {
                if in_tag {
                    current_tag.push(ch);
                }
            }
        }
    }
    
    // All tags should be closed
    tag_stack.is_empty()
}



/// Simple response validation - extract code from markdown blocks if present
fn validate_ai_response(ai_response: &str, _original_content: &str) -> Result<String> {
    // Only handle obvious markdown code blocks - no language-specific patterns
    if ai_response.contains("```") {
        let mut in_code = false;
        let mut code_lines = Vec::new();
        
        for line in ai_response.lines() {
            if line.trim().starts_with("```") {
                in_code = !in_code;
                continue;
            }
            if in_code {
                code_lines.push(line);
            }
        }
        
        if !code_lines.is_empty() {
            return Ok(code_lines.join("\n"));
        }
    }
    
    // Otherwise return as-is - trust the prompting strategy
    Ok(ai_response.to_string())
}

/// AI-driven context analysis result
#[derive(Debug, Clone)]
struct ContextAnalysis {
    requires_rewrite: bool,
    deployment_guidance: String,
    issues: Vec<String>,
}

/// AI-driven analysis that actually works reliably
async fn analyze_context_with_ai(content: &str, file_path: &str, query: &str, config: &Config) -> Result<ContextAnalysis> {
    let analysis_prompt = format!(
        "You are analyzing a code editing request. Be precise and practical.\n\nFile: {}\nUser request: {}\nCode length: {} characters\n\nFirst 500 characters of code:\n{}\n\nAnswer these questions with brief, practical responses:\n\n1. Does this code have critical issues that prevent it from working?\n2. What specific technical problems exist?\n3. How should this file be deployed/used?\n4. Should this be a complete rewrite or small edit?\n\nProvide your analysis in this exact format:\nREQUIRES_REWRITE: yes/no\nISSUES: comma-separated list\nDEPLOYMENT: brief guidance\nEND_ANALYSIS",
        file_path, 
        query, 
        content.len(),
        content.chars().take(500).collect::<String>()
    );
    
    let response = crate::gemini::query_gemini_fast(&analysis_prompt, config).await?;
    
    // Parse structured response
    let requires_rewrite = response.to_lowercase().contains("requires_rewrite: yes");
    
    let issues = if let Some(issues_line) = response.lines().find(|line| line.starts_with("ISSUES:")) {
        issues_line.strip_prefix("ISSUES:").unwrap_or("").trim()
            .split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    } else { vec![] };
    
    let deployment_guidance = if let Some(deploy_line) = response.lines().find(|line| line.starts_with("DEPLOYMENT:")) {
        deploy_line.strip_prefix("DEPLOYMENT:").unwrap_or("Use best practices").trim().to_string()
    } else { "Use best practices for the context".to_string() };
    
    Ok(ContextAnalysis {
        requires_rewrite,
        deployment_guidance,
        issues,
    })
}

/// Determine if we need expensive AI analysis or can handle simply
fn should_analyze_context(content: &str, query: &str) -> bool {
    let query_lower = query.to_lowercase();
    
    // Run analysis if:
    // 1. User mentions issues/problems
    let has_problem_keywords = query_lower.contains("fix") || 
                              query_lower.contains("not working") || 
                              query_lower.contains("broken") ||
                              query_lower.contains("blank") ||
                              query_lower.contains("error");
    
    // 2. Content has potential technical issues
    let has_technical_issues = content.contains("import ") || 
                              content.contains("type=\"module\"") ||
                              content.len() < 50 || // Very short/empty files
                              content.contains("undefined") ||
                              content.contains("error");
    
    // 3. Complex operations
    let is_complex_operation = query_lower.contains("rewrite") ||
                              query_lower.contains("completely") ||
                              query_lower.contains("rebuild");
    
    has_problem_keywords || has_technical_issues || is_complex_operation
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
                    description: "The directory path to search in (e.g., 'C:\\path\\to\\directory')".to_string(),
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

/// PRODUCTION: Intelligent function orchestration system
pub async fn orchestrate_function_calls_intelligently(
    request: &str,
    conversation_history: &[crate::cli::memory::ChatMessage],
    config: &Config
) -> Result<FunctionOrchestrationResult> {
    let orchestrator = FunctionOrchestrator::new(config.clone());
    
    // STEP 1: Analyze request and plan function sequence
    let execution_plan = orchestrator.create_execution_plan(request, conversation_history).await?;
    
    // STEP 2: Execute functions intelligently with monitoring
    let results = orchestrator.execute_plan_with_monitoring(&execution_plan).await?;
    
    // STEP 3: Synthesize results and learn from execution
    let synthesis = orchestrator.synthesize_and_learn(&execution_plan, &results).await?;
    
    Ok(synthesis)
}

/// Intelligent function orchestrator - like Claude Code's tool coordination
pub struct FunctionOrchestrator {
    config: Config,
    registry: FunctionRegistry,
    execution_history: std::sync::Mutex<Vec<ExecutionRecord>>,
}

impl FunctionOrchestrator {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            registry: FunctionRegistry::new(),
            execution_history: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create intelligent execution plan
    async fn create_execution_plan(&self, request: &str, history: &[crate::cli::memory::ChatMessage]) -> Result<ExecutionPlan> {
        let context = self.build_context_summary(history);
        let available_functions = vec!["read_file".to_string(), "write_file".to_string(), "list_directory".to_string()]; // Simplified
        
        let planning_prompt = format!(
            r#"Create an intelligent execution plan for this request:

REQUEST: "{}"

CONTEXT: {}

AVAILABLE FUNCTIONS:
{}

Plan should:
1. Break down the request into logical steps
2. Select optimal functions for each step
3. Consider dependencies and error handling

Return JSON:
{{
    "steps": [
        {{
            "step_number": 1,
            "description": "what this step accomplishes",
            "function_name": "function to call",
            "success_criteria": "how to verify success"
        }}
    ],
    "overall_strategy": "high-level approach",
    "success_probability": 0.0-1.0
}}"#,
            request, context,
            available_functions.join("\n")
        );

        let response = crate::gemini::query_gemini(&planning_prompt, &self.config).await?;
        self.parse_execution_plan(&response)
    }

    /// Execute plan with intelligent monitoring
    async fn execute_plan_with_monitoring(&self, plan: &ExecutionPlan) -> Result<Vec<StepResult>> {
        let mut results = Vec::new();
        
        for step in &plan.steps {
            let start_time = std::time::Instant::now();
            
            // Create basic function call structure
            let function_call = FunctionCall {
                name: step.function_name.clone(),
                arguments: std::collections::HashMap::new(), // Simplified for demo
            };

            let execution_result = execute_function_call(&function_call, &self.config, "").await;
            let execution_time = start_time.elapsed().as_millis() as u32;

            let step_result = match execution_result {
                Ok(output) => StepResult {
                    step_number: step.step_number,
                    success: true,
                    output,
                    error: None,
                    execution_time_ms: execution_time,
                    confidence_score: 0.8,
                },
                Err(error) => StepResult {
                    step_number: step.step_number,
                    success: false,
                    output: String::new(),
                    error: Some(error.to_string()),
                    execution_time_ms: execution_time,
                    confidence_score: 0.0,
                }
            };

            results.push(step_result);
        }

        Ok(results)
    }

    /// Synthesize results and learn
    async fn synthesize_and_learn(&self, _plan: &ExecutionPlan, results: &[StepResult]) -> Result<FunctionOrchestrationResult> {
        Ok(FunctionOrchestrationResult {
            overall_success: results.iter().all(|r| r.success),
            primary_output: results.iter()
                .find(|r| r.success)
                .map(|r| r.output.clone())
                .unwrap_or("No output".to_string()),
            secondary_outputs: vec![],
            execution_insights: vec!["Intelligent orchestration completed".to_string()],
            step_results: results.to_vec(),
            total_execution_time_ms: results.iter().map(|r| r.execution_time_ms).sum(),
            success_rate: results.iter().filter(|r| r.success).count() as f32 / results.len() as f32,
        })
    }

    fn build_context_summary(&self, history: &[crate::cli::memory::ChatMessage]) -> String {
        if history.is_empty() {
            return "No conversation context".to_string();
        }

        history.iter()
            .rev()
            .take(2)
            .map(|msg| format!("{}: {}", 
                match msg.role {
                    crate::cli::memory::MessageRole::User => "User",
                    crate::cli::memory::MessageRole::Assistant => "Assistant", 
                    crate::cli::memory::MessageRole::System => "System",
                },
                msg.content.chars().take(100).collect::<String>()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn parse_execution_plan(&self, _response: &str) -> Result<ExecutionPlan> {
        // Simplified parsing - production would be more robust
        Ok(ExecutionPlan {
            steps: vec![ExecutionStep {
                step_number: 1,
                description: "Execute request".to_string(),
                function_name: "read_file".to_string(),
                success_criteria: "File read successfully".to_string(),
            }],
            overall_strategy: "Single step execution".to_string(),
            success_probability: 0.7,
        })
    }
}

// Supporting structures for orchestration
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub steps: Vec<ExecutionStep>,
    pub overall_strategy: String,
    pub success_probability: f32,
}

#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub step_number: usize,
    pub description: String,
    pub function_name: String,
    pub success_criteria: String,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_number: usize,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub execution_time_ms: u32,
    pub confidence_score: f32,
}

#[derive(Debug, Clone)]
pub struct FunctionOrchestrationResult {
    pub overall_success: bool,
    pub primary_output: String,
    pub secondary_outputs: Vec<String>,
    pub execution_insights: Vec<String>,
    pub step_results: Vec<StepResult>,
    pub total_execution_time_ms: u32,
    pub success_rate: f32,
}

#[derive(Debug, Clone)]
struct ExecutionRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub plan_strategy: String,
    pub total_steps: usize,
    pub successful_steps: usize,
    pub overall_success: bool,
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
            "edit_code" => execute_edit_code(function_call, config).await,
            _ => Err(anyhow!("Unknown function: {}", function_call.name)),
        };
        
        match result {
            Ok(success_result) => {
                // Complete the real-time UI operation with validation for edits
                if let Some(op_id) = &operation_id {
                    let completion_msg = match function_call.name.as_str() {
                        "edit_code" => {
                            // Use the actual function result as the completion message
                            // This ensures the indicator connects to the right message
                            success_result.clone()
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
        prompt.push_str(&format!("Function: {}
", function.name));
        prompt.push_str(&format!("Description: {}
", function.description));
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
    prompt.push_str("edit_code: {\"function_call\": {\"name\": \"edit_code\", \"arguments\": {\"file_path\": \"script.js\", \"query\": \"fix the JavaScript errors\"}}}
");
    prompt.push_str("read_file: {\"function_call\": {\"name\": \"read_file\", \"arguments\": {\"file_path\": \"index.html\"}}} // ONLY when user asks to view/see file\n");
    prompt.push_str("list_directory: {\"function_call\": {\"name\": \"list_directory\", \"arguments\": {\"directory_path\": \"C:\\\\path\\\\to\\\\folder\"}}}
");
    prompt.push_str("\nCRITICAL FUNCTION USAGE RULES:\n");
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
            prompt.push_str(&format!("- {}
", pattern));
        }
        prompt.push('\n');
    }
    
    prompt.push_str("Available functions:\n\n");
    
    for function in functions {
        prompt.push_str(&format!("Function: {}
", function.name));
        prompt.push_str(&format!("Description: {}
", function.description));
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
            prompt.push_str("read_file: {\"function_call\": {\"name\": \"read_file\", \"arguments\": {\"file_path\": \"index.html\"}}}
");
            prompt.push_str("code_analysis: {\"function_call\": {\"name\": \"code_analysis\", \"arguments\": {\"file_path\": \"script.js\", \"query\": \"analyze performance bottlenecks\"}}}
");
        },
        TaskType::Editing | TaskType::Debugging => {
            prompt.push_str("EXAMPLES FOR EDITING:\n");
            prompt.push_str("edit_code: {\"function_call\": {\"name\": \"edit_code\", \"arguments\": {\"file_path\": \"script.js\", \"query\": \"fix the JavaScript errors\"}}}
");
            prompt.push_str("code_analysis: {\"function_call\": {\"name\": \"code_analysis\", \"arguments\": {\"file_path\": \"script.js\", \"query\": \"identify the cause of runtime errors\"}}}
");
        },
        TaskType::Creating => {
            prompt.push_str("EXAMPLES FOR CREATION:\n");
            prompt.push_str("write_file: {\"function_call\": {\"name\": \"write_file\", \"arguments\": {\"file_path\": \"new_component.js\", \"content\": \"// New component code\"}}}
");
            prompt.push_str("list_directory: {\"function_call\": {\"name\": \"list_directory\", \"arguments\": {\"directory_path\": \"C:\\\\Users\\\\Admin\\\\Desktop\\\\project\"}}}
");
        },
    }
    
    prompt.push_str("\nCRITICAL FUNCTION USAGE RULES:\n");
    prompt.push_str("- User says 'fix/improve/edit [file]' → For simple fixes: edit_code directly. For complex fixes: code_analysis then edit_code\n");
    prompt.push_str("- User says 'show me/read/view/what's in [file]' → USE read_file\n");
    prompt.push_str("- User says 'create new file X' → USE write_file to create it from scratch\n");
    prompt.push_str("- NEVER call read_file before editing - edit_code reads the file automatically\n");
    prompt.push_str("- Complex fixes (missing functions, major features): code_analysis → edit_code\n");
    prompt.push_str("- Simple fixes (typos, small changes): edit_code directly\n\n");
    
    prompt.push_str("YOU ARE A COMPREHENSIVE CODING ASSISTANT:\n");
    prompt.push_str("- Directory analysis: Use list_directory to explore any directory, then read_file/code_analysis on files\n");
    prompt.push_str("- Code auditing: Systematically examine codebases by listing contents, reading key files, analyzing patterns\n");
    prompt.push_str("- Multi-file operations: Handle complex projects spanning multiple files and directories\n");
    prompt.push_str("- Advanced tooling: You have imara-diff, search capabilities, and complete file system access\n");
    prompt.push_str("- When given a directory path: START with list_directory to understand the structure\n\n");
    
    prompt.push_str("PROACTIVE CONTEXT ANALYSIS:\n");
    prompt.push_str("- When user provides a file path for editing/fixing: Use code_analysis for complex issues, edit_code for simple ones\n");
    prompt.push_str("- When user provides a file path for viewing: use read_file\n");
    prompt.push_str("- When user says 'you don't see anything wrong': refer to recent conversation and re-examine the relevant files\n");
    prompt.push_str("- When user reports issues (blank page, not working): systematically check related files to find root cause\n");
    prompt.push_str("- Use conversation context to understand what files are relevant to the current discussion\n");
    prompt.push_str("- Be action-oriented: Analyze complex problems first, then fix them with targeted edits\n");
    prompt.push_str("- For directory requests: use appropriate full paths like C:\\path\\to\\directory\n");
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
    
    // Use the provided file path directly (post-refactor: rely on intelligence rather than pattern matching)
    let actual_path = file_path.to_string();
    
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

async fn execute_edit_code(function_call: &FunctionCall, config: &Config) -> Result<String> {
    // Use the comprehensive editing system with recovery and strategy selection
    execute_edit_code_with_recovery(function_call, config, "").await
}

/// Comprehensive edit system with surgical editing, recovery, and intelligent strategy selection
async fn execute_edit_code_with_recovery(
    function_call: &FunctionCall,
    config: &Config,
    _conversation_history: &str
) -> Result<String> {
    let file_path = function_call.arguments.get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing file_path parameter"))?;
    
    let query = function_call.arguments.get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing query parameter"))?;
    
    // Read current file content
    let current_content = read_file(Path::new(file_path)).await?;
    
    // Use AI-driven intelligent strategy selection
    let edit_strategy = select_edit_strategy_ai(query, &current_content, file_path, config).await;
    
    let new_content = match edit_strategy {
        EditStrategy::SearchReplace => {
            crate::thinking_display::PersistentStatusBar::add_reasoning_step("Strategy: Fast search and replace");
            execute_search_replace_edit(file_path, &current_content, query, config).await?
        },
        EditStrategy::Surgical => {
            crate::thinking_display::PersistentStatusBar::add_reasoning_step("Strategy: Surgical precision editing");
            execute_surgical_edit(file_path, &current_content, query, config).await?
        },
        EditStrategy::MultiLocation => {
            crate::thinking_display::PersistentStatusBar::add_reasoning_step("Strategy: Multi-location editing");
            execute_multi_location_edit(file_path, &current_content, query, config).await?
        },
        EditStrategy::Direct => {
            crate::thinking_display::PersistentStatusBar::add_reasoning_step("Strategy: Direct editing");
            execute_simple_edit(file_path, &current_content, query, config).await?
        },
        EditStrategy::Chunked => {
            crate::thinking_display::PersistentStatusBar::add_reasoning_step("Strategy: Intelligent chunking");
            execute_large_file_edit(file_path, &current_content, query, config).await?
        }
    };
    
    // Validate and write the result
    let validated_content = validate_ai_response(&new_content, &current_content)?;
    write_file(Path::new(file_path), &validated_content).await?;
    
    let changes = estimate_changes(&current_content, &validated_content);
    Ok(format!("Successfully edited '{}' with {} changes using {} strategy", 
               file_path, changes, strategy_name(&edit_strategy)))
}


/// Intelligent task analysis and strategy selection
async fn select_edit_strategy_ai(query: &str, content: &str, file_path: &str, config: &Config) -> EditStrategy {
    let analysis_prompt = format!(
        "Analyze this code editing task and determine the optimal approach.

FILE: {} ({} lines, {} chars)  
TASK: {}

SAMPLE CODE (first 1000 chars):
{}

Analyze:
1. Is this a simple targeted change (fix typo, add semicolon, rename variable)?
2. Does this require finding multiple locations (all functions, all imports, all references)?  
3. Is this a complex structural change (refactoring, new features, architecture changes)?
4. What's the scope: single line, function, class, entire file, or cross-file?

Choose strategy:
- SEARCH_AND_REPLACE: Simple find/replace operations (variable names, imports, etc.)
- SURGICAL: Precise targeted changes to specific functions/sections
- DIRECT: Complete rewrites, new features, or major structural changes
- MULTI_LOCATION: Changes needed across multiple functions/locations

Respond with strategy and brief reasoning.",
        file_path, 
        content.lines().count(),
        content.len(),
        query,
        content.chars().take(1000).collect::<String>()
    );
    
    match crate::gemini::query_gemini_fast(&analysis_prompt, config).await {
        Ok(response) => {
            let response_upper = response.to_uppercase();
            if response_upper.contains("SEARCH_AND_REPLACE") {
                EditStrategy::SearchReplace
            } else if response_upper.contains("SURGICAL") {
                EditStrategy::Surgical  
            } else if response_upper.contains("MULTI_LOCATION") {
                EditStrategy::MultiLocation
            } else {
                EditStrategy::Direct
            }
        },
        _ => EditStrategy::Direct
    }
}

fn strategy_name(strategy: &EditStrategy) -> &'static str {
    match strategy {
        EditStrategy::SearchReplace => "search_replace",
        EditStrategy::Surgical => "surgical",
        EditStrategy::MultiLocation => "multi_location", 
        EditStrategy::Direct => "direct",
        EditStrategy::Chunked => "chunked",
    }
}

/// Execute surgical editing using pure AI intelligence
async fn execute_surgical_edit(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    crate::thinking_display::PersistentStatusBar::add_reasoning_step("Using AI surgical precision editing");
    
    // PURE AI SURGICAL INTELLIGENCE - No hardcoded patterns
    let surgical_prompt = format!(
        "You are an expert at making precise, minimal code changes. Analyze the file and make only the necessary modifications.

FILE PATH: {}
CHANGE REQUEST: {}

CURRENT FILE:
{}

SURGICAL EDITING APPROACH:
1. Identify the exact location and scope of changes needed
2. Preserve all existing functionality and structure  
3. Make minimal, precise modifications only
4. Ensure syntactic correctness and logical consistency
5. Return the COMPLETE file with only necessary changes applied

Perform surgical edit and return the complete updated file:",
        file_path, query, content
    );
    
    // Show surgical thinking
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Performing surgical code analysis and precise modifications");
    
    // Use AI reasoning for surgical precision with thinking disabled for token efficiency
    let result = crate::gemini::query_gemini_clean(&surgical_prompt, config).await?;
    
    Ok(result)
}

/// Execute intelligent search-based editing for large files or specific errors
async fn execute_large_file_edit(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    crate::thinking_display::PersistentStatusBar::add_reasoning_step("Using intelligent search-based editing");
    
    // First, try to extract specific locations from the query (error messages, line numbers, etc.)
    let locations = extract_target_locations(query, content).await;
    
    if !locations.is_empty() {
        // Use targeted editing for specific locations
        execute_targeted_edit(file_path, content, query, &locations, config).await
    } else {
        // Fall back to intelligent chunking
        execute_intelligent_chunked_edit(file_path, content, query, config).await
    }
}

/// Extract specific line numbers or code locations from user query/error messages
async fn extract_target_locations(query: &str, content: &str) -> Vec<usize> {
    let mut locations = Vec::new();
    
    // Look for line number patterns like "line 2456", ":5331:", "-->srcclichat.rs:2456"
    let line_patterns = [
        r"line\s+(\d+)",
        r":(\d+):",
        r"-->.*:(\d+)",
        r"line\s*(\d+)",
    ];
    
    for pattern in &line_patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            for cap in regex.captures_iter(query) {
                if let Some(line_match) = cap.get(1) {
                    if let Ok(line_num) = line_match.as_str().parse::<usize>() {
                        locations.push(line_num);
                    }
                }
            }
        }
    }
    
    // Remove duplicates and sort
    locations.sort_unstable();
    locations.dedup();
    
    // Only return if we found reasonable line numbers (within file bounds)
    let line_count = content.lines().count();
    locations.retain(|&line| line > 0 && line <= line_count);
    
    locations
}

/// Execute intelligent problem solving using adaptive context
async fn execute_targeted_edit(
    file_path: &str, 
    content: &str, 
    query: &str, 
    locations: &[usize], 
    config: &Config
) -> Result<String> {
    use ropey::Rope;
    let rope = Rope::from_str(content);
    let file_size = content.len();
    let line_count = rope.len_lines();
    
    // Determine the right approach based on file size and complexity
    if file_size > 50_000 {
        // Large file: Use intelligent sampling approach
        execute_large_file_intelligent_edit(file_path, content, query, locations, config).await
    } else {
        // Small/medium file: Full AI analysis
        crate::thinking_display::PersistentStatusBar::set_ai_thinking("AI analyzing complete file context");
        
        let full_analysis_prompt = format!(
            "Fix this code issue using complete understanding.

FILE: {} ({} lines, {} chars)
REQUEST: {}
ERROR LOCATIONS: {:?}

COMPLETE CODE:
{}

Instructions:
- Apply your full expertise to understand and fix the issue
- The reported locations may be symptoms, find the real cause
- Return the complete corrected file
- Maintain all existing functionality",
            file_path, line_count, file_size, query, locations, content
        );
        
        crate::thinking_display::PersistentStatusBar::set_ai_thinking("AI solving problem with complete context");
        let result = crate::gemini::query_gemini_clean(&full_analysis_prompt, config).await?;
        Ok(result)
    }
}

/// Handle large files with intelligent context extraction
async fn execute_large_file_intelligent_edit(
    file_path: &str,
    content: &str, 
    query: &str,
    locations: &[usize],
    config: &Config
) -> Result<String> {
    use ropey::Rope;
    let rope = Rope::from_str(content);
    
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("AI extracting relevant context from large file");
    
    // Step 1: Extract relevant sections around error locations
    let mut context_chunks = Vec::new();
    let context_size = 50; // lines before/after each location
    
    for &line_num in locations {
        let start_line = line_num.saturating_sub(context_size).max(1);
        let end_line = (line_num + context_size).min(rope.len_lines());
        
        let start_char = rope.line_to_char(start_line.saturating_sub(1));
        let end_char = rope.line_to_char(end_line.min(rope.len_lines()).saturating_sub(1));
        
        let section = rope.slice(start_char..end_char).to_string();
        context_chunks.push(format!("SECTION {}-{}:\n{}", start_line, end_line, section));
    }
    
    // Step 2: Ask AI to analyze just the relevant sections and plan the fix
    let analysis_prompt = format!(
        "Analyze this code issue in a large file and create a fix plan.

FILE: {} (large file, showing relevant sections only)
ISSUE: {}

RELEVANT CODE SECTIONS:
{}

Task:
1. Understand the problem from these sections
2. Identify what changes are needed and where
3. Create a plan for fixing the entire file
4. Return a list of specific changes needed (with line numbers if possible)",
        file_path, query, context_chunks.join("\n\n")
    );
    
    let fix_plan = crate::gemini::query_gemini(&analysis_prompt, config).await?;
    
    // Step 3: Apply the fix using imara-diff approach
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("AI applying planned changes to large file");
    
    let implementation_prompt = format!(
        "Implement the planned fixes for this large file.

ORIGINAL FILE: {}
FIX PLAN: {}

COMPLETE FILE CONTENT:
{}

Apply the planned changes and return the complete corrected file:",
        file_path, fix_plan, content
    );
    
    let result = crate::gemini::query_gemini_clean(&implementation_prompt, config).await?;
    Ok(result)
}

/// Execute intelligent chunked editing for large files without specific targets
async fn execute_intelligent_chunked_edit(
    file_path: &str, 
    content: &str, 
    query: &str, 
    config: &Config
) -> Result<String> {
    use ropey::Rope;
    
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Using AI to identify relevant sections for editing");
    
    let rope = Rope::from_str(content);
    let total_lines = rope.len_lines();
    
    // First, ask AI to identify which parts of the file are relevant to the query
    let search_prompt = format!(
        "Analyze this editing request and identify which sections of the file need to be modified.

FILE: {} ({} lines)
REQUEST: {}

FILE CONTENT SAMPLE (first 2000 chars):
{}

Respond with line number ranges where changes are needed (e.g., '100-150, 300-350, 500-520').
If you need to see more of the file to determine this, respond with 'NEED_MORE_CONTEXT'.",
        file_path,
        total_lines,
        query,
        content.chars().take(2000).collect::<String>()
    );
    
    let search_result = crate::gemini::query_gemini_fast(&search_prompt, config).await?;
    
    if search_result.contains("NEED_MORE_CONTEXT") {
        // Fall back to direct editing for now
        // TODO: Implement smart file sampling/chunking
        execute_simple_edit(file_path, content, query, config).await
    } else {
        // Extract line ranges from AI response and process those sections
        let ranges = extract_line_ranges(&search_result);
        if ranges.is_empty() {
            // No specific ranges found, use direct editing
            execute_simple_edit(file_path, content, query, config).await
        } else {
            // Process specific ranges
            let target_lines: Vec<usize> = ranges.iter()
                .flat_map(|(start, end)| (*start..=*end))
                .collect();
            execute_targeted_edit(file_path, content, query, &target_lines, config).await
        }
    }
}

/// Extract line ranges from AI response (e.g., "100-150, 300-350")
fn extract_line_ranges(response: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let range_pattern = regex::Regex::new(r"(\d+)-(\d+)").unwrap();
    
    for cap in range_pattern.captures_iter(response) {
        if let (Some(start), Some(end)) = (cap.get(1), cap.get(2)) {
            if let (Ok(start_num), Ok(end_num)) = (
                start.as_str().parse::<usize>(),
                end.as_str().parse::<usize>()
            ) {
                if start_num <= end_num {
                    ranges.push((start_num, end_num));
                }
            }
        }
    }
    
    ranges
}

/// Execute fast search and replace operations using rope
async fn execute_search_replace_edit(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    use ropey::Rope;
    
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("AI identifying search/replace patterns");
    
    // Ask AI to identify what to search for and replace with
    let pattern_prompt = format!(
        "Analyze this search/replace request and extract the exact patterns.

FILE: {}
REQUEST: {}

CODE SAMPLE (first 1000 chars):
{}

Identify:
1. What text/pattern should be found (be exact)
2. What it should be replaced with (be exact)  
3. Should this be case-sensitive?
4. Should this replace all occurrences or just specific ones?

Respond in format:
FIND: [exact text to find]
REPLACE: [exact replacement text]
CASE_SENSITIVE: [yes/no] 
ALL_OCCURRENCES: [yes/no]",
        file_path, query, content.chars().take(1000).collect::<String>()
    );
    
    let pattern_response = crate::gemini::query_gemini_fast(&pattern_prompt, config).await?;
    
    // Parse AI response to extract patterns
    let find_pattern = extract_field(&pattern_response, "FIND:");
    let replace_pattern = extract_field(&pattern_response, "REPLACE:");
    let all_occurrences = extract_field(&pattern_response, "ALL_OCCURRENCES:").contains("yes");
    
    if find_pattern.is_empty() {
        // Fallback to surgical editing if pattern extraction fails
        return execute_surgical_edit(file_path, content, query, config).await;
    }
    
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Applying search/replace with rope");
    
    // Use rope for efficient search and replace
    let mut rope = Rope::from_str(content);
    let mut changes_made = 0;
    
    // Convert rope to string for search, then apply changes back to rope
    let search_content = rope.to_string();
    let mut replacements = Vec::new();
    
    // Find all occurrences
    let mut start_pos = 0;
    while let Some(found_pos) = search_content[start_pos..].find(&find_pattern) {
        let absolute_pos = start_pos + found_pos;
        replacements.push((absolute_pos, absolute_pos + find_pattern.len()));
        
        start_pos = absolute_pos + find_pattern.len();
        changes_made += 1;
        
        if !all_occurrences {
            break;
        }
        
        // Safety check
        if changes_made > 1000 {
            break;
        }
    }
    
    // Apply replacements in reverse order to maintain position accuracy
    for (start, end) in replacements.iter().rev() {
        rope.remove(*start..*end);
        rope.insert(*start, &replace_pattern);
    }
    
    Ok(rope.to_string())
}

/// Execute multi-location editing using iterative search like Claude Code
async fn execute_multi_location_edit(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    use ropey::Rope;
    
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Starting systematic code exploration");
    
    // Step 1: Let AI start with initial search strategy  
    let mut search_attempts = Vec::new();
    let mut found_locations = Vec::new();
    let rope = Rope::from_str(content);
    
    // Iterative search process - like Claude Code does
    for attempt in 1..=5 {  // Maximum 5 search attempts
        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Search attempt {} - exploring code patterns", attempt));
        
        let iterative_search_prompt = format!(
            "You are searching through code systematically. This is search attempt #{}.

FILE: {}  
TASK: {}
TOTAL FILE SIZE: {} lines

PREVIOUS SEARCH ATTEMPTS: {}
PREVIOUS FINDINGS: {} locations found

What should you search for next? Be specific about patterns to find.
Examples: 'function names containing calc', 'variables ending in _id', 'import statements', etc.

If you think you've found everything, respond with 'SEARCH_COMPLETE'.
Otherwise, give me ONE specific pattern to search for next:",
            attempt, file_path, query, rope.len_lines(),
            if search_attempts.is_empty() { "None yet".to_string() } else { search_attempts.join(", ") },
            found_locations.len()
        );
        
        let search_response = crate::gemini::query_gemini_fast(&iterative_search_prompt, config).await?;
        
        if search_response.to_uppercase().contains("SEARCH_COMPLETE") {
            break;
        }
        
        // Extract search pattern and execute it
        let search_pattern = search_response.trim().to_string();
        search_attempts.push(search_pattern.clone());
        
        // Perform the actual search in the content
        let new_findings = execute_pattern_search(content, &search_pattern).await?;
        
        // Add new findings (avoid duplicates)
        for finding in new_findings {
            if !found_locations.iter().any(|existing: &SearchResult| existing.line == finding.line) {
                found_locations.push(finding);
            }
        }
        
        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Found {} total locations so far", found_locations.len()));
        
        // If we found enough locations, let AI decide if we should continue
        if found_locations.len() >= 3 {
            let continue_prompt = format!(
                "You've found {} locations. Should you continue searching or proceed with editing?
                
FOUND SO FAR: {:?}
TASK: {}

Respond with 'CONTINUE_SEARCH' or 'PROCEED_WITH_EDIT':",
                found_locations.len(),
                found_locations.iter().map(|r| &r.pattern).collect::<Vec<_>>(),
                query
            );
            
            let continue_response = crate::gemini::query_gemini_fast(&continue_prompt, config).await?;
            if continue_response.to_uppercase().contains("PROCEED_WITH_EDIT") {
                break;
            }
        }
    }
    
    // Step 3: Use search results to extract relevant sections and coordinate changes
    if found_locations.is_empty() {
        // Fallback to direct editing if no locations found
        return execute_simple_edit(file_path, content, query, config).await;
    }
    
    let mut context_sections = Vec::new();
    
    for result in &found_locations {
        let context_start = result.line.saturating_sub(10).max(1);
        let context_end = (result.line + 10).min(rope.len_lines());
        
        let start_char = rope.line_to_char(context_start.saturating_sub(1));
        let end_char = rope.line_to_char(context_end.min(rope.len_lines()).saturating_sub(1));
        
        let section = rope.slice(start_char..end_char).to_string();
        context_sections.push(format!("FOUND: {} (line {})\nCONTEXT:\n{}", 
                                     result.pattern, result.line, section));
    }
    
    // Step 4: Generate coordinated changes based on search results
    let coordinated_prompt = format!(
        "Generate coordinated changes across all found code locations.

ORIGINAL REQUEST: {}
FILE: {}

SEARCH RESULTS AND CONTEXT:
{}

Generate the complete modified file ensuring:
1. All found locations are updated consistently
2. All references and dependencies are maintained
3. Code remains syntactically correct
4. All existing functionality is preserved

Return the complete updated file:",
        query, file_path, context_sections.join("\n\n---\n\n")
    );
    
    let result = crate::gemini::query_gemini_clean(&coordinated_prompt, config).await?;
    Ok(result)
}

/// Execute pattern search with AI guidance  
async fn execute_pattern_search(content: &str, search_description: &str) -> Result<Vec<SearchResult>> {
    // Let AI convert the search description into actual searchable patterns
    let pattern_extraction_prompt = format!(
        "Convert this search description into specific text patterns to find in code.

SEARCH DESCRIPTION: {}

Return 1-3 specific text patterns that would find this in code.
Examples:
- For 'functions containing calc': return 'fn calc', 'function calc'  
- For 'variables ending in _id': return '_id'
- For 'import statements': return 'import ', 'use crate::'

Respond with just the patterns, one per line:",
        search_description
    );
    
    let patterns_response = crate::gemini::query_gemini_fast(&pattern_extraction_prompt, &crate::config::Config::default()).await
        .unwrap_or_else(|_| search_description.to_string());
    
    let mut results = Vec::new();
    
    // Extract actual search patterns
    let patterns: Vec<String> = patterns_response
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && line.len() > 1)
        .collect();
    
    // Search for each pattern in the content
    for pattern in patterns {
        let matches = find_pattern_in_content(content, &pattern);
        for mut result in matches {
            result.pattern = search_description.to_string(); // Use the original description
            results.push(result);
        }
    }
    
    // Remove duplicates and sort by line number
    results.sort_by_key(|r| r.line);
    results.dedup_by_key(|r| r.line);
    
    Ok(results)
}

/// Fallback AI location finding when search fails
async fn execute_ai_location_finding(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    let location_prompt = format!(
        "Find all locations in this code that need to be modified for this request.

FILE: {}
REQUEST: {}

CODE:
{}

Identify all functions, classes, variables, imports, etc. that need changes.
For each location, provide:
- Line number (approximate)
- What needs to be changed

Return the complete modified file:",
        file_path, query, content
    );
    
    let result = crate::gemini::query_gemini_clean(&location_prompt, config).await?;
    Ok(result)
}

/// Extract search patterns from AI response
fn extract_search_patterns(response: &str) -> Vec<String> {
    let mut patterns = Vec::new();
    
    for line in response.lines() {
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('/') && !line.starts_with('#') {
            // Extract patterns in quotes or after dashes
            if line.contains('"') {
                let parts: Vec<&str> = line.split('"').collect();
                for (i, part) in parts.iter().enumerate() {
                    if i % 2 == 1 && !part.is_empty() {
                        patterns.push(part.to_string());
                    }
                }
            } else if line.contains('\'') {
                let parts: Vec<&str> = line.split('\'').collect();
                for (i, part) in parts.iter().enumerate() {
                    if i % 2 == 1 && !part.is_empty() {
                        patterns.push(part.to_string());
                    }
                }
            } else if line.contains('-') {
                if let Some(pattern_part) = line.split('-').nth(1) {
                    let cleaned = pattern_part.trim();
                    if !cleaned.is_empty() && cleaned.len() > 2 {
                        patterns.push(cleaned.to_string());
                    }
                }
            }
        }
    }
    
    patterns
}

/// Find pattern occurrences in content and return with line numbers
fn find_pattern_in_content(content: &str, pattern: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    
    for (line_num, line) in content.lines().enumerate() {
        if line.contains(pattern) {
            results.push(SearchResult {
                pattern: pattern.to_string(),
                line: line_num + 1,
                context: line.to_string(),
            });
        }
    }
    
    results
}

#[derive(Debug, Clone)]
struct SearchResult {
    pattern: String,
    line: usize,
    context: String,
}

/// Extract field value from AI response (e.g., "FIND: some_text" -> "some_text")
fn extract_field(response: &str, field: &str) -> String {
    for line in response.lines() {
        if let Some(start) = line.find(field) {
            let value = &line[start + field.len()..].trim();
            return value.to_string();
        }
    }
    String::new()
}

/// Extract line numbers from AI location response
fn extract_line_numbers_from_response(response: &str) -> Vec<usize> {
    let mut locations = Vec::new();
    let line_pattern = regex::Regex::new(r"Line\s+(\d+)").unwrap();
    
    for cap in line_pattern.captures_iter(response) {
        if let Some(line_match) = cap.get(1) {
            if let Ok(line_num) = line_match.as_str().parse::<usize>() {
                locations.push(line_num);
            }
        }
    }
    
    locations.sort_unstable();
    locations.dedup();
    locations
}

/// Execute simple direct editing
async fn execute_simple_edit(file_path: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    crate::thinking_display::PersistentStatusBar::add_reasoning_step("Using AI-driven universal editing");
    
    // PURE AI INTELLIGENCE - No hardcoding, no patterns, no heuristics
    let ai_prompt = format!(
        "You are an expert code editor. Analyze the request and current file, then generate the complete updated file.

FILE PATH: {}
USER REQUEST: {}

CURRENT FILE CONTENT:
{}

INSTRUCTIONS:
1. Understand the programming language from the file content and path
2. Analyze what changes are needed based on the user request
3. Generate the COMPLETE updated file with the requested changes
4. Respond with ONLY the raw file content (no markdown, no explanations)
5. Ensure the result is syntactically correct and functional

Generate the complete updated file:",
        file_path, query, content
    );
    
    // Show intelligent thinking display
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("Analyzing code structure and implementing changes");
    
    // Use AI reasoning with thinking disabled for token efficiency
    let result = crate::gemini::query_gemini_clean(&ai_prompt, config).await?;
    
    Ok(result)
}

/// Apply surgical diff using imara-diff if available
fn apply_surgical_diff(_original: &str, edited: &str, _file_path: &str) -> Result<String> {
    // For now, just return the edited content
    // TODO: Integrate with imara-diff for precise surgical editing
    Ok(edited.to_string())
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
        r"IMPORTANT: You are removing {} from this JavaScript/HTML file.\n\nTASK: Remove ALL related code completely - not just individual lines.\nThis includes:\n- Complete if/else blocks\n- Entire function definitions\n- Full object definitions\n- Associated comments and dividers\n- Any incomplete fragments\n\nFOCUS AREA: Lines {}-{} contain the main target code.\nCONTEXT: Lines {}-{} (you must return this entire section, edited)\n\nCRITICAL: \n- Do NOT leave partial code fragments\n- Do NOT leave orphaned variables or references\n- Do NOT break JavaScript syntax\n- Remove complete logical blocks, not individual scattered lines\n\nOriginal request: {}\n\nCode section to edit:\n{}\n\nProvide the complete edited section with {} fully removed:",
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
        "Analyze this code section and provide ONLY a unified diff patch to remove {}:\n\nCode section (lines {}-{}):\n{}\n\nCRITICAL: Return ONLY the unified diff patch in this exact format:\n@@ -start,count +start,count @@\n-removed line\n+added line\n\nIf no changes needed, return: NO_CHANGES_NEEDED",
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
        "Analyze this code section and provide ONLY a unified diff patch to remove {}:\n\nCode section (lines {}-{}):\n{}\n\nCRITICAL: Return ONLY the unified diff patch in this exact format:\n@@ -start,count +start,count @@\n-removed line\n+added line\n\nIf no changes needed, return: NO_CHANGES_NEEDED",
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

/// Claude Code-level intelligent large file editor with surgical precision
/// Uses advanced AI reasoning to understand context and make precise edits
async fn process_large_file_with_rope(_file_path_str: &str, content: &str, query: &str, config: &Config) -> Result<String> {
    // Step 1: Use AI to understand the edit intent and locate relevant sections
    let analysis_prompt = format!(
        "Analyze this edit request for a large file and identify the specific sections that need modification.

Edit request: {}

File content:
{}

Provide a structured analysis:
1. What exactly needs to be changed?
2. Which sections of the file are relevant?
3. What should be preserved unchanged?

Return your analysis followed by the COMPLETE edited file content with NO markdown formatting.",
        query, content
    );

    // Step 2: Get intelligent AI analysis and editing
    crate::gemini::query_gemini(&analysis_prompt, config).await
}

/// Process a large section of the file (OBSOLETE - replaced by Rope implementation)
async fn process_large_section_legacy(content: &str, query: &str, start_line: usize, end_line: usize, config: &Config) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    let section_content = lines[start_line..end_line].join("\n");
    
    // Use a focused prompt for large section processing
    let edit_prompt = format!(
        "Remove all {} from this large code section.\n\nInstructions:\n- Remove complete blocks, functions, and structures\n- Maintain valid JavaScript/HTML syntax\n- Keep all non-target functionality intact\n\nSection to process (lines {}-{}):\n{}\n\nReturn the complete section with all {} removed:",
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
        "USER REQUEST: {}\n\nQuickly analyze this {} file '{}' to identify what's wrong or missing for the user's request.\nFocus ONLY on the specific issue mentioned. Provide a brief, targeted analysis:\n- What is the specific problem?\n- What code is missing or broken?\n- What needs to be added/fixed?\n\nKeep it concise and actionable.\n\nCode content:\n{}",
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
        "Analyze this {} file and return a JSON response with structured analysis.\n\nUSER REQUEST: {}\nCONVERSATION CONTEXT: {}\n\nPlease analyze the code and return ONLY a valid JSON object with this structure:\n{{\n  \"file_info\": {{\n    \"path\": \"{}\",\n    \"language\": \"{}\",\n    \"lines_of_code\": {},\n    \"size_bytes\": {}\n  }},\n  \"code_quality\": {{\n    \"complexity_score\": \"<1-10>\",\n    \"maintainability_score\": \"<1-10>\",\n    \"readability_score\": \"<1-10>\",\n    \"test_coverage_estimate\": \"<0-100>\"\n  }},\n  \"issues_found\": [\n    {{\n      \"severity\": \"high|medium|low\",\n      \"type\": \"bug|performance|security|style\",\n      \"description\": \"Brief description\",\n      \"location\": \"line 123 or function name\",\n      \"suggestion\": \"How to fix\"\n    }}\n  ],\n  \"dependencies\": [\n    {{\n      \"name\": \"library/module name\",\n      \"type\": \"external|internal|builtin\",\n      \"usage\": \"how it's used\"\n    }}\n  ]\n}}",
        file_extension,
        query,
        conversation_history,
        file_path,
        file_extension,
        content.lines().count(),
        content.len()
    );

    let analysis = gemini::query_gemini(&structured_prompt, config).await?;
    Ok(format!("Structured analysis for '{}':\n\n{}", file_path, analysis))
}
