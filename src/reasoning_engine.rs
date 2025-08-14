use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::config::Config;
use crate::gemini;
use crate::cli::colors::{EMERALD_BRIGHT, BLUE_BRIGHT, GRAY_DIM, RESET};

/// Advanced reasoning engine for handling ambiguous requests
pub struct ReasoningEngine {
    config: Config,
    heuristic_database: HeuristicDatabase,
}

/// Database of heuristics for common ambiguous request patterns
#[derive(Debug, Clone)]
struct HeuristicDatabase {
    domain_criteria: Vec<DomainCriterion>,
    code_improvement_patterns: Vec<CodeImprovementPattern>,
    data_filter_templates: HashMap<String, FilterTemplate>,
}

/// Criteria for evaluating domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCriterion {
    pub name: String,
    pub description: String,
    pub weight: f32,
    pub evaluation_fn: DomainEvalFunction,
}

/// Code improvement pattern matching
#[derive(Debug, Clone)]
struct CodeImprovementPattern {
    pattern: String,
    description: String,
    suggested_improvements: Vec<String>,
    priority: u8,
}

/// Generic filter template for data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterTemplate {
    pub name: String,
    pub description: String,
    pub criteria: Vec<FilterCriterion>,
    pub default_limit: usize,
}

/// Individual filter criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCriterion {
    pub field: String,
    pub operator: FilterOperator,
    pub value: FilterValue,
    pub weight: f32,
}

/// Supported filter operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Contains,
    StartsWith,
    EndsWith,
    LengthLessThan,
    LengthGreaterThan,
    Equals,
    Matches,
    Score,
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Regex(String),
}

/// Domain evaluation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainEvalFunction {
    Length,
    Brandability,
    KeywordRelevance,
    Memorability,
    PronunciationEase,
    TldValue,
    MarketValue,
}

/// Result of reasoning about an ambiguous request
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    pub interpretation: String,
    pub suggested_criteria: Vec<String>,
    pub clarification_questions: Vec<String>,
    pub confidence: f32,
    pub actionable_plan: Vec<String>,
    pub capability_assessment: Vec<String>,
    pub metacognitive_notes: String,
}

impl ReasoningEngine {
    pub fn new(config: &Config) -> Self {
        let heuristic_database = HeuristicDatabase::new();
        
        Self {
            config: config.clone(),
            heuristic_database,
        }
    }

    /// Fast parallel reasoning with instant local analysis
    pub async fn reason_about_request_with_metacognition(&self, request: &str, context: &str) -> Result<ReasoningResult> {
        // Start multiple fast local analyses in parallel
        let request_lower = request.to_lowercase();
        
        let (pattern_analysis, context_analysis, capability_analysis) = tokio::join!(
            // Pattern matching (instant)
            async {
                (
                    self.is_domain_request(&request_lower),
                    self.is_improvement_request(&request_lower),
                    self.is_selection_request(&request_lower),
                    self.is_analysis_request(&request_lower)
                )
            },
            
            // Context analysis (instant)
            async {
                let has_file_refs = context.contains(".rs") || context.contains(".js") || context.contains(".py");
                let has_error_context = context.contains("error") || context.contains("failed");
                (has_file_refs, has_error_context)
            },
            
            // Capability assessment (instant)
            async {
                vec![
                    "I have access to conversation history via sled database".to_string(),
                    "I can search for previous changes and content".to_string(),
                    "I can analyze file structures and code".to_string(),
                    "I can make intelligent inferences from context".to_string(),
                ]
            }
        );
        
        // Build reasoning result from parallel analysis (no API call needed!)
        let (is_domain, is_improvement, is_selection, is_analysis) = pattern_analysis;
        let (has_files, has_errors) = context_analysis;
        
        let mut reasoning = ReasoningResult {
            interpretation: self.build_instant_interpretation(&request_lower, is_domain, is_improvement, is_selection, is_analysis),
            suggested_criteria: self.build_instant_criteria(&request_lower),
            clarification_questions: vec![], // Avoid asking questions - be proactive
            confidence: 0.9, // High confidence in local analysis
            actionable_plan: self.build_instant_plan(&request_lower, has_files, has_errors),
            capability_assessment: capability_analysis.clone(),
            metacognitive_notes: "Fast parallel local analysis - no API calls needed for basic reasoning".to_string(),
        };
        
        // Only use external AI if we genuinely can't handle locally
        if reasoning.confidence < 0.5 {
            reasoning = self.reason_about_request(request, context).await?;
            reasoning.capability_assessment = capability_analysis;
        }
        
        Ok(reasoning)
    }
    
    /// Build interpretation instantly from pattern matching
    fn build_instant_interpretation(&self, request: &str, is_domain: bool, is_improvement: bool, is_selection: bool, is_analysis: bool) -> String {
        if is_domain {
            "Domain evaluation request - I can analyze domain criteria".to_string()
        } else if is_improvement {
            "Code improvement request - I can suggest enhancements".to_string()
        } else if is_selection {
            "Selection/filtering request - I can help choose the best options".to_string()
        } else if is_analysis {
            "Analysis request - I can examine and explain".to_string()
        } else {
            format!("General request: '{}' - I can take action", request)
        }
    }
    
    /// Build criteria instantly from request patterns
    fn build_instant_criteria(&self, request: &str) -> Vec<String> {
        let mut criteria = Vec::new();
        
        if request.contains("best") || request.contains("good") {
            criteria.push("Quality and effectiveness".to_string());
        }
        if request.contains("fast") || request.contains("quick") {
            criteria.push("Speed and efficiency".to_string());
        }
        if request.contains("simple") || request.contains("easy") {
            criteria.push("Simplicity and usability".to_string());
        }
        
        if criteria.is_empty() {
            criteria.push("Practical and actionable solutions".to_string());
        }
        
        criteria
    }
    
    /// Build action plan instantly from request analysis
    fn build_instant_plan(&self, request: &str, has_files: bool, has_errors: bool) -> Vec<String> {
        let mut plan = Vec::new();
        
        if request.contains("analyze") || request.contains("examine") {
            plan.push("Examine the relevant files or data".to_string());
        }
        
        if request.contains("fix") || request.contains("repair") || has_errors {
            plan.push("Identify and resolve the issue".to_string());
        }
        
        if request.contains("improve") || request.contains("enhance") {
            plan.push("Suggest and implement improvements".to_string());
        }
        
        if has_files {
            plan.push("Work with the specified files".to_string());
        }
        
        if plan.is_empty() {
            plan.push("Take appropriate action based on the request".to_string());
        }
        
        plan
    }
    
    /// Try to resolve request internally using existing capabilities and smart intent recognition
    pub async fn try_internal_resolution(&self, request: &str, _context: &str, reasoning: &ReasoningResult) -> Result<Option<String>> {
        let request_lower = request.to_lowercase();
        
        // SMART FILE OPERATION INTENT DETECTION
        // Handle confusing requests like "can you make a new html can you remake this file"
        if let Some(intent) = self.detect_file_operation_intent(&request_lower) {
            match intent {
                FileOperationIntent::CreateNew { path, description } => {
                    return Ok(Some(format!(
                        "I understand you want to create a new file at '{}' {}. Let me create that for you.",
                        path, description
                    )));
                },
                FileOperationIntent::EditExisting { path, changes } => {
                    return Ok(Some(format!(
                        "I understand you want to edit the existing file '{}' to {}. Let me modify that file.",
                        path, changes
                    )));
                },
                FileOperationIntent::CreateOrEdit { path, description } => {
                    // Check if file exists to decide
                    if std::path::Path::new(&path).exists() {
                        return Ok(Some(format!(
                            "File '{}' exists. I'll edit it to {}.", path, description
                        )));
                    } else {
                        return Ok(Some(format!(
                            "File '{}' doesn't exist. I'll create it as {}.", path, description
                        )));
                    }
                }
            }
        }
        
        // Check if this is something we can handle without external AI
        if request_lower.contains("revert") && request_lower.contains("change") {
            // This is exactly the type of metacognitive thinking we want!
            return Ok(Some(format!(
                "I understand you want to revert changes. While I don't have direct version control access, I do have conversation history stored in a sled database. Let me search for the previous state of the file you mentioned.\n\nBased on my capabilities: {}\n\nI should search the conversation history for the original content.",
                reasoning.capability_assessment.join(", ")
            )));
        }
        
        if request_lower.contains("what does") && (request_lower.contains("app do") || request_lower.contains("application do")) {
            // Show immediate understanding rather than confusion
            return Ok(Some(
                "I can analyze the application structure to understand its purpose. Let me examine the source code and provide you with a clear explanation of what this application does.".to_string()
            ));
        }
        
        // Check if this is a file listing request that got confused
        if request_lower.contains("what") && request_lower.contains("src") {
            return Ok(Some(
                "I can examine the src folder structure to understand what files and functionality are present. Let me analyze the codebase for you.".to_string()
            ));
        }
        
        Ok(None)
    }
    
    /// Smart file operation intent detection for ambiguous requests
    fn detect_file_operation_intent(&self, request: &str) -> Option<FileOperationIntent> {
        // Extract potential file paths from the request
        let file_path = self.extract_file_path(request)?;
        
        // Analyze the request for intent keywords
        let create_indicators = ["make a new", "create a new", "new html", "new file"];
        let edit_indicators = ["remake this", "edit this", "modify this", "update this"];
        let ambiguous_indicators = ["can you make", "improve", "fix"];
        
        let has_create = create_indicators.iter().any(|&indicator| request.contains(indicator));
        let has_edit = edit_indicators.iter().any(|&indicator| request.contains(indicator));
        let has_ambiguous = ambiguous_indicators.iter().any(|&indicator| request.contains(indicator));
        
        // Extract description/requirements
        let description = self.extract_description(request);
        
        if has_create && !has_edit {
            Some(FileOperationIntent::CreateNew { 
                path: file_path, 
                description 
            })
        } else if has_edit && !has_create {
            Some(FileOperationIntent::EditExisting { 
                path: file_path, 
                changes: description 
            })
        } else if has_ambiguous || (has_create && has_edit) {
            // Let the system decide based on file existence
            Some(FileOperationIntent::CreateOrEdit { 
                path: file_path, 
                description 
            })
        } else {
            None
        }
    }
    
    /// Extract file path from request text
    fn extract_file_path(&self, request: &str) -> Option<String> {
        // Look for quoted paths first
        if let Some(start) = request.find('"') {
            if let Some(end) = request[start + 1..].find('"') {
                return Some(request[start + 1..start + 1 + end].to_string());
            }
        }
        
        // Look for common file extensions
        let extensions = [".html", ".js", ".css", ".py", ".rs", ".txt"];
        for ext in extensions {
            if let Some(pos) = request.find(ext) {
                // Find the start of the filename (look backwards for space or start)
                let start = request[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
                let end = pos + ext.len();
                return Some(request[start..end].trim().to_string());
            }
        }
        
        None
    }
    
    /// Extract description/requirements from request
    fn extract_description(&self, request: &str) -> String {
        // Simple extraction - look for phrases after "so it" or "to"
        if let Some(pos) = request.find("so it") {
            return request[pos + 5..].trim().to_string();
        }
        if let Some(pos) = request.find(" to ") {
            return request[pos + 4..].trim().to_string();
        }
        
        // Fallback: return the request itself
        request.to_string()
    }
    
    /// Analyze ambiguous request and generate intelligent response
    pub async fn reason_about_request(&self, request: &str, context: &str) -> Result<ReasoningResult> {
        // Direct printing to stdout interferes with the TUI in chat.rs.
        // This reasoning information should be passed up to the UI layer to be displayed
        // within the ratatui framework, for example, in the status bar.
        
        let request_lower = request.to_lowercase();
        let _context_lower = context.to_lowercase();
        
        // Detect the type of ambiguous request
        let reasoning_result = if self.is_domain_request(&request_lower) {
            self.reason_about_domains(&request_lower, context).await?
        } else if self.is_improvement_request(&request_lower) {
            self.reason_about_code_improvement(&request_lower, context).await?
        } else if self.is_selection_request(&request_lower) {
            self.reason_about_selection(&request_lower, context).await?
        } else if self.is_analysis_request(&request_lower) {
            self.reason_about_analysis(&request_lower, context).await?
        } else {
            self.reason_about_generic_request(&request_lower, context).await?
        };

        Ok(reasoning_result)
    }

    /// Apply reasoning result to actual data
    pub async fn apply_reasoning<T>(&self, _reasoning: &ReasoningResult, data: Vec<T>) -> Result<Vec<T>> 
    where 
        T: Clone + std::fmt::Debug
    {
        // This would apply the generated criteria to filter/sort the data
        // For now, return first few items as a basic implementation
        let limit = 10.min(data.len());
        Ok(data.into_iter().take(limit).collect())
    }

    /// Analyze if I'm capable of handling this request based on available functions
    pub async fn assess_my_capabilities(&self, request: &str, available_functions: &[String]) -> Result<CapabilityAssessment> {
        let functions_list = available_functions.join(", ");
        
        let prompt = format!(
            r#"I am an AI assistant trying to determine if I can handle a user request.

USER REQUEST: {}

MY AVAILABLE FUNCTIONS: {}

I need to honestly assess my capabilities. I should be confident and experimental - I can combine functions creatively to solve complex problems. But I should also be honest about genuine limitations.

Think through this step by step:
1. What does the user want me to do?
2. Do I have functions that could accomplish this directly or creatively combined?  
3. Should I attempt this task or is it genuinely impossible for me?
4. If I should attempt it, what approach should I try first?

Respond in JSON format:
{{
    "can_attempt": true/false,
    "confidence_level": 1-10,
    "reasoning": "why I think I can or cannot do this",
    "suggested_approach": "what I should try first",
    "experimental": true/false (if this requires creative function combination)
}}"#,
            request, functions_list
        );

        let response = crate::gemini::query_gemini(&prompt, &self.config).await?;
        
        // More robust JSON cleaning and parsing
        let mut cleaned_response = response.trim();
        
        // Remove various markdown code block formats
        if cleaned_response.starts_with("```json") {
            cleaned_response = cleaned_response.strip_prefix("```json").unwrap_or(cleaned_response);
        } else if cleaned_response.starts_with("```") {
            cleaned_response = cleaned_response.strip_prefix("```").unwrap_or(cleaned_response);
        }
        
        if cleaned_response.ends_with("```") {
            cleaned_response = cleaned_response.strip_suffix("```").unwrap_or(cleaned_response);
        }
        
        cleaned_response = cleaned_response.trim();
        
        // Try to find JSON within the response if direct parsing fails
        let assessment: serde_json::Value = match serde_json::from_str(cleaned_response) {
            Ok(val) => val,
            Err(_) => {
                // Look for JSON object within the text
                if let Some(start) = cleaned_response.find('{') {
                    if let Some(end) = cleaned_response.rfind('}') {
                        let json_part = &cleaned_response[start..=end];
                        serde_json::from_str(json_part)
                            .map_err(|e| anyhow::anyhow!("Failed to parse capability assessment JSON: {}. Original response: {}", e, response))?
                    } else {
                        return Err(anyhow::anyhow!("No valid JSON found in capability assessment response: {}", response));
                    }
                } else {
                    return Err(anyhow::anyhow!("No JSON object found in capability assessment response: {}", response));
                }
            }
        };
        
        Ok(CapabilityAssessment {
            can_attempt: assessment.get("can_attempt").and_then(|v| v.as_bool()).unwrap_or(false),
            confidence_level: assessment.get("confidence_level").and_then(|v| v.as_u64()).unwrap_or(1) as u8,
            reasoning: assessment.get("reasoning").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string(),
            suggested_approach: assessment.get("suggested_approach").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string(),
            experimental: assessment.get("experimental").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    }

    /// Check if request is about domains
    fn is_domain_request(&self, request: &str) -> bool {
        let domain_indicators = ["domain", "domains", "website", "site", "brandable", "web", "url"];
        let evaluation_terms = ["best", "good", "valuable", "top", "select", "choose", "pick"];
        
        let has_domain_term = domain_indicators.iter().any(|&term| request.contains(term));
        let has_eval_term = evaluation_terms.iter().any(|&term| request.contains(term));
        
        has_domain_term && has_eval_term
    }

    /// Check if request is about code improvement
    fn is_improvement_request(&self, request: &str) -> bool {
        let improvement_indicators = [
            "improve", "better", "optimize", "fix", "enhance", "refactor", 
            "clean", "modernize", "upgrade", "polish"
        ];
        let code_indicators = [
            "code", "script", "function", "file", "program", "implementation"
        ];
        
        let has_improvement = improvement_indicators.iter().any(|&term| request.contains(term));
        let has_code = code_indicators.iter().any(|&term| request.contains(term));
        
        has_improvement && has_code
    }

    /// Check if request is about selection/filtering
    fn is_selection_request(&self, request: &str) -> bool {
        let selection_terms = [
            "best", "top", "select", "choose", "pick", "find", "get", 
            "show me", "give me", "list", "filter"
        ];
        
        selection_terms.iter().any(|&term| request.contains(term))
    }

    /// Check if request is about analysis
    fn is_analysis_request(&self, request: &str) -> bool {
        let analysis_terms = [
            "analyze", "analysis", "evaluate", "assess", "review", 
            "examine", "study", "investigate", "what", "how", "why"
        ];
        
        analysis_terms.iter().any(|&term| request.contains(term))
    }

    /// Generate reasoning for domain-related requests
    async fn reason_about_domains(&self, request: &str, context: &str) -> Result<ReasoningResult> {
        // Direct printing interferes with the TUI. This status should be communicated
        // to the UI layer to be displayed in the status bar.
        // Example: PersistentStatusBar::update_status("Reasoning about domain criteria...");
        
        let prompt = format!(
            "The user wants to evaluate domains with this request: '{}'\n\
            Context: {}\n\n\
            Generate specific, actionable criteria for evaluating domains. Consider:\n\
            - Business value and brandability\n\
            - Technical factors (length, TLD, etc.)\n\
            - Market considerations\n\
            - User intent and industry\n\n\
            Provide a JSON response with:\n\
            {{\n\
                \"interpretation\": \"What the user likely wants\",\n\
                \"criteria\": [\"criterion1\", \"criterion2\", ...],\n\
                \"questions\": [\"clarifying question1\", ...],\n\
                \"confidence\": 0.8\n\
            }}",
            request, context
        );

        let response = gemini::query_gemini(&prompt, &self.config).await?;
        self.parse_reasoning_response(&response, "domain evaluation").await
    }

    /// Generate reasoning for code improvement requests
    async fn reason_about_code_improvement(&self, request: &str, context: &str) -> Result<ReasoningResult> {
        // Direct printing interferes with the TUI. This status should be communicated
        // to the UI layer to be displayed in the status bar.
        // Example: PersistentStatusBar::update_status("Reasoning about code improvement...");

        // Use heuristic patterns to suggest improvements
        let matching_patterns = self.heuristic_database.code_improvement_patterns.iter()
            .filter(|pattern| request.contains(&pattern.pattern))
            .collect::<Vec<_>>();

        let mut suggested_improvements = Vec::new();
        for pattern in &matching_patterns {
            suggested_improvements.extend(pattern.suggested_improvements.clone());
        }

        if suggested_improvements.is_empty() {
            // Fallback to AI reasoning
            let prompt = format!(
                "The user wants to improve code with this request: '{}'\n\
                Context: {}\n\n\
                What specific improvements should be made? Consider:\n\
                - Performance optimizations\n\
                - Code readability and maintainability\n\
                - Error handling and robustness\n\
                - Modern best practices\n\
                - Security considerations\n\n\
                Provide specific, actionable improvement suggestions.",
                request, context
            );

            let response = gemini::query_gemini(&prompt, &self.config).await?;
            suggested_improvements = response.lines()
                .map(|line| line.trim().to_string())
                .filter(|line| !line.is_empty())
                .take(5)
                .collect();
        }

        Ok(ReasoningResult {
            interpretation: format!("Code improvement request focusing on: {}", 
                matching_patterns.iter()
                    .map(|p| p.description.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")),
            suggested_criteria: suggested_improvements.clone(),
            clarification_questions: vec![
                "What aspect should I prioritize: performance, readability, or robustness?".to_string(),
                "Should I maintain backward compatibility?".to_string(),
                "Are there specific coding standards to follow?".to_string(),
            ],
            confidence: 0.85,
            actionable_plan: suggested_improvements.clone(),
            capability_assessment: vec![
                "I can read and analyze code files".to_string(),
                "I can suggest specific improvements".to_string(),
                "I can apply changes if given specific instructions".to_string(),
            ],
            metacognitive_notes: "I should focus on actionable improvements rather than asking questions".to_string(),
        })
    }

    /// Generate reasoning for selection requests
    async fn reason_about_selection(&self, request: &str, context: &str) -> Result<ReasoningResult> {
        // Direct printing interferes with the TUI. This status should be communicated
        // to the UI layer to be displayed in the status bar.
        // Example: PersistentStatusBar::update_status("Reasoning about selection criteria...");

        let prompt = format!(
            "The user wants to select/filter items with: '{}'\n\
            Context: {}\n\n\
            What are the most logical selection criteria? Consider:\n\
            - Quality indicators\n\
            - Relevance factors\n\
            - User preferences\n\
            - Practical constraints\n\n\
            Provide specific, measurable criteria for selection.",
            request, context
        );

        let response = gemini::query_gemini(&prompt, &self.config).await?;
        self.parse_reasoning_response(&response, "selection").await
    }

    /// Generate reasoning for analysis requests
    async fn reason_about_analysis(&self, request: &str, context: &str) -> Result<ReasoningResult> {
        // Direct printing interferes with the TUI. This status should be communicated
        // to the UI layer to be displayed in the status bar.
        // Example: PersistentStatusBar::update_status("Reasoning about analysis approach...");

        let prompt = format!(
            "The user wants analysis with: '{}'\n\
            Context: {}\n\n\
            What should be analyzed and how? Consider:\n\
            - Key metrics and indicators\n\
            - Comparative analysis\n\
            - Trends and patterns\n\
            - Actionable insights\n\n\
            Provide a structured analysis approach.",
            request, context
        );

        let response = gemini::query_gemini(&prompt, &self.config).await?;
        self.parse_reasoning_response(&response, "analysis").await
    }

    /// Fallback for generic ambiguous requests
    async fn reason_about_generic_request(&self, request: &str, context: &str) -> Result<ReasoningResult> {
        // Direct printing interferes with the TUI. This status should be communicated
        // to the UI layer to be displayed in the status bar.
        // Example: PersistentStatusBar::update_status("Reasoning about ambiguous request...");

        let prompt = format!(
            "The user made this ambiguous request: '{}'\n\
            Context: {}\n\n\
            Help clarify what they might want by:\n\
            1. Interpreting the most likely intent\n\
            2. Suggesting specific criteria or parameters\n\
            3. Asking clarifying questions\n\
            4. Providing an actionable plan\n\n\
            Be specific and helpful in your suggestions.",
            request, context
        );

        let response = gemini::query_gemini(&prompt, &self.config).await?;
        self.parse_reasoning_response(&response, "general").await
    }

    /// Parse AI response into structured reasoning result
    async fn parse_reasoning_response(&self, response: &str, request_type: &str) -> Result<ReasoningResult> {
        // Try to parse as JSON first
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(response) {
            let interpretation = json_value.get("interpretation")
                .and_then(|v| v.as_str())
                .unwrap_or("Analyzing request...")
                .to_string();

            let criteria = json_value.get("criteria")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect())
                .unwrap_or_else(Vec::new);

            let questions = json_value.get("questions")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect())
                .unwrap_or_else(Vec::new);

            let confidence = json_value.get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.7) as f32;

            return Ok(ReasoningResult {
                interpretation,
                suggested_criteria: criteria.clone(),
                clarification_questions: questions,
                confidence,
                actionable_plan: criteria.clone(),
                capability_assessment: vec![
                    "I can analyze and interpret requests".to_string(),
                    "I can break down complex tasks".to_string(),
                ],
                metacognitive_notes: "Parsed structured response successfully".to_string(),
            });
        }

        // Fallback to text parsing
        let lines: Vec<&str> = response.lines().collect();
        let criteria: Vec<String> = lines.iter()
            .filter(|line| line.starts_with('-') || line.starts_with('•') || line.starts_with('*'))
            .map(|line| line.trim_start_matches(['-', '•', '*']).trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();

        Ok(ReasoningResult {
            interpretation: format!("AI interpretation of {} request", request_type),
            suggested_criteria: criteria.clone(),
            clarification_questions: vec![
                "Could you be more specific about your requirements?".to_string(),
                "What criteria are most important to you?".to_string(),
            ],
            confidence: 0.6,
            actionable_plan: criteria.clone(),
            capability_assessment: vec![
                "I can process text and provide analysis".to_string(),
                "I can suggest criteria and approaches".to_string(),
            ],
            metacognitive_notes: "Fallback text parsing used".to_string(),
        })
    }

    /// Present reasoning results to user for confirmation
    pub fn present_reasoning(&self, reasoning: &ReasoningResult) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("{}🧠 **Reasoning Analysis**{}\n", EMERALD_BRIGHT, RESET));
        output.push_str(&format!("**Interpretation**: {}\n\n", reasoning.interpretation));
        
        if !reasoning.suggested_criteria.is_empty() {
            output.push_str("**Suggested Criteria**:\n");
            for (i, criterion) in reasoning.suggested_criteria.iter().enumerate() {
                output.push_str(&format!("{}{}. {}{}\n", BLUE_BRIGHT, i + 1, criterion, RESET));
            }
            output.push('\n');
        }

        if !reasoning.clarification_questions.is_empty() {
            output.push_str("**Clarifying Questions**:\n");
            for question in &reasoning.clarification_questions {
                output.push_str(&format!("{}• {}{}\n", GRAY_DIM, question, RESET));
            }
            output.push('\n');
        }

        output.push_str(&format!("**Confidence**: {:.0}%\n\n", reasoning.confidence * 100.0));
        
        output.push_str(&format!("{}Would you like me to proceed with these criteria, or would you like to specify different ones?{}", GRAY_DIM, RESET));

        output
    }
}

impl HeuristicDatabase {
    fn new() -> Self {
        Self {
            domain_criteria: Self::init_domain_criteria(),
            code_improvement_patterns: Self::init_code_patterns(),
            data_filter_templates: Self::init_filter_templates(),
        }
    }

    fn init_domain_criteria() -> Vec<DomainCriterion> {
        vec![
            DomainCriterion {
                name: "Short Length".to_string(),
                description: "Prefer domains under 12 characters".to_string(),
                weight: 0.8,
                evaluation_fn: DomainEvalFunction::Length,
            },
            DomainCriterion {
                name: "Brandability".to_string(),
                description: "Easy to remember and pronounce".to_string(),
                weight: 0.9,
                evaluation_fn: DomainEvalFunction::Brandability,
            },
            DomainCriterion {
                name: ".com TLD".to_string(),
                description: "Prefer .com domains for commercial value".to_string(),
                weight: 0.7,
                evaluation_fn: DomainEvalFunction::TldValue,
            },
        ]
    }

    fn init_code_patterns() -> Vec<CodeImprovementPattern> {
        vec![
            CodeImprovementPattern {
                pattern: "improve".to_string(),
                description: "general code enhancement".to_string(),
                suggested_improvements: vec![
                    "Add error handling".to_string(),
                    "Improve variable naming".to_string(),
                    "Add documentation".to_string(),
                    "Optimize performance".to_string(),
                ],
                priority: 1,
            },
            CodeImprovementPattern {
                pattern: "optimize".to_string(),
                description: "performance optimization".to_string(),
                suggested_improvements: vec![
                    "Reduce algorithmic complexity".to_string(),
                    "Cache repeated calculations".to_string(),
                    "Use more efficient data structures".to_string(),
                    "Minimize memory allocations".to_string(),
                ],
                priority: 2,
            },
        ]
    }

    fn init_filter_templates() -> HashMap<String, FilterTemplate> {
        let mut templates = HashMap::new();
        
        templates.insert("best_items".to_string(), FilterTemplate {
            name: "Best Items Filter".to_string(),
            description: "General filter for selecting best items".to_string(),
            criteria: vec![
                FilterCriterion {
                    field: "quality_score".to_string(),
                    operator: FilterOperator::Score,
                    value: FilterValue::Number(0.8),
                    weight: 1.0,
                },
            ],
            default_limit: 10,
        });

        templates
    }
}

/// Helper function to create a reasoning engine and process ambiguous requests
/// Reason about code editing failures and user frustration  
pub async fn reason_about_edit_failure(
    request: &str,
    current_content: &str, 
    user_feedback: &str,
    config: &Config
) -> Result<EditFailureReasoning> {
    // Use status bar instead of println to avoid UI scrambling
    crate::thinking_display::PersistentStatusBar::set_ai_thinking("🤔 Analyzing edit failure and user feedback");
    
    let reasoning = EditFailureReasoning::analyze(request, current_content, user_feedback, config).await?;
    
    // Show the reasoning process via status bar
    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!(
        "Assessment: {} | Issue: {} | Approach: {}", 
        reasoning.self_assessment, reasoning.likely_problem, reasoning.recommended_approach
    ));
    
    Ok(reasoning)
}

/// Reasoning about edit failures
#[derive(Debug, Clone)]
pub struct EditFailureReasoning {
    pub self_assessment: String,
    pub likely_problem: String, 
    pub recommended_approach: String,
    pub user_frustration_level: u8, // 1-10
    pub should_try_different_approach: bool,
}

/// Assessment of my capabilities for a given request
#[derive(Debug, Clone)]
pub struct CapabilityAssessment {
    pub can_attempt: bool,
    pub confidence_level: u8, // 1-10
    pub reasoning: String,
    pub suggested_approach: String,
    pub experimental: bool, // if this requires creative function combination
}

/// File operation intent detected from ambiguous requests
#[derive(Debug, Clone)]
enum FileOperationIntent {
    CreateNew { path: String, description: String },
    EditExisting { path: String, changes: String },
    CreateOrEdit { path: String, description: String },
}

impl EditFailureReasoning {
    /// Analyze an edit failure using an LLM for more generic and powerful reasoning.
    pub async fn analyze(request: &str, current_content: &str, user_feedback: &str, config: &Config) -> Result<Self> {
        let prompt = format!(
            r#"An AI code editing task failed. Analyze the situation to determine the cause and recommend a new approach.

            **Original User Request:**
            "{}"

            **User's Feedback on the Result:**
            "{}"

            **The (failed) Code Generated by the AI (first 2000 chars):**
            ```
            {}
            ```

            **Analysis Task:**
            1.  **Self-Assessment:** In one sentence, what did the AI likely do wrong? (e.g., "I misunderstood the interactivity requirement," "I generated incomplete code.").
            2.  **Likely Problem:** Based on the request and feedback, what is the most likely technical problem with the generated code? (e.g., "The HTML is static and lacks JavaScript for interaction," "The logic is flawed and doesn't handle edge cases.").
            3.  **Recommended Approach:** What is the best strategy to fix this? (e.g., "Rewrite the code from scratch with a focus on interactivity," "Add the missing JavaScript logic for the core features.").
            4.  **User Frustration Level:** On a scale of 1-10, how frustrated is the user?

            **Respond with ONLY a JSON object in the following format:**
            {{
                "self_assessment": "...",
                "likely_problem": "...",
                "recommended_approach": "...",
                "user_frustration_level": 8
            }}"#,
            request, user_feedback, current_content.chars().take(2000).collect::<String>()
        );

        let response_str = gemini::query_gemini(&prompt, config).await?;
        
        // Parse the JSON response, cleaning it first
        let cleaned_response = response_str.trim().strip_prefix("```json").unwrap_or(&response_str).strip_suffix("```").unwrap_or(&response_str).trim();
        let parsed: serde_json::Value = serde_json::from_str(cleaned_response)
            .map_err(|e| anyhow::anyhow!("Failed to parse reasoning response from AI: {}. Response was: {}", e, cleaned_response))?;

        let self_assessment = parsed.get("self_assessment").and_then(|v| v.as_str()).unwrap_or("Assessment failed").to_string();
        let likely_problem = parsed.get("likely_problem").and_then(|v| v.as_str()).unwrap_or("Problem unclear").to_string();
        let recommended_approach = parsed.get("recommended_approach").and_then(|v| v.as_str()).unwrap_or("Retry with original instructions").to_string();
        let frustration_level = parsed.get("user_frustration_level").and_then(|v| v.as_u64()).unwrap_or(5) as u8;

        Ok(Self {
            self_assessment,
            likely_problem,
            recommended_approach,
            user_frustration_level: frustration_level,
            should_try_different_approach: frustration_level >= 6,
        })
    }
}

pub async fn handle_ambiguous_request(request: &str, context: &str, config: &Config) -> Result<String> {
    let engine = ReasoningEngine::new(config);
    
    // Enhanced metacognitive reasoning with capability assessment
    let reasoning = engine.reason_about_request_with_metacognition(request, context).await?;
    
    // Check if we can handle this internally first (metacognitive self-assessment)
    if let Some(internal_response) = engine.try_internal_resolution(request, context, &reasoning).await? {
        return Ok(internal_response);
    }
    
    // If we need external reasoning, be more intelligent about it
    let action_prompt = format!(
        "The user made this request: '{}'\n\nContext: {}\n\nMy metacognitive analysis suggests: {}\n\nI have these capabilities: {}\n\nBased on this self-assessment, I will: {}\n\nTaking action now:",
        request, context, reasoning.interpretation, 
        reasoning.capability_assessment.join(", "),
        reasoning.actionable_plan.join(", ")
    );
    
    // Use Gemini to take action based on the enhanced reasoning
    let action_response = crate::gemini::query_gemini(&action_prompt, config).await?;
    
    Ok(action_response)
}