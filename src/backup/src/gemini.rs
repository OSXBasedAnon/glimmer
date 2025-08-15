use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::config::Config;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

/// Smart rate limiter to prevent API overload and optimize performance
#[derive(Debug)]
struct RateLimiter {
    last_request_times: VecDeque<Instant>,
    max_requests_per_minute: u32,
    min_request_interval: Duration,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            last_request_times: VecDeque::new(),
            max_requests_per_minute: 60, // Conservative limit
            min_request_interval: Duration::from_millis(100), // Minimum 100ms between requests
        }
    }
    
    /// Check if we can make a request and wait if necessary
    async fn check_and_wait(&mut self) {
        if let Some(wait_duration) = self.get_wait_duration() {
            tokio::time::sleep(wait_duration).await;
        }
        self.record_request();
    }
    
    /// Get the required wait duration without async
    fn get_wait_duration(&mut self) -> Option<Duration> {
        let now = Instant::now();
        
        // Remove requests older than 1 minute
        while let Some(&front_time) = self.last_request_times.front() {
            if now.duration_since(front_time).as_secs() > 60 {
                self.last_request_times.pop_front();
            } else {
                break;
            }
        }
        
        // Check if we're at the rate limit
        let mut max_wait = Duration::from_millis(0);
        
        if self.last_request_times.len() >= self.max_requests_per_minute as usize {
            if let Some(&front_time) = self.last_request_times.front() {
                let wait_time = Duration::from_secs(60) - now.duration_since(front_time);
                if wait_time > Duration::from_millis(0) {
                    max_wait = max_wait.max(wait_time);
                }
            }
        }
        
        // Check minimum interval
        if let Some(&last_time) = self.last_request_times.back() {
            let elapsed = now.duration_since(last_time);
            if elapsed < self.min_request_interval {
                max_wait = max_wait.max(self.min_request_interval - elapsed);
            }
        }
        
        if max_wait > Duration::from_millis(0) {
            Some(max_wait)
        } else {
            None
        }
    }
    
    /// Record that a request was made
    fn record_request(&mut self) {
        self.last_request_times.push_back(Instant::now());
    }
}

lazy_static::lazy_static! {
    static ref RATE_LIMITER: Arc<Mutex<RateLimiter>> = Arc::new(Mutex::new(RateLimiter::new()));
    static ref HTTP_CLIENT: Client = Client::builder()
        .timeout(Duration::from_secs(60)) // Default timeout, will be overridden per request
        .pool_max_idle_per_host(10) // Keep connections alive for reuse
        .pool_idle_timeout(Duration::from_secs(300)) // Keep idle connections for 5 minutes
        .connection_verbose(false)
        .build()
        .expect("Failed to create HTTP client");
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiRequest {
    contents: Vec<Content>,
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum Part {
    Text { text: String },
    FunctionCall { 
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall 
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GenerationConfig {
    temperature: f32,
    max_output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AdvancedGeminiRequest {
    contents: Vec<Content>,
    generation_config: GenerationConfig,
    system_instruction: Option<SystemInstruction>,
    tools: Option<Vec<Tool>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Tool {
    function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: u32,
    #[serde(rename = "totalTokenCount")]
    total_token_count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Candidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_ratings: Option<Vec<SafetyRating>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    citation_metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grounding_attributions: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs_result: Option<LogprobsResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SafetyRating {
    category: String,
    probability: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LogprobsResult {
    top_candidates: Option<Vec<TopCandidate>>,
    chosen_candidates: Option<Vec<TopCandidate>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TopCandidate {
    token: String,
    log_probability: f64,
}

// Token counting structures
#[derive(Debug, Serialize, Deserialize)]
struct CountTokensRequest {
    contents: Vec<Content>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CountTokensResponse {
    total_tokens: u32,
}

// Token usage information
#[derive(Debug, Clone)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: Option<u32>,
    pub total_tokens: u32,
}

// Confidence assessment for AI responses
#[derive(Debug, Clone)]
pub struct ConfidenceAssessment {
    pub score: f64,           // 0-100 confidence percentage
    pub level: ConfidenceLevel,
    pub uncertainty: f64,     // Variance/uncertainty measure
    pub token_count: usize,   // Number of tokens analyzed
}

#[derive(Debug, Clone)]
pub enum ConfidenceLevel {
    VeryHigh,  // 90-100%
    High,      // 70-89% 
    Medium,    // 50-69%
    Low,       // 30-49%
    VeryLow,   // <30%
}

impl std::fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfidenceLevel::VeryHigh => write!(f, "Very High"),
            ConfidenceLevel::High => write!(f, "High"),
            ConfidenceLevel::Medium => write!(f, "Medium"),
            ConfidenceLevel::Low => write!(f, "Low"),
            ConfidenceLevel::VeryLow => write!(f, "Very Low"),
        }
    }
}

pub async fn edit_code(
    original_code: &str,
    query: &str,
    language: &str,
    config: &Config,
) -> Result<String> {
    let api_key = match config.gemini.api_key.as_ref() {
        Some(key) if !key.trim().is_empty() => key,
        _ => {
            return Err(anyhow::anyhow!(
                "Gemini API key not configured or empty. Please set GEMINI_API_KEY in:\n\
                 1. Environment variable: export GEMINI_API_KEY=your_key\n\
                 2. .env file in current directory: GEMINI_API_KEY=your_key\n\
                 3. Config file at ~/.config/glimmer/config.toml"
            ));
        }
    };

    // Dynamic timeout based on request complexity
    let is_simple_change = query.len() < 100 && (
        query.to_lowercase().contains("rename") ||
        query.to_lowercase().contains("change") && query.to_lowercase().contains("title")
    );
    let timeout_secs = if is_simple_change { 15 } else { config.gemini.timeout_seconds.min(45) };
    
    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .context("Failed to create HTTP client")?;

    let prompt = create_edit_prompt(original_code, query, language);

    let request = GeminiRequest {
        contents: vec![Content {
            parts: vec![Part::Text {
                text: prompt,
            }],
        }],
        generation_config: GenerationConfig {
            temperature: 0.3,
            max_output_tokens: 8192, // Increased from 4096 to prevent truncation
            response_logprobs: Some(true),
            logprobs: None,
        },
    };

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        config.gemini.default_model, api_key
    );

    let mut last_error = None;
    
    for attempt in 1..=config.gemini.max_retries {
        match make_request(&client, &url, &request).await {
            Ok(response) => {
                return extract_code_from_response(&response, original_code);
            }
            Err(e) => {
                last_error = Some(anyhow::anyhow!(e.to_string()));
                
                // Check for specific error types
                let error_msg = e.to_string();
                if error_msg.contains("timeout") {
                    eprintln!("Request timed out (attempt {}/{}). Retrying with longer timeout...", attempt, config.gemini.max_retries);
                } else if error_msg.contains("401") || error_msg.contains("403") {
                    return Err(anyhow::anyhow!("Authentication failed. Please check your Gemini API key."));
                } else if error_msg.contains("429") {
                    eprintln!("Rate limit exceeded (attempt {}/{}). Waiting longer before retry...", attempt, config.gemini.max_retries);
                    tokio::time::sleep(Duration::from_secs(5)).await;
                } else if error_msg.contains("network") || error_msg.contains("connection") {
                    eprintln!("Network error (attempt {}/{}). Retrying...", attempt, config.gemini.max_retries);
                } else {
                    eprintln!("API error (attempt {}/{}): {}", attempt, config.gemini.max_retries, error_msg);
                }
                
                if attempt < config.gemini.max_retries {
                    tokio::time::sleep(Duration::from_millis(1000 * attempt as u64)).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All retry attempts failed")))
}

async fn make_request(
    client: &Client,
    url: &str,
    request: &GeminiRequest,
) -> Result<GeminiResponse> {
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .json(request)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                anyhow::anyhow!("Request timed out. The Gemini API may be slow or overloaded.")
            } else if e.is_connect() {
                anyhow::anyhow!("Could not connect to Gemini API. Please check your internet connection.")
            } else if e.is_request() {
                anyhow::anyhow!("Request failed: {}. Please check your API key and try again.", e)
            } else {
                anyhow::anyhow!("Network error communicating with Gemini API: {}", e)
            }
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        
        let error_message = match status.as_u16() {
            400 => format!("Bad request (400): The request was malformed. {}", error_text),
            401 => "Authentication failed (401): Invalid or missing API key. Please check your GEMINI_API_KEY.".to_string(),
            403 => "Access forbidden (403): API key may not have required permissions.".to_string(),
            404 => "Not found (404): The requested model or endpoint was not found.".to_string(),
            429 => "Rate limit exceeded (429): Too many requests. Please wait and try again.".to_string(),
            500 => "Internal server error (500): Gemini API is experiencing issues.".to_string(),
            503 => "Service unavailable (503): Gemini API is temporarily unavailable.".to_string(),
            _ => format!("Gemini API error ({}): {}", status, error_text),
        };
        
        return Err(anyhow::anyhow!("{}", error_message));
    }

    // First get the raw response text for better error handling
    let response_text = response
        .text()
        .await
        .context("Failed to read Gemini API response")?;
    
    // Try to parse the response, with fallback handling
    match serde_json::from_str::<GeminiResponse>(&response_text) {
        Ok(response) => Ok(response),
        Err(_parse_error) => {
            // Try to extract response using fallback parsing
            match try_fallback_parsing(&response_text) {
                Some(fallback_response) => Ok(fallback_response),
                None => {
                    let _preview = if response_text.len() > 200 {
                        format!("{}...", &response_text[..200])
                    } else {
                        response_text.clone()
                    };
                    // Final fallback: try to extract any useful text from the response
                    if let Some(extracted_text) = extract_text_from_response(&response_text) {
                        Ok(GeminiResponse {
                            candidates: vec![Candidate {
                                content: Some(Content {
                                    parts: vec![Part::Text {
                                        text: extracted_text,
                                    }],
                                }),
                                finish_reason: None,
                                safety_ratings: None,
                                citation_metadata: None,
                                token_count: None,
                                grounding_attributions: None,
                                logprobs_result: None,
                            }],
                            usage_metadata: None,
                        })
                    } else {
                        Err(anyhow::anyhow!(
                            "Unable to parse Gemini API response. Response preview: {}", 
                            if response_text.len() > 500 { format!("{}...", &response_text[..500]) } else { response_text.clone() }
                        ))
                    }
                }
            }
        }
    }
}

/// Make request with ESC interrupt support
async fn make_request_with_interrupt(
    client: &Client,
    url: &str,
    request: &GeminiRequest,
) -> Result<GeminiResponse> {
    // Create the request future
    let request_future = client
        .post(url)
        .header("Content-Type", "application/json")
        .json(request)
        .send();
    
    // Poll both the request and ESC key checking
    let response = tokio::select! {
        result = request_future => {
            result.map_err(|e| {
                // Provide more specific error information
                if e.is_timeout() {
                    anyhow::anyhow!("Gemini API request timed out")
                } else if e.is_connect() {
                    anyhow::anyhow!("Failed to connect to Gemini API (network error): {}", e)
                } else if e.status() == Some(reqwest::StatusCode::TOO_MANY_REQUESTS) {
                    anyhow::anyhow!("Gemini API rate limit exceeded - please wait a moment and try again")
                } else if e.status() == Some(reqwest::StatusCode::UNAUTHORIZED) {
                    anyhow::anyhow!("Gemini API authentication failed - please check your API key")
                } else {
                    anyhow::anyhow!("Failed to send request to Gemini API: {}", e)
                }
            })?
        }
        _ = check_for_interrupt_during_request() => {
            return Err(anyhow::anyhow!("Request interrupted by ESC key"));
        }
    };
    
    parse_gemini_response(response).await
}

/// Check for ESC interrupts during long requests
async fn check_for_interrupt_during_request() {
    loop {
        if crate::input_handler::check_for_escape_key().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Parse Gemini response with better error handling
async fn parse_gemini_response(response: reqwest::Response) -> Result<GeminiResponse> {
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("Gemini API error {}: {}", status, error_text));
    }

    // First get the raw response text for better error handling
    let response_text = response
        .text()
        .await
        .context("Failed to read Gemini API response")?;
    
    // Try to parse the response, with fallback handling
    match serde_json::from_str::<GeminiResponse>(&response_text) {
        Ok(response) => Ok(response),
        Err(_parse_error) => {
            // Try to extract response using fallback parsing
            match try_fallback_parsing(&response_text) {
                Some(fallback_response) => Ok(fallback_response),
                None => {
                    let _preview = if response_text.len() > 200 {
                        format!("{}...", &response_text[..200])
                    } else {
                        response_text.clone()
                    };
                    // Final fallback: try to extract any useful text from the response
                    if let Some(extracted_text) = extract_text_from_response(&response_text) {
                        Ok(GeminiResponse {
                            candidates: vec![Candidate {
                                content: Some(Content {
                                    parts: vec![Part::Text {
                                        text: extracted_text,
                                    }],
                                }),
                                finish_reason: None,
                                safety_ratings: None,
                                citation_metadata: None,
                                token_count: None,
                                grounding_attributions: None,
                                logprobs_result: None,
                            }],
                            usage_metadata: None,
                        })
                    } else {
                        Err(anyhow::anyhow!(
                            "Unable to parse Gemini API response. Response preview: {}", 
                            if response_text.len() > 500 { format!("{}...", &response_text[..500]) } else { response_text.clone() }
                        ))
                    }
                }
            }
        }
    }
}

/// Extract any readable text from malformed response as last resort
fn extract_text_from_response(response_text: &str) -> Option<String> {
    // Strategy 1: Look for any text field in the JSON
    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(response_text) {
        if let Some(text) = find_text_in_json(&json_value) {
            if !text.trim().is_empty() && text.len() > 10 {
                return Some(text);
            }
        }
    }
    
    // Strategy 2: Look for quoted strings that might be responses
    let text_patterns = [
        r#""text"\s*:\s*"([^"]+)""#,
        r#""content"\s*:\s*"([^"]+)""#,
        r#""response"\s*:\s*"([^"]+)""#,
    ];
    
    for pattern in &text_patterns {
        if let Ok(regex) = regex::Regex::new(pattern) {
            if let Some(captures) = regex.captures(response_text) {
                if let Some(matched) = captures.get(1) {
                    let text = matched.as_str();
                    if text.len() > 10 {
                        return Some(text.to_string());
                    }
                }
            }
        }
    }
    
    None
}

/// Recursively search for text values in JSON
fn find_text_in_json(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) if s.len() > 10 => {
            // Only skip obvious error patterns, not legitimate content
            let obvious_errors = [
                "api error", "error:", "failed to", "unable to", 
                "i cannot", "i can't", "sorry, i", "i'm sorry"
            ];
            
            let s_lower = s.to_lowercase();
            if obvious_errors.iter().any(|pattern| s_lower.starts_with(pattern)) {
                return None;
            }
            
            // Skip exact model names when they appear alone (not within content)
            if s == "gemini-2.5-flash" || s == "claude-3-5-sonnet" || s == "gpt-4" {
                return None;
            }
            
            Some(s.clone())
        },
        serde_json::Value::Object(map) => {
            // Priority order for likely text fields
            let priority_keys = ["text", "content", "response", "message", "answer"];
            
            for key in &priority_keys {
                if let Some(val) = map.get(*key) {
                    if let Some(text) = find_text_in_json(val) {
                        return Some(text);
                    }
                }
            }
            
            // Search other fields, but skip common metadata fields
            let skip_keys = [
                "id", "response_id", "request_id", "session_id", "responseId", "model", "version",
                "timestamp", "token_count", "usage", "metadata", "headers", "status",
                "finish_reason", "logprobs", "safety_ratings", "citation_metadata"
            ];
            
            for (key, val) in map {
                if !skip_keys.contains(&key.as_str()) {
                    if let Some(text) = find_text_in_json(val) {
                        return Some(text);
                    }
                }
            }
            None
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(text) = find_text_in_json(item) {
                    return Some(text);
                }
            }
            None
        }
        _ => None,
    }
}

fn try_fallback_parsing(response_text: &str) -> Option<GeminiResponse> {
    // Strategy 1: Try to find text in various common patterns
    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(response_text) {
        // Pattern 1: Standard structure with candidates
        if let Some(candidates) = json_value.get("candidates").and_then(|c| c.as_array()) {
            if let Some(first_candidate) = candidates.first() {
                if let Some(content) = first_candidate.get("content") {
                    if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
                        if let Some(first_part) = parts.first() {
                            if let Some(text) = first_part.get("text").and_then(|t| t.as_str()) {
                                return Some(GeminiResponse {
                                    candidates: vec![Candidate {
                                        content: Some(Content {
                                            parts: vec![Part::Text {
                                                text: text.to_string(),
                                            }],
                                        }),
                                        finish_reason: None,
                                        safety_ratings: None,
                                        citation_metadata: None,
                                        token_count: None,
                                        grounding_attributions: None,
                                        logprobs_result: None,
                                    }],
                                    usage_metadata: None,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        // Pattern 2: Direct text field
        if let Some(text) = json_value.get("text").and_then(|t| t.as_str()) {
            return Some(GeminiResponse {
                candidates: vec![Candidate {
                    content: Some(Content {
                        parts: vec![Part::Text {
                            text: text.to_string(),
                        }],
                    }),
                    finish_reason: None,
                    safety_ratings: None,
                    citation_metadata: None,
                    token_count: None,
                    grounding_attributions: None,
                    logprobs_result: None,
                }],
                usage_metadata: None,
            });
        }
        
        // Pattern 3: Response field
        if let Some(response) = json_value.get("response").and_then(|r| r.as_str()) {
            return Some(GeminiResponse {
                candidates: vec![Candidate {
                    content: Some(Content {
                        parts: vec![Part::Text {
                            text: response.to_string(),
                        }],
                    }),
                    finish_reason: None,
                    safety_ratings: None,
                    citation_metadata: None,
                    token_count: None,
                    grounding_attributions: None,
                    logprobs_result: None,
                }],
                usage_metadata: None,
            });
        }
        
        // Pattern 4: Error responses with message
        if let Some(error) = json_value.get("error") {
            if let Some(message) = error.get("message").and_then(|m| m.as_str()) {
                // Return error message as response so it can be handled upstream
                return Some(GeminiResponse {
                    candidates: vec![Candidate {
                        content: Some(Content {
                            parts: vec![Part::Text {
                                text: format!("API Error: {}", message),
                            }],
                        }),
                        finish_reason: None,
                        safety_ratings: None,
                        citation_metadata: None,
                        token_count: None,
                        grounding_attributions: None,
                        logprobs_result: None,
                    }],
                    usage_metadata: None,
                });
            }
        }
    }
    
    // Strategy 2: Try to extract any meaningful text from the response
    // This handles cases where the response might be malformed JSON but contains usable text
    if response_text.trim().len() > 10 && !response_text.starts_with('{') {
        return Some(GeminiResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    parts: vec![Part::Text {
                        text: response_text.trim().to_string(),
                    }],
                }),
                finish_reason: None,
                safety_ratings: None,
                citation_metadata: None,
                token_count: None,
                grounding_attributions: None,
                logprobs_result: None,
            }],
            usage_metadata: None,
        });
    }
    
    None
}

fn create_edit_prompt(original_code: &str, query: &str, language: &str) -> String {
    // Analyze request complexity to optimize prompt
    let is_simple_change = query.len() < 100 && (
        query.to_lowercase().contains("rename") ||
        query.to_lowercase().contains("change") && query.to_lowercase().contains("title") ||
        query.to_lowercase().contains("fix") && query.split_whitespace().count() < 10
    );
    
    if is_simple_change {
        format!(
            r#"SIMPLE EDIT REQUEST - Make this specific change efficiently:

REQUEST: {query}
FILE TYPE: {language}

ORIGINAL CODE:
{original_code}

Return ONLY the complete modified file. Focus on the exact change requested - be precise and efficient."#,
            query = query,
            language = language,
            original_code = original_code
        )
    } else {
        format!(
            r#"You are an expert software engineer. Your task is to modify the provided {language} code according to the user's request.
The code you produce must be complete, functional, and high-quality.
Apply modern best practices and ensure the fix actually addresses the issue described.

CRITICAL: Return ONLY the complete, modified code file. Do not include any explanations, markdown formatting, or code block markers.
Make sure your changes actually solve the problem described in the request.

USER REQUEST: {query}

ORIGINAL CODE:
```{language}
{original_code}
```

Return the complete corrected file below (no markdown, no explanations):"#,
            language = language,
            original_code = original_code,
            query = query
        )
    }
}

fn extract_code_from_response(response: &GeminiResponse, original_code: &str) -> Result<String> {
let candidate = response.candidates.first()
.context("No candidates in Gemini response")?;

let part = candidate.content.as_ref().and_then(|c| c.parts.first())
    .context("No parts in Gemini response")?;

let text = match part {
    Part::Text { text } => text,
    Part::FunctionCall { .. } => {
        return Err(anyhow!("Expected text response but received function call"));
    }
};

// More robust code extraction from AI response
let extracted_code = extract_code_from_ai_response(text, original_code);

// Validate the extracted code is reasonable
if is_valid_code_response(&extracted_code, original_code) {
    Ok(extracted_code)
} else {
    // Return original code if extraction failed
    Ok(original_code.to_string())
}
}

/// Extract code from AI response, handling various formats
fn extract_code_from_ai_response(response_text: &str, original_code: &str) -> String {
    let text = response_text.trim();
    
    // Method 1: Look for code blocks with language specifiers
    if let Some(code) = extract_from_code_blocks(text) {
        if !code.trim().is_empty() && code.len() > 20 {
            return code;
        }
    }
    
    // Method 2: Look for complete HTML/JS/CSS documents (starts with common patterns)
    if text.starts_with("<!DOCTYPE") || text.starts_with("<html") || text.starts_with("<?xml") {
        return text.to_string();
    }
    
    // Method 3: If response starts with code-like content (no explanation)
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() > 0 && looks_like_code_start(lines[0]) {
        return text.to_string();
    }
    
    // Method 4: Look for code after "MODIFIED CODE:" or similar headers
    if let Some(code) = extract_code_after_headers(text) {
        if !code.trim().is_empty() && code.len() > 20 {
            return code;
        }
    }
    
    // Method 5: Find the largest code block in the response
    if let Some(code) = find_largest_code_section(text) {
        if !code.trim().is_empty() && code.len() > 20 {
            return code;
        }
    }
    
    // Method 6: Try to extract from explanatory text
    if let Some(code) = extract_from_explanatory_response(text, original_code) {
        return code;
    }
    
    // Method 7: Fall back to original code if nothing looks valid
    println!("Warning: Could not extract valid code from AI response, keeping original code");
    original_code.to_string()
}

/// Extract code from markdown code blocks
fn extract_from_code_blocks(text: &str) -> Option<String> {
    let mut in_code_block = false;
    let mut code_lines = Vec::new();
    
    for line in text.lines() {
        if line.starts_with("```") {
            if in_code_block {
                // End of code block
                break;
            } else {
                // Start of code block
                in_code_block = true;
                continue;
            }
        }
        
        if in_code_block {
            code_lines.push(line);
        }
    }
    
    if !code_lines.is_empty() {
        Some(code_lines.join("\n"))
    } else {
        None
    }
}

/// Check if a line looks like the start of code
fn looks_like_code_start(line: &str) -> bool {
    let code_patterns = [
        "<!DOCTYPE", "<html", "<head", "<body", "<?xml",
        "function ", "const ", "let ", "var ",
        "class ", "def ", "fn ", "public ", "private ",
        "import ", "from ", "#include", "package ",
        "{", "$(", "function(", "window.",
    ];
    
    let line_lower = line.to_lowercase();
    code_patterns.iter().any(|&pattern| line_lower.contains(pattern))
}

/// Find the largest section that looks like code
fn find_largest_code_section(text: &str) -> Option<String> {
    // This is a heuristic to find substantial code sections
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    
    let mut best_section = "";
    let mut best_score = 0;
    
    for paragraph in paragraphs {
        let score = calculate_code_likelihood_score(paragraph);
        if score > best_score && paragraph.len() > 50 {
            best_score = score;
            best_section = paragraph;
        }
    }
    
    if best_score > 3 {
        Some(best_section.to_string())
    } else {
        None
    }
}

/// Calculate how likely a text section is to be code
fn calculate_code_likelihood_score(text: &str) -> i32 {
    let mut score = 0;
    
    // Look for code indicators
    let code_indicators = [
        ("{", 1), ("}", 1), (";", 1), ("()", 1), ("[]", 1),
        ("function", 2), ("const", 2), ("let", 2), ("var", 2),
        ("class", 2), ("<!DOCTYPE", 3), ("<html", 3), ("<head", 3),
        ("import", 1), ("#include", 1), ("def ", 1), ("fn ", 1),
    ];
    
    for (indicator, points) in &code_indicators {
        score += text.matches(indicator).count() as i32 * points;
    }
    
    // Penalize if it looks like prose/explanation
    let prose_indicators = [
        "the code", "this will", "you can", "here's", "this is",
        "let me", "i'll", "we can", "should", "would"
    ];
    
    for indicator in &prose_indicators {
        score -= text.to_lowercase().matches(indicator).count() as i32;
    }
    
    score
}

/// Extract code after common headers like "MODIFIED CODE:", "Here's the fixed code:", etc.
fn extract_code_after_headers(text: &str) -> Option<String> {
    let headers = [
        "MODIFIED CODE:",
        "Here's the fixed code:",
        "Here's the corrected code:",
        "Updated code:",
        "Fixed code:",
        "The corrected code is:",
        "Here is the updated code:",
        "RESULT:",
        "OUTPUT:",
    ];
    
    for header in &headers {
        if let Some(start_pos) = text.find(header) {
            let after_header = &text[start_pos + header.len()..];
            let trimmed = after_header.trim();
            
            // Look for code block markers
            if let Some(code) = extract_from_code_blocks(trimmed) {
                return Some(code);
            }
            
            // If no code blocks, take the content after header until end or next explanation
            let lines: Vec<&str> = trimmed.lines().collect();
            let mut code_lines = Vec::new();
            let mut found_code = false;
            
            for line in lines {
                if line.trim().is_empty() && !found_code {
                    continue; // Skip empty lines before code
                }
                
                if looks_like_code_start(line) || found_code {
                    found_code = true;
                    // Stop if we hit explanatory text after finding code
                    if line.starts_with("This ") || line.starts_with("The ") || 
                       line.starts_with("I ") || line.starts_with("Note:") {
                        break;
                    }
                    code_lines.push(line);
                } else if found_code {
                    // Stop collecting if we've found code and hit non-code
                    break;
                }
            }
            
            if !code_lines.is_empty() {
                return Some(code_lines.join("\n"));
            }
        }
    }
    
    None
}

/// Extract code from explanatory responses that might contain inline code
fn extract_from_explanatory_response(text: &str, original_code: &str) -> Option<String> {
    // Look for patterns like "change X to Y" or "replace X with Y"
    let lines: Vec<&str> = text.lines().collect();
    
    // Check if this is a diff-like response
    if text.contains("```diff") || text.contains("--- ") || text.contains("+++ ") {
        return apply_diff_style_changes(text, original_code);
    }
    
    // Look for specific change instructions
    for line in &lines {
        let lower_line = line.to_lowercase();
        if (lower_line.contains("change") || lower_line.contains("replace") || 
            lower_line.contains("update")) && 
           (lower_line.contains("to:") || lower_line.contains("with:")) {
            
            // Try to extract the replacement from this line and following lines
            if let Some(replacement) = extract_replacement_from_instruction(line, &lines) {
                return Some(apply_replacement_to_original(original_code, replacement));
            }
        }
    }
    
    None
}

/// Apply diff-style changes to original code
fn apply_diff_style_changes(diff_text: &str, original_code: &str) -> Option<String> {
    let mut result = original_code.to_string();
    let lines: Vec<&str> = diff_text.lines().collect();
    
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("- ") || line.starts_with("-") {
            let old_text = line.strip_prefix("- ").unwrap_or(line.strip_prefix("-").unwrap_or(line));
            // Look for the corresponding + line
            if i + 1 < lines.len() && (lines[i + 1].starts_with("+ ") || lines[i + 1].starts_with("+")) {
                let new_text = lines[i + 1].strip_prefix("+ ").unwrap_or(lines[i + 1].strip_prefix("+").unwrap_or(lines[i + 1]));
                result = result.replace(old_text.trim(), new_text.trim());
                i += 2; // Skip both - and + lines
                continue;
            }
        }
        i += 1;
    }
    
    if result != original_code {
        Some(result)
    } else {
        None
    }
}

/// Extract replacement text from instruction line
fn extract_replacement_from_instruction(_instruction: &str, _lines: &[&str]) -> Option<String> {
    // This is a simplified implementation - could be expanded
    // to handle more complex instruction parsing
    None
}

/// Apply a simple replacement to original code
fn apply_replacement_to_original(original: &str, _replacement: String) -> String {
    // This is a simplified implementation
    original.to_string()
}

/// Validate that extracted code is reasonable
fn is_valid_code_response(extracted_code: &str, original_code: &str) -> bool {
    let extracted = extracted_code.trim();
    let original = original_code.trim();
    
    // Check for corruption patterns (same as in function_calling.rs)
    let corruption_patterns = [
        "api error", "failed to", "error:", "unable to", "i cannot",
        "i can't", "sorry, i", "i'm sorry"
    ];
    
    // Separate check for model names
    if extracted_code.trim() == "gemini-2.5-flash" || 
       extracted_code.trim().starts_with("gemini-") ||
       extracted_code.trim().starts_with("claude-") ||
       extracted_code.trim().starts_with("gpt-") {
        return false;
    }
    
    let extracted_lower = extracted.to_lowercase();
    for pattern in &corruption_patterns {
        if extracted_lower.contains(pattern) && extracted.len() < 100 {
            return false;
        }
    }
    
    // Must not be empty (unless original was also empty)
    if extracted.is_empty() && !original.is_empty() {
        return false;
    }
    
    // Must be reasonably sized compared to original
    if original.len() > 200 && extracted.len() < 20 {
        return false;
    }
    
    true
}

/// Count tokens for a request with function definitions
async fn count_tokens_for_request(prompt: &str, config: &Config) -> Result<TokenUsage> {
    count_tokens(prompt, config).await
}

/// Extract actual token usage from response metadata 
fn extract_actual_token_usage(response: &GeminiResponse, estimated: Option<TokenUsage>) -> Option<TokenUsage> {
    if let Some(usage) = &response.usage_metadata {
        return Some(TokenUsage {
            input_tokens: usage.prompt_token_count,
            output_tokens: Some(usage.candidates_token_count),
            total_tokens: usage.total_token_count,
        });
    }
    
    // Fallback to estimated if no usage metadata
    // Note: Gemini API may not return usageMetadata in all cases, estimated tokens work fine
    estimated
}

// Count tokens in a prompt before making the actual request
pub async fn count_tokens(prompt: &str, config: &Config) -> Result<TokenUsage> {
let client = Client::new();
let api_key = &config.gemini.api_key;
let model = &config.gemini.default_model;

let url = format!(
    "https://generativelanguage.googleapis.com/v1beta/models/{}:countTokens?key={}",
    model, api_key.as_ref().ok_or_else(|| anyhow::anyhow!("Gemini API key not configured"))?
);

let request_body = CountTokensRequest {
    contents: vec![Content {
        parts: vec![Part::Text {
            text: prompt.to_string(),
        }],
    }],
};

let response = client
    .post(&url)
    .json(&request_body)
    .timeout(Duration::from_secs(30))
    .send()
    .await
    .context("Failed to count tokens")?;

if !response.status().is_success() {
    let status = response.status();
    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
    return Err(anyhow::anyhow!("Token counting failed with status {}: {}", status, error_text));
}

let count_response: CountTokensResponse = response
    .json()
    .await
    .context("Failed to parse token count response")?;

Ok(TokenUsage {
    input_tokens: count_response.total_tokens,
    output_tokens: None,
    total_tokens: count_response.total_tokens,
})
}

/// BLAZING FAST Gemini query - always asks for concise responses
pub async fn query_gemini_fast(prompt: &str, config: &Config) -> Result<String> {
    let optimized_prompt = format!(
        "{}\n\nIMPORTANT: Be extremely concise. If editing code, return ONLY a diff patch. If answering questions, be brief.",
        prompt
    );
    query_gemini(&optimized_prompt, config).await
}

pub async fn query_gemini(prompt: &str, config: &Config) -> Result<String> {
    // Apply rate limiting to prevent API overload
    let wait_duration = if let Ok(mut limiter) = RATE_LIMITER.lock() {
        limiter.get_wait_duration()
    } else {
        None
    };
    
    if let Some(duration) = wait_duration {
        tokio::time::sleep(duration).await;
    }
    
    // Record the request after waiting
    if let Ok(mut limiter) = RATE_LIMITER.lock() {
        limiter.record_request();
    }
    
    let api_key = match config.gemini.api_key.as_ref() {
        Some(key) if !key.trim().is_empty() => key,
        _ => {
            return Err(anyhow::anyhow!(
                "Gemini API key not configured or empty. Please check your configuration."
            ));
        }
    };

// Use the shared HTTP client with per-request timeout
let client = &*HTTP_CLIENT;

let request = GeminiRequest {
    contents: vec![Content {
        parts: vec![Part::Text {
            text: prompt.to_string(),
        }],
    }],
    generation_config: GenerationConfig {
        temperature: 0.2, // Lower temperature for more consistent, focused code generation
        max_output_tokens: 8192, // INCREASED for diff patches
        response_logprobs: Some(true),
        logprobs: None,
    },
};

let url = format!(
    "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
    config.gemini.default_model, api_key
);

// Add retry logic with ESC interrupt checking
let mut last_error = None;

for attempt in 1..=config.gemini.max_retries {
    // Check for ESC key interrupt before each attempt
    if crate::input_handler::check_for_escape_key().await {
        return Err(anyhow::anyhow!("Operation interrupted by ESC key"));
    }
    
    match make_request_with_interrupt(&client, &url, &request).await {
        Ok(response) => {
            // Use robust parsing
            return extract_text_from_gemini_response(&response);
        }
        Err(e) => {
            if e.to_string().contains("interrupted by ESC") {
                return Err(e); // Don't retry interrupts
            }
            last_error = Some(e);
            if attempt < config.gemini.max_retries {
                tokio::time::sleep(Duration::from_millis(1000 * attempt as u64)).await;
            }
        }
    }
}

Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All retry attempts failed")))
}

fn extract_text_from_gemini_response(response: &GeminiResponse) -> Result<String> {
let candidate = response.candidates.first()
.context("No candidates in response")?;

if let Some(finish_reason) = &candidate.finish_reason {
    if finish_reason == "MAX_TOKENS" {
        return Err(anyhow::anyhow!("Response truncated due to MAX_TOKENS limit. Request smaller chunks or use diff patches."));
    }
    if finish_reason == "SAFETY" {
        return Err(anyhow::anyhow!("Response blocked due to safety concerns."));
    }
}

let part = candidate.content.as_ref().and_then(|c| c.parts.first())
    .context("No parts in Google response")?;

match part {
    Part::Text { text } => {
        if text.starts_with("API Error:") {
            Err(anyhow::anyhow!("{}", text))
        } else {
            // Extract thinking content and send to 💭 display
            let (main_response, _thinking) = extract_thinking_content(text);
            Ok(main_response)
        }
    },
    Part::FunctionCall { .. } => {
        Err(anyhow!("Expected text response but received function call"))
    }
}
}

/// Enhanced query with function calling support and thinking display
pub async fn query_gemini_with_function_calling(
prompt: &str,
config: &Config,
function_definitions: Option<&[crate::function_calling::FunctionDefinition]>,
) -> Result<(String, Option<crate::function_calling::FunctionCall>, Option<TokenUsage>)> {
    // Apply rate limiting to prevent API overload
    let wait_duration = if let Ok(mut limiter) = RATE_LIMITER.lock() {
        limiter.get_wait_duration()
    } else {
        None
    };
    
    if let Some(duration) = wait_duration {
        tokio::time::sleep(duration).await;
    }
    
    // Record the request after waiting
    if let Ok(mut limiter) = RATE_LIMITER.lock() {
        limiter.record_request();
    }
    
    let api_key = match config.gemini.api_key.as_ref() {
        Some(key) if !key.trim().is_empty() => key,
        _ => {
            return Err(anyhow::anyhow!(
                "Gemini API key not configured or empty. Please check your configuration."
            ));
        }
    };

    // Use shared HTTP client for better connection reuse
    let client = &*HTTP_CLIENT;

    let mut tools = None;
    if let Some(func_defs) = function_definitions {
        let function_declarations: Vec<FunctionDeclaration> = func_defs
            .iter()
            .map(|f| {
                // Create proper JSON Schema for Gemini API
                let params_schema = create_function_parameter_schema(&f.parameters);
                    
                FunctionDeclaration {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    parameters: params_schema,
                }
            })
            .collect();

        tools = Some(vec![Tool {
            function_declarations,
        }]);
    }

    // Status bar will show thinking, not processing messages
    
    // Show what the AI is thinking about
    let request_type = if prompt.contains("function_call") {
        "Deciding which function to call"
    } else if prompt.contains("edit") || prompt.contains("modify") {
        "Analyzing code for modifications"  
    } else if prompt.contains("create") || prompt.contains("write") {
        "Planning content creation"
    } else if prompt.contains("explain") || prompt.contains("what") {
        "Analyzing and explaining"
    } else {
        "Processing request"
    };
    
    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("💭 {}", request_type));
    
    // Store the internal prompt for debugging/reasoning display
    crate::thinking_display::PersistentStatusBar::set_ai_internal_prompt(&prompt);

    let request = AdvancedGeminiRequest {
        contents: vec![Content {
            parts: vec![Part::Text {
                text: prompt.to_string(),
            }],
        }],
        generation_config: GenerationConfig {
            temperature: 0.1, // Very low for focused, deterministic code generation
            max_output_tokens: 2048,
            response_logprobs: Some(true),
            logprobs: None,
        },
        system_instruction: Some(SystemInstruction {
            parts: vec![Part::Text {
                text: "You are a professional coding assistant with perfect memory and problem-solving instincts.

CRITICAL EMERGENCY PROTOCOLS:
- When users report 'there is no X now' or 'X is broken/missing', immediately PANIC and start fixing
- NEVER ask for clarification when users report obvious breakage - investigate and fix immediately  
- You have perfect memory of what files you just modified - use that context aggressively
- When something stops working after your changes, immediately read the affected files and diagnose

ANTI-DEFENSIVE BEHAVIOR:
- NEVER respond with \"could you please clarify\" or \"I don\'t have a record of\" when context is obvious
- When users mention files, folders, or code, immediately investigate rather than asking questions
- Assume intelligent intent - if user mentions \'cube\' after HTML/JS work, they mean the display element
- Take action first, ask questions only when genuinely ambiguous

CORE PRINCIPLES:
- Take action immediately - use functions to solve problems rather than explaining what you would do
- CAUSE & EFFECT REASONING: When users report issues, immediately connect them to your recent actions
- Analyze the PURPOSE of files you modified: HTML = structure, CSS = styling, JS = behavior/logic
- If user reports missing/broken elements, prioritize the file type most likely responsible:
  * Visual missing → check HTML structure first, then JS if dynamic
  * Styling issues → check CSS first
  * Functionality broken → check JS logic first
- Always read recent modifications to understand what changed and could cause the reported issue
- Be direct and focused in your responses - investigate immediately based on file purpose
- Use the RECENT FILES I MODIFIED section to understand causality

Use available functions proactively to complete tasks efficiently.".to_string(),
            }],
        }),
        tools,
    };

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        config.gemini.default_model, api_key
    );

    // Start thinking display 
    let thinking_display = crate::thinking_display::ThinkingDisplay::new();
    let thinking_handle = thinking_display.start_thinking("Processing request").await?;
    
    // Update thinking with more context
    // Skip redundant "Sending request" step
    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("💭 Sending {} chars to AI", prompt.len()));

    // Get estimated token count for fallback
    thinking_display.update_status("Counting tokens");
    let estimated_tokens = match count_tokens_for_request(prompt, config).await {
        Ok(usage) => Some(usage),
        Err(_e) => {
            // Create a rough estimate as fallback
            let word_count = prompt.split_whitespace().count() as u32;
            let estimated_input_tokens = (word_count * 4) / 3; // Rough tokens per word estimate
            Some(TokenUsage {
                input_tokens: estimated_input_tokens,
                output_tokens: None,
                total_tokens: estimated_input_tokens,
            })
        }
    };
    thinking_display.update_status("Analyzing request");

    let mut last_error = None;
    let start_time = Instant::now();

    for attempt in 1..=config.gemini.max_retries {
        // Update thinking for each attempt
        if attempt == 1 {
            // Skip "Waiting for AI response" - redundant
            // Show what we're actually asking the AI
            let thinking_preview = if prompt.len() > 100 {
                format!("💭 Analyzing: {}", &prompt[..100].replace('\n', " "))
            } else {
                format!("💭 Processing: {}", prompt.replace('\n', " "))
            };
            crate::thinking_display::PersistentStatusBar::set_ai_thinking(&thinking_preview);
        } else {
            crate::thinking_display::PersistentStatusBar::add_reasoning_step(&format!("→ Retry attempt {}", attempt));
            crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("💭 Retrying request (attempt {})", attempt));
        }
        
        match make_advanced_request(&client, &url, &request).await {
            Ok(response) => {
                let response_time_ms = start_time.elapsed().as_millis() as u32;
                
                // Calculate actual token usage from response
                let token_usage = extract_actual_token_usage(&response, estimated_tokens);
                
                // Update thinking display with final token info including timing
                if let Some(usage) = &token_usage {
                    thinking_display.update_status_with_tokens(
                        "Complete", 
                        usage.input_tokens, 
                        usage.output_tokens
                    );
                    // Update status bar with tokens and timing
                    crate::thinking_display::PersistentStatusBar::update_status_with_tokens_and_timing(
                        "Complete", 
                        usage.input_tokens, 
                        usage.output_tokens,
                        response_time_ms
                    );
                } else {
                    // Update with just timing if no token info
                    crate::thinking_display::PersistentStatusBar::update_response_time(response_time_ms);
                }
                
                // Capture AI response for real-time display
                let (text, function_call) = extract_function_call_response(&response)?;
                
                if let Some(ref func_call) = function_call {
                    // Let the function execution handle its own display
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Calling {}", func_call.name));
                } else {
                    // Just update thinking, don't add to reasoning steps
                    // Show more meaningful status about the completion
                    let response_preview = if text.len() > 50 { 
                        format!("{}...", &text[..50].replace('\n', " ")) 
                    } else { 
                        text.replace('\n', " ") 
                    };
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Response ready: {}", response_preview));
                }
                
                return Ok((text, function_call, token_usage));
            }
            Err(e) => {
                last_error = Some(e);
                if attempt < config.gemini.max_retries {
                    thinking_display.update_status(&format!("Retry {}/{}", attempt + 1, config.gemini.max_retries));
                    tokio::time::sleep(Duration::from_millis(1000 * attempt as u64)).await;
                }
            }
        }
    }

    thinking_handle.finish_with_error("Request failed");
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All retry attempts failed")))
}

pub async fn query_gemini_with_thinking(prompt: &str, config: &Config, thinking_budget: Option<u32>) -> Result<(String, Option<String>)> {
    let api_key = config.gemini.api_key.as_ref()
    .context("Gemini API key not configured")?;

    // Use shared HTTP client for better connection reuse
    let client = &*HTTP_CLIENT;

    // Create advanced request with thinking capabilities
    let system_instruction = Some(SystemInstruction {
        parts: vec![Part::Text {
            text: "You are an intelligent coding assistant. When you encounter complex tasks:
THINK through the problem step by step in <thinking> tags

Break complex tasks into manageable sub-tasks

Provide detailed reasoning for your decisions

If a task requires multiple steps, create a structured plan

Be thorough in your analysis before providing solutions".to_string(),
        }],
    });

    let enhanced_prompt = if thinking_budget.is_some() {
        format!(
            "{}\n\nIMPORTANT: This is a complex request. Please:\n\n1. First, analyze the task complexity in <thinking> tags\n\n2. If the task has multiple steps, break it down into a clear plan\n\n3. Identify any dependencies or prerequisites\n\n4. Consider potential issues or edge cases\n\n5. Then provide your response\n\nUse <thinking>...</thinking> tags for your internal reasoning.",
            prompt
        )
    } else {
        prompt.to_string()
    };

    let request = AdvancedGeminiRequest {
        contents: vec![Content {
            parts: vec![Part::Text {
                text: enhanced_prompt,
            }],
        }],
        generation_config: GenerationConfig {
            temperature: 0.3, // Lower temperature for more focused thinking
            max_output_tokens: thinking_budget.unwrap_or(2048),
            response_logprobs: Some(true),
            logprobs: Some(3), // Top 3 token probabilities
        },
        system_instruction,
        tools: None,
    };

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        config.gemini.default_model, api_key
    );

    // Use retry logic
    let mut last_error = None;

    for attempt in 1..=config.gemini.max_retries {
        match make_advanced_request(&client, &url, &request).await {
            Ok(response) => {
                return extract_thinking_response(&response);
            }
            Err(e) => {
                last_error = Some(e);
                if attempt < config.gemini.max_retries {
                    tokio::time::sleep(Duration::from_millis(1500 * attempt as u64)).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All retry attempts failed")))
}

async fn make_advanced_request(
    client: &Client,
    url: &str,
    request: &AdvancedGeminiRequest,
) -> Result<GeminiResponse> {
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .json(request)
        .send()
        .await
        .context("Failed to send advanced request to Gemini API")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("Gemini API error {}: {}", status, error_text));
    }

    // Get raw response text for better error handling
    let response_text = response
        .text()
        .await
        .context("Failed to read Gemini API response")?;

    // Raw response logging removed to prevent ratatui display corruption

    // Try to parse with fallback handling
    match serde_json::from_str::<GeminiResponse>(&response_text) {
        Ok(response) => Ok(response),
        Err(_parse_error) => {
            match try_fallback_parsing(&response_text) {
                Some(fallback_response) => Ok(fallback_response),
                None => {
                    let _preview = if response_text.len() > 200 {
                        format!("{}...", &response_text[..200])
                    } else {
                        response_text.clone()
                    };
                    // Final fallback: try to extract any useful text from the response
                    if let Some(extracted_text) = extract_text_from_response(&response_text) {
                        Ok(GeminiResponse {
                            candidates: vec![Candidate {
                                content: Some(Content {
                                    parts: vec![Part::Text {
                                        text: extracted_text,
                                    }],
                                }),
                                finish_reason: None,
                                safety_ratings: None,
                                citation_metadata: None,
                                token_count: None,
                                grounding_attributions: None,
                                logprobs_result: None,
                            }],
                            usage_metadata: None,
                        })
                    } else {
                        Err(anyhow::anyhow!(
                            "Unable to parse Gemini API response. Response preview: {}", 
                            if response_text.len() > 500 { format!("{}...", &response_text[..500]) } else { response_text.clone() }
                        ))
                    }
                }
            }
        }
    }
}

fn extract_streaming_text(chunk: &str) -> Option<String> {
    // For Gemini API streaming, chunks are JSON objects
    // Try to parse each chunk and extract text content
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(chunk) {
        if let Some(candidates) = json.get("candidates").and_then(|c| c.as_array()) {
            for candidate in candidates {
                if let Some(content) = candidate.get("content") {
                    if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                return Some(text.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn extract_thinking_response(response: &GeminiResponse) -> Result<(String, Option<String>)> {
    let candidate = response.candidates.first()
        .context("No candidates in Gemini response")?;

    let part = candidate.content.as_ref().and_then(|c| c.parts.first())
        .context("No parts in Gemini response")?;

    let text = match part {
        Part::Text { text } => text,
        Part::FunctionCall { .. } => {
            return Err(anyhow!("Expected text response but received function call"));
        }
    };

    // Check if this is an error response
    if text.starts_with("API Error:") {
        return Err(anyhow::anyhow!("{}", text));
    }

    // Extract thinking content if present
    let (main_response, thinking) = extract_thinking_content(text);

    // Get confidence metrics if available
    let confidence_info = if let Some(logprobs_result) = &candidate.logprobs_result {
        extract_confidence_metrics(logprobs_result)
    } else {
        None
    };

    let final_response = if let Some(confidence) = confidence_info {
        format!("{}\n\n_Confidence: {}_", main_response, confidence)
    } else {
        main_response
    };

    Ok((final_response, thinking))
}

fn extract_thinking_content(text: &str) -> (String, Option<String>) {
    // Look for <thinking>...</thinking> tags
    if let Some(thinking_start) = text.find("<thinking>") {
        if let Some(thinking_end) = text.find("</thinking>") {
            let thinking_content = text[thinking_start + 10..thinking_end].trim().to_string();
            let main_content = text[..thinking_start].trim().to_string() +
            text[thinking_end + 11..].trim();
            
            // Extract the most interesting reasoning from thinking content
            let best_reasoning = extract_best_reasoning(&thinking_content);
            if let Some(reasoning) = best_reasoning {
                // Send the best reasoning line to the 💭 display
                crate::thinking_display::PersistentStatusBar::set_ai_thinking(&reasoning);
            }
            
            return (main_content.trim().to_string(), Some(thinking_content));
        }
    }
    
    // Try to extract reasoning from other patterns if no <thinking> tags
    let inferred_reasoning = extract_reasoning_from_text(text);
    if let Some(reasoning) = inferred_reasoning {
        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&reasoning);
    }

    (text.to_string(), None)
}

fn extract_best_reasoning(thinking_content: &str) -> Option<String> {
    // Split into sentences and find the most informative ones
    let sentences: Vec<&str> = thinking_content
        .split(&['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.len() > 10)
        .collect();
    
    // Look for high-value reasoning patterns
    for sentence in &sentences {
        let lower = sentence.to_lowercase();
        
        // Strategy and planning sentences
        if lower.contains("i need to") || lower.contains("i should") || lower.contains("let me") {
            return Some(format!("Strategy: {}", sentence));
        }
        
        // Analysis sentences  
        if lower.contains("this suggests") || lower.contains("this indicates") || lower.contains("looking at") {
            return Some(format!("Analysis: {}", sentence));
        }
        
        // Problem solving
        if lower.contains("the issue") || lower.contains("the problem") || lower.contains("to fix") {
            return Some(format!("Problem solving: {}", sentence));
        }
        
        // Code understanding
        if lower.contains("the code") || lower.contains("function") || lower.contains("implementation") {
            return Some(format!("Code analysis: {}", sentence));
        }
    }
    
    // Fallback to the first substantial sentence
    sentences.first().map(|s| format!("Reasoning: {}", s))
}

fn extract_reasoning_from_text(text: &str) -> Option<String> {
    // Just extract the last sentence from the response
    let sentences: Vec<&str> = text
        .split(&['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.len() > 10)
        .collect();
    
    // Get the last meaningful sentence
    if let Some(last_sentence) = sentences.last() {
        Some(last_sentence.to_string())
    } else {
        // Fallback to first line if no sentences found
        text.lines()
            .find(|line| line.trim().len() > 10)
            .map(|line| line.trim().to_string())
    }
}

fn extract_confidence_metrics(logprobs_result: &LogprobsResult) -> Option<String> {
    if let Some(assessment) = calculate_confidence_assessment(logprobs_result) {
        let uncertainty_label = if assessment.uncertainty > 15.0 {
            " (High Uncertainty)"
        } else if assessment.uncertainty > 8.0 {
            " (Medium Uncertainty)" 
        } else {
            ""
        };
        
        Some(format!("{}% ({}{})", assessment.score as u32, assessment.level, uncertainty_label))
    } else {
        None
    }
}

fn calculate_confidence_assessment(logprobs_result: &LogprobsResult) -> Option<ConfidenceAssessment> {
    if let Some(chosen) = &logprobs_result.chosen_candidates {
        if !chosen.is_empty() {
            let total_tokens = chosen.len() as f64;
            let avg_logprob = chosen.iter()
                .map(|c| c.log_probability)
                .sum::<f64>() / total_tokens;
            
            let avg_confidence = avg_logprob.exp() * 100.0;
            
            // Calculate uncertainty (variance in confidence)
            let variance = chosen.iter()
                .map(|c| {
                    let prob = c.log_probability.exp();
                    let diff = prob - avg_logprob.exp();
                    diff * diff
                })
                .sum::<f64>() / total_tokens;
            
            let uncertainty = variance.sqrt() * 100.0;
            
            // Determine confidence level
            let level = match avg_confidence as u32 {
                90..=100 => ConfidenceLevel::VeryHigh,
                70..=89 => ConfidenceLevel::High, 
                50..=69 => ConfidenceLevel::Medium,
                30..=49 => ConfidenceLevel::Low,
                _ => ConfidenceLevel::VeryLow
            };
            
            return Some(ConfidenceAssessment {
                score: avg_confidence.round(),
                level,
                uncertainty: uncertainty.round(),
                token_count: chosen.len(),
            });
        }
    }
    None
}

/// Advanced function calling with thinking capabilities - the ultimate integration
pub async fn query_gemini_with_function_calling_and_thinking(
    prompt: &str, 
    config: &Config, 
    function_definitions: Option<&[crate::function_calling::FunctionDefinition]>,
    thinking_budget: Option<u32>
) -> Result<(String, Option<crate::function_calling::FunctionCall>, Option<TokenUsage>)> {
    let api_key = match config.gemini.api_key.as_ref() {
        Some(key) if !key.trim().is_empty() => key,
        _ => {
            return Err(anyhow::anyhow!(
                "Gemini API key not configured or empty. Please check your configuration."
            ));
        }
    };

    // Use shared HTTP client for better connection reuse and faster requests  
    let client = &*HTTP_CLIENT;

    let mut generation_config = GenerationConfig {
        temperature: 0.7, // Use default temperature for thinking
        max_output_tokens: 4096, // Default output tokens
        response_logprobs: Some(true),
        logprobs: Some(3),
    };

    // Adjust for thinking budget if provided
    if let Some(budget) = thinking_budget {
        generation_config.max_output_tokens = generation_config.max_output_tokens.max(budget);
    }

    // Enhanced system instruction with thinking guidelines
    let system_instruction = Some(SystemInstruction {
        parts: vec![Part::Text {
            text: "You are an advanced AI assistant with metacognitive capabilities.

THINKING PROCESS:
- Use <thinking> tags when you need to reason through complex problems
- Think through user intent, context, and optimal approaches
- Consider multiple possibilities before choosing actions
- Evaluate whether your planned actions will fully satisfy the user's request

FUNCTION CALLING:
- Call functions when you need to perform actions or get information
- Be strategic about function calls - make sure each one moves toward the goal
- Always consider the full context when choosing what action to take

METACOGNITION:
- Reflect on whether your response addresses the user's actual need
- Consider if additional steps might be required
- Think about the quality and completeness of your solution".to_string(),
        }]
    });

    // Convert function definitions to Gemini format
    let tools = if let Some(functions) = function_definitions {
        let function_declarations: Vec<FunctionDeclaration> = functions.iter().map(|func| {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();

            for param in &func.parameters {
                let param_type = match param.param_type.as_str() {
                    "string" => serde_json::Value::Object({
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".to_string(), serde_json::Value::String("string".to_string()));
                        obj.insert("description".to_string(), serde_json::Value::String(param.description.clone()));
                        obj
                    }),
                    _ => serde_json::Value::Object({
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".to_string(), serde_json::Value::String(param.param_type.clone()));
                        obj.insert("description".to_string(), serde_json::Value::String(param.description.clone()));
                        obj
                    }),
                };

                properties.insert(param.name.clone(), param_type);
                if param.required {
                    required.push(param.name.clone());
                }
            }

            let parameters = serde_json::Value::Object({
                let mut params = serde_json::Map::new();
                params.insert("type".to_string(), serde_json::Value::String("object".to_string()));
                params.insert("properties".to_string(), serde_json::Value::Object(properties));
                params.insert("required".to_string(), serde_json::Value::Array(
                    required.into_iter().map(serde_json::Value::String).collect()
                ));
                params
            });

            FunctionDeclaration {
                name: func.name.clone(),
                description: func.description.clone(),
                parameters,
            }
        }).collect();

        Some(vec![Tool { function_declarations }])
    } else {
        None
    };

    let request = AdvancedGeminiRequest {
        contents: vec![Content {
            parts: vec![Part::Text { text: prompt.to_string() }],
        }],
        generation_config,
        system_instruction,
        tools,
    };

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        config.gemini.default_model, api_key
    );

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            // Provide more specific error information for advanced requests
            if e.is_timeout() {
                anyhow::anyhow!("Gemini API request timed out")
            } else if e.is_connect() {
                anyhow::anyhow!("Failed to connect to Gemini API (network error): {}", e)
            } else if e.status() == Some(reqwest::StatusCode::TOO_MANY_REQUESTS) {
                anyhow::anyhow!("Gemini API rate limit exceeded - too many requests after edit operations")
            } else if e.status() == Some(reqwest::StatusCode::UNAUTHORIZED) {
                anyhow::anyhow!("Gemini API authentication failed - please check your API key")
            } else {
                anyhow::anyhow!("Failed to send request to Gemini API: {}", e)
            }
        })?;

    let response_text = response.text().await?;
    
    match serde_json::from_str::<GeminiResponse>(&response_text) {
        Ok(gemini_response) => {
            let token_usage = Some(calculate_token_usage(&gemini_response, prompt));
            
            // Check for malformed function call finish reason
            if let Some(candidate) = gemini_response.candidates.first() {
                if let Some(finish_reason) = &candidate.finish_reason {
                    if finish_reason == "MALFORMED_FUNCTION_CALL" {
                        // Fallback to simple text generation without function calling
                        return query_gemini_with_thinking(prompt, config, thinking_budget).await
                            .map(|(response, _thinking)| (response, None, token_usage));
                    }
                }
            }
            
            // Try to extract function call first
            match extract_function_call_response(&gemini_response) {
                Ok((text_response, function_call)) => {
                    // Extract thinking content from the text response
                    let (main_response, _thinking) = extract_thinking_content(&text_response);
                    Ok((main_response, function_call, token_usage))
                }
                Err(_) => {
                    // Fallback to thinking response extraction
                    let (text_response, _thinking) = extract_thinking_response(&gemini_response)?;
                    Ok((text_response, None, token_usage))
                }
            }
        }
        Err(parse_error) => {
            Err(anyhow::anyhow!(
                "Unable to parse Gemini API response: {}. Response: {}",
                parse_error,
                if response_text.len() > 500 { 
                    format!("{}...", &response_text[..500]) 
                } else { 
                    response_text 
                }
            ))
        }
    }
}

/// Calculate token usage from Gemini response
fn calculate_token_usage(response: &GeminiResponse, prompt: &str) -> TokenUsage {
    if let Some(usage) = &response.usage_metadata {
        return TokenUsage {
            input_tokens: usage.prompt_token_count,
            output_tokens: Some(usage.candidates_token_count),
            total_tokens: usage.total_token_count,
        };
    }
    
    // Fallback to rough estimate if no usage metadata
    let input_tokens = prompt.split_whitespace().count() as u32;
    let output_tokens = response.candidates.first()
        .and_then(|c| c.token_count);
    
    TokenUsage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens.unwrap_or(0),
    }
}

fn extract_function_call_response(response: &GeminiResponse) -> Result<(String, Option<crate::function_calling::FunctionCall>)> {
    let candidate = response.candidates.first()
        .context("No candidates in Gemini response")?;
    
    // Handle cases where content field is missing (e.g., MALFORMED_FUNCTION_CALL)
    if candidate.content.as_ref().map_or(true, |c| c.parts.is_empty()) {
        return Ok(("The AI encountered an issue generating a response. Let me try a different approach.".to_string(), None));
    }

    let part = candidate.content.as_ref().and_then(|c| c.parts.first())
        .context("No parts in Gemini response")?;

    match part {
        Part::FunctionCall { function_call } => {
            // Validate function call parameters before creating FunctionCall
            let func_call = match validate_function_call_parameters(&function_call.name, &function_call.args) {
                Ok(validated_params) => {
                    // Convert Value to HashMap<String, Value>
                    let args_map = if let serde_json::Value::Object(obj) = validated_params {
                        obj.into_iter().collect()
                    } else {
                        std::collections::HashMap::new()
                    };
                    
                    crate::function_calling::FunctionCall {
                        name: function_call.name.clone(),
                        arguments: args_map,
                    }
                },
                Err(e) => {
                    // Enhanced error logging with function call details
                    // Function validation failed - log to status bar instead of corrupting UI
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("🔧 Function validation failed: {}", function_call.name));
                    
                    // Attempt parameter correction for common issues
                    if let Ok(corrected_call) = attempt_parameter_correction(&function_call.name, &function_call.args) {
                        // Parameters auto-corrected - show in status bar
                        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("✅ Auto-corrected parameters for {}", function_call.name));
                        return Ok((String::new(), Some(corrected_call)));
                    }
                    
                    // If correction fails, return informative error
                    let error_text = format!(
                        "Function call validation failed for '{}'. Error: {}\n\nThis is a system issue that has been logged. Please try rephrasing your request.",
                        function_call.name, e
                    );
                    return Ok((error_text, None));
                }
            };
            Ok((String::new(), Some(func_call)))
        },
        Part::Text { text } => {
            if text.starts_with("API Error:") {
                Err(anyhow::anyhow!("{}", text))
            } else {
                // Also try to parse function calls from text for backward compatibility
                if let Some(func_call) = parse_function_call_from_text_safely(text) {
                    Ok((String::new(), Some(func_call)))
                } else {
                    Ok((text.clone(), None))
                }
            }
        }
    }
}

/// Attempt to auto-correct common parameter issues
fn attempt_parameter_correction(function_name: &str, params: &serde_json::Value) -> Result<crate::function_calling::FunctionCall> {
    let mut corrected_params = params.clone();
    
    // Common corrections based on function type
    match function_name {
        "read_file" | "write_file" | "edit_code" => {
            // Ensure file_path exists and is a string
            if let serde_json::Value::Object(ref mut obj) = corrected_params {
                // Fix common path format issues
                if let Some(path_val) = obj.get_mut("file_path") {
                    if let Some(path_str) = path_val.as_str() {
                        // Normalize path separators for Windows
                        let normalized_path = path_str.replace("/", "\\");
                        *path_val = serde_json::Value::String(normalized_path);
                    }
                }
                
                // Add missing required parameters with defaults
                if !obj.contains_key("file_path") && obj.contains_key("path") {
                    if let Some(path) = obj.remove("path") {
                        obj.insert("file_path".to_string(), path);
                    }
                }
                
                // Fix parameter name mapping for edit_code function
                if function_name == "edit_code" {
                    // Map "instructions" or "content" to "query"
                    if !obj.contains_key("query") {
                        if let Some(instructions) = obj.remove("instructions") {
                            obj.insert("query".to_string(), instructions);
                        } else if let Some(content) = obj.remove("content") {
                            obj.insert("query".to_string(), content);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    
    // Try validation again with corrected parameters
    let validated = validate_function_call_parameters(function_name, &corrected_params)?;
    
    let args_map = if let serde_json::Value::Object(obj) = validated {
        obj.into_iter().collect()
    } else {
        std::collections::HashMap::new()
    };
    
    Ok(crate::function_calling::FunctionCall {
        name: function_name.to_string(),
        arguments: args_map,
    })
}

/// Create proper JSON Schema for function parameters (Gemini API compatible)
fn create_function_parameter_schema(parameters: &[crate::function_calling::Parameter]) -> serde_json::Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    
    for param in parameters {
        // Map our parameter types to JSON Schema types
        let json_type = match param.param_type.as_str() {
            "string" => "string",
            "number" | "integer" => "number",
            "boolean" => "boolean",
            "array" => "array",
            "object" => "object",
            _ => "string", // Default fallback
        };
        
        let mut property_schema = serde_json::json!({
            "type": json_type,
            "description": param.description
        });
        
        // Add rich validation patterns based on parameter name and type
        if json_type == "string" {
            match param.name.as_str() {
                name if name.contains("url") || name.contains("link") => {
                    property_schema["format"] = serde_json::json!("uri");
                },
                name if name.contains("email") => {
                    property_schema["format"] = serde_json::json!("email");
                },
                name if name.contains("date") => {
                    property_schema["format"] = serde_json::json!("date");
                    property_schema["pattern"] = serde_json::json!("^\\d{4}-\\d{2}-\\d{2}$");
                },
                name if name.contains("time") => {
                    property_schema["format"] = serde_json::json!("time");
                },
                name if name.contains("path") || name.contains("file") => {
                    property_schema["pattern"] = serde_json::json!("^[^\\x00-\\x1f\\x7f\"*<>?|]+$");
                },
                name if name.contains("color") => {
                    property_schema["pattern"] = serde_json::json!("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$");
                },
                name if name.contains("version") => {
                    property_schema["pattern"] = serde_json::json!("^\\d+\\.\\d+(\\.\\d+)?(-[a-zA-Z0-9]+)?$");
                },
                _ => {}
            }
            
            // Add common string constraints
            if param.description.contains("required") || param.required {
                property_schema["minLength"] = serde_json::json!(1);
            }
        }
        
        // Add numeric constraints
        if json_type == "number" {
            match param.name.as_str() {
                name if name.contains("port") => {
                    property_schema["minimum"] = serde_json::json!(1);
                    property_schema["maximum"] = serde_json::json!(65535);
                },
                name if name.contains("percentage") || name.contains("percent") => {
                    property_schema["minimum"] = serde_json::json!(0);
                    property_schema["maximum"] = serde_json::json!(100);
                },
                name if name.contains("timeout") || name.contains("delay") => {
                    property_schema["minimum"] = serde_json::json!(0);
                },
                name if name.contains("limit") || name.contains("max") => {
                    property_schema["minimum"] = serde_json::json!(1);
                },
                _ => {
                    // General numeric constraints
                    if param.description.contains("positive") {
                        property_schema["minimum"] = serde_json::json!(0);
                    }
                }
            }
        }
        
        // Add array constraints
        if json_type == "array" {
            property_schema["minItems"] = serde_json::json!(0);
            if param.required {
                property_schema["minItems"] = serde_json::json!(1);
            }
        }
        
        properties.insert(param.name.clone(), property_schema);
        
        if param.required {
            required.push(param.name.clone());
        }
    }
    
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required
    })
}

/// Validate function call parameters to prevent malformed errors
fn validate_function_call_parameters(function_name: &str, params: &serde_json::Value) -> Result<serde_json::Value> {
    // Basic validation - ensure parameters is an object
    if !params.is_object() {
        return Err(anyhow!("Parameters must be a JSON object, got: {}", params));
    }
    
    // Function-specific validation
    match function_name {
        "create_file" | "edit_code" => {
            let obj = params.as_object().unwrap();
            if !obj.contains_key("file_path") {
                return Err(anyhow!("Missing required file_path parameter"));
            }
            if !obj.contains_key("query") {
                return Err(anyhow!("Missing required query parameter"));
            }
        },
        "search_music" => {
            let obj = params.as_object().unwrap();
            if !obj.contains_key("query") {
                return Err(anyhow!("Missing required query parameter"));
            }
        },
        _ => {
            // For unknown functions, just ensure it's a valid object
            if params.as_object().is_none() {
                return Err(anyhow!("Invalid parameter object"));
            }
        }
    }
    
    Ok(params.clone())
}

/// Safely parse function calls from text with better error handling
fn parse_function_call_from_text_safely(text: &str) -> Option<crate::function_calling::FunctionCall> {
    // Look for JSON function call in the response with improved parsing
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            let json_str = &text[start..=end];

            // Try to parse as function call with error recovery
            match serde_json::from_str::<serde_json::Value>(json_str) {
                Ok(parsed) => {
                    if let Some(function_call_obj) = parsed.get("function_call") {
                        if let (Some(name), Some(params)) = (
                            function_call_obj.get("name").and_then(|n| n.as_str()),
                            function_call_obj.get("arguments")
                        ) {
                            // Validate before returning
                            match validate_function_call_parameters(name, params) {
                                Ok(validated_params) => {
                                    // Convert Value to HashMap<String, Value>
                                    let args_map = if let serde_json::Value::Object(obj) = validated_params {
                                        obj.into_iter().collect()
                                    } else {
                                        std::collections::HashMap::new()
                                    };
                                    
                                    return Some(crate::function_calling::FunctionCall {
                                        name: name.to_string(),
                                        arguments: args_map,
                                    });
                                },
                                Err(_) => {
                                    // Invalid function parameters - show in status bar
                                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("⚠️ Invalid function call parameters: {}", name));
                                    return None;
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    // JSON parse failed - show in status bar instead of corrupting display
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("⚠️ Failed to parse JSON from Gemini response: {}", e));
                }
            }
        }
    }

    None
}

// Removed old parse_function_call_from_text - replaced with safer version above

pub async fn analyze_task_complexity(input: &str, context: &str, config: &Config) -> Result<TaskComplexity> {
    let analysis_prompt = format!(
        r#"Analyze this user request for task complexity:
Request: "{}"
Context: {}
Determine:
1. Is this a simple task (can be done in one step)?
2. Is this a complex task requiring multiple steps?
3. What are the main steps needed?
4. Are there any dependencies or prerequisites?
Respond with JSON in this exact format:
{{
"complexity": "simple" | "moderate" | "complex",
"reasoning": "explanation of complexity",
"steps": ["step1", "step2", ...],
"requires_planning": true | false,
"estimated_time": "short" | "medium" | "long"
}}"#,
        input,
        context
    );

    let response = query_gemini(&analysis_prompt, config).await?;

    // Try to parse JSON response
    match parse_task_complexity_response(&response) {
        Ok(complexity) => Ok(complexity),
        Err(_) => {
            // Fallback to simple analysis
            Ok(TaskComplexity {
                complexity: determine_fallback_complexity(input),
                reasoning: "Automated analysis based on keyword detection".to_string(),
                steps: vec![input.to_string()],
                requires_planning: false,
                estimated_time: "medium".to_string(),
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskComplexity {
    pub complexity: String,
    pub reasoning: String,
    pub steps: Vec<String>,
    pub requires_planning: bool,
    pub estimated_time: String,
}

fn parse_task_complexity_response(response: &str) -> Result<TaskComplexity> {
    // Try to extract JSON from response
    let json_str = if response.contains('{') {
        let start = response.find('{').unwrap();
        let end = response.rfind('}').unwrap_or(response.len() - 1);
        &response[start..=end]
    } else {
        response
    };

    let parsed: serde_json::Value = serde_json::from_str(json_str)?;

    Ok(TaskComplexity {
        complexity: parsed.get("complexity")
            .and_then(|v| v.as_str())
            .unwrap_or("moderate")
            .to_string(),
        reasoning: parsed.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("Analysis not available")
            .to_string(),
        steps: parsed.get("steps")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect())
            .unwrap_or_else(|| vec!["Complete the task".to_string()]),
        requires_planning: parsed.get("requires_planning")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        estimated_time: parsed.get("estimated_time")
            .and_then(|v| v.as_str())
            .unwrap_or("medium")
            .to_string(),
    })
}

fn determine_fallback_complexity(input: &str) -> String {
    let input_lower = input.to_lowercase();
    let complex_indicators = [
        "implement", "create from scratch", "build entire", "refactor", "optimize",
        "multiple files", "several", "many", "all", "every", "complete system",
        "architecture", "design", "full", "entire", "comprehensive", "advanced"
    ];

    let simple_indicators = [
        "fix this", "change", "update", "edit", "modify", "small", "quick",
        "simple", "just", "only", "single"
    ];

    let complex_count = complex_indicators.iter()
        .filter(|&indicator| input_lower.contains(indicator))
        .count();
    
    let simple_count = simple_indicators.iter()
        .filter(|&indicator| input_lower.contains(indicator))
        .count();

    if complex_count > simple_count && complex_count >= 2 {
        "complex".to_string()
    } else if complex_count > 0 || input.len() > 100 {
        "moderate".to_string() 
    } else {
        "simple".to_string()
    }
}