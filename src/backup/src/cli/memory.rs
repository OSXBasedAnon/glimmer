use anyhow::{anyhow, Context as AnyhowContext, Result};
use sled::{Db, Tree};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use std::fs;
use dirs;
use std::sync::{Arc, Mutex, atomic::{AtomicU64, Ordering}};
use crate::config::Config;
use crate::gemini;

// --- Data Structures for Memory ---

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System, // For summaries or system-level context
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "User"),
            MessageRole::Assistant => write!(f, "Assistant"),
            MessageRole::System => write!(f, "System"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub importance: MessageImportance,
    #[serde(default)]
    pub metadata: MessageMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MessageImportance {
    Critical,    // Problem statements, error messages, code diffs
    Important,   // User preferences, successful solutions, file references
    #[default]
    Contextual,  // Follow-up questions, confirmations
    Casual,      // General chat
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessageMetadata {
    pub files_mentioned: Vec<String>,
    pub error_type: Option<String>,
    pub task_type: Option<String>,
    pub success_indicator: bool,
    pub code_changes: Vec<String>,
}

// --- Memory Engine ---

#[derive(Clone)]
pub struct MemoryEngine {
    db: Db,
    working_memory_tree: Tree,      // 7 most recent focused messages
    background_storage_tree: Tree,   // All historical messages with metadata
    changes_tree: Tree,             // Track code changes and outcomes  
    index_tree: Tree,              // Search index for semantic retrieval
    metadata_tree: Tree,
    config: Config,
    last_background_move: Arc<Mutex<DateTime<Utc>>>,
    message_count: Arc<AtomicU64>,
}

// Modern context window management (separate from database storage)
const WORKING_CONTEXT_FOCUSED: usize = 7;      // Core focused messages (problem statements, recent exchanges)
const WORKING_CONTEXT_TRUNCATED: usize = 25;   // Smart truncated messages (summaries, key decisions)
const COMPACTION_THRESHOLD: u64 = 150;         // Start considering compaction after 150 messages
const FORCE_COMPACTION_LIMIT: u64 = 300;       // Force compaction - conversation is getting unwieldy  
const DATABASE_RETENTION_DAYS: i64 = 30;       // Keep searchable history for 30 days
const SEMANTIC_SEARCH_LIMIT: usize = 5;        // Additional relevant messages from search (NOT in context window)

impl MemoryEngine {
    pub fn new(config: &Config) -> Result<Self> {
        let db_dir = dirs::data_dir()
            .ok_or_else(|| anyhow!("Could not get data directory"))?
            .join("glimmer");

        // Ensure the directory exists
        fs::create_dir_all(&db_dir)?;

        let db_path = db_dir.join("conversation.db");

        let db = sled::open(&db_path)
            .with_context(|| format!("Failed to open sled database at {:?}", db_path))?;
        let working_memory_tree = db.open_tree("working_memory")?;
        let background_storage_tree = db.open_tree("background_storage")?;
        let changes_tree = db.open_tree("changes")?;
        let index_tree = db.open_tree("index")?;
        let metadata_tree = db.open_tree("metadata")?;
        
        // Load or initialize metadata
        let (last_background_move, message_count) = Self::load_metadata(&metadata_tree)?;
        
        Ok(Self {
            db,
            working_memory_tree,
            background_storage_tree,
            changes_tree,
            index_tree,
            metadata_tree,
            config: config.clone(),
            last_background_move: Arc::new(Mutex::new(last_background_move)),
            message_count: Arc::new(AtomicU64::new(message_count)),
        })
    }

    pub async fn add_message(&self, role: MessageRole, content: &str) -> Result<()> {
        let message = ChatMessage {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            importance: MessageImportance::default(),
            metadata: MessageMetadata::default(),
        };
        let key = message.timestamp.to_rfc3339();
        let value = serde_json::to_vec(&message)?;
        self.working_memory_tree.insert(key.as_bytes(), value)?;
        
        // Increment message count and save metadata
        self.message_count.fetch_add(1, Ordering::Relaxed);
        self.save_metadata()?;
        
        // PERFORMANCE FIX: Don't block user input with compaction
        // Check if we need compaction but run it in background
        if self.should_compact().await? {
            // Clone necessary data for background compaction
            let db_clone = self.db.clone();
            let working_tree_clone = self.working_memory_tree.clone();
            let background_tree_clone = self.background_storage_tree.clone();
            let config_clone = self.config.clone();
            let metadata_tree_clone = self.metadata_tree.clone();
            let last_move_clone = self.last_background_move.clone();
            let count_clone = self.message_count.clone();
            
            // Run compaction in background - don't block user
            tokio::spawn(async move {
                if let Err(e) = Self::compact_memory_background(
                    db_clone, working_tree_clone, background_tree_clone, 
                    config_clone, metadata_tree_clone, last_move_clone, count_clone
                ).await {
                    // Background compaction failed - log to status bar instead
                    crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("Background compaction failed: {}", e));
                }
            });
        }
        
        Ok(())
    }

    pub async fn get_recent_messages(&self, limit: usize) -> Result<Vec<ChatMessage>> {
        let mut messages = Vec::new();
        // Iterate backwards
        for item in self.working_memory_tree.iter().rev() {
            if messages.len() >= limit {
                break;
            }
            let (_, value) = item?;
            let message: ChatMessage = serde_json::from_slice(&value)?;
            messages.push(message);
        }
        messages.reverse(); // Put them back in chronological order
        Ok(messages)
    }

    pub async fn clear_conversation(&self) -> Result<()> {
        self.working_memory_tree.clear()?;
        self.background_storage_tree.clear()?;
        
        // Reset metadata to prevent compaction loops
        self.message_count.store(0, Ordering::Relaxed);
        {
            let mut last_move = self.last_background_move.lock().unwrap();
            *last_move = Utc::now();
        }
        self.save_metadata()?;
        
        self.db.flush_async().await?;
        Ok(())
    }

    /// Modern context retrieval - focused recent + smart truncated (does NOT flood context window)
    pub async fn get_context(&self, _recent_limit: usize, _max_summary_tokens: usize) -> Result<String> {
        let mut context = String::new();
        
        // Get all recent messages for analysis
        let all_recent = self.get_recent_messages(50).await?; // Analyze more, send less
        
        if all_recent.len() <= WORKING_CONTEXT_FOCUSED {
            // Small conversation - send all
            for msg in &all_recent {
                context.push_str(&format!("{}: {}\n", msg.role, self.truncate_message_content(&msg.content)));
            }
        } else {
            // Large conversation - use focused + truncated approach
            let focused_count = WORKING_CONTEXT_FOCUSED;
            let truncated_count = WORKING_CONTEXT_TRUNCATED - focused_count;
            
            // Take most recent messages as focused (full content)
            let focused_messages = &all_recent[all_recent.len().saturating_sub(focused_count)..];
            for msg in focused_messages {
                context.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
            
            // Add separator
            if truncated_count > 0 {
                context.push_str("\n--- Earlier Context (Summarized) ---\n");
                
                // Take earlier messages and truncate them
                let truncated_start = all_recent.len().saturating_sub(WORKING_CONTEXT_TRUNCATED);
                let truncated_end = all_recent.len().saturating_sub(focused_count);
                let truncated_messages = &all_recent[truncated_start..truncated_end];
                
                for msg in truncated_messages {
                    let truncated_content = self.truncate_message_content(&msg.content);
                    context.push_str(&format!("{}: {}\n", msg.role, truncated_content));
                }
            }
        }

        Ok(context)
    }
    
    /// Intelligently truncate message content (send diffs for code, summaries for long messages)
    fn truncate_message_content(&self, content: &str) -> String {
        // If it's a code file or very long content, create intelligent summary
        if content.len() > 1000 || self.looks_like_code_file(content) {
            if content.contains("```") || content.contains("diff") {
                // It's code - try to extract key changes
                self.extract_code_changes(content)
            } else {
                // Long text - summarize key points
                format!("{}...\n[Message truncated - {} chars total]", 
                    &content[..content.len().min(200)], content.len())
            }
        } else {
            content.to_string()
        }
    }
    
    fn looks_like_code_file(&self, content: &str) -> bool {
        content.lines().count() > 20 && (
            content.contains("function ") || 
            content.contains("def ") ||
            content.contains("class ") ||
            content.contains("impl ") ||
            content.contains("struct ") ||
            content.contains("#include") ||
            content.contains("import ")
        )
    }
    
    fn extract_code_changes(&self, content: &str) -> String {
        if content.contains("diff") || content.contains("@@") {
            // Already a diff - keep it
            content.to_string()
        } else if content.contains("```") {
            // Extract just the key functions/changes, not the whole file
            let lines: Vec<&str> = content.lines().collect();
            let mut result = String::new();
            let mut in_code_block = false;
            let mut code_lines = 0;
            
            for line in lines {
                if line.contains("```") {
                    in_code_block = !in_code_block;
                    result.push_str(line);
                    result.push('\n');
                } else if in_code_block {
                    code_lines += 1;
                    if code_lines <= 15 {  // Only show first 15 lines of each code block
                        result.push_str(line);
                        result.push('\n');
                    } else if code_lines == 16 {
                        result.push_str("... [code truncated] ...\n");
                    }
                } else {
                    result.push_str(line);
                    result.push('\n');
                }
            }
            
            if result.len() < content.len() {
                result.push_str(&format!("\n[Full content: {} chars, showing key excerpts]", content.len()));
            }
            
            result
        } else {
            // Large non-code content
            format!("{}...\n[Content truncated - {} total chars]", 
                &content[..content.len().min(300)], content.len())
        }
    }

    /// Gets a summary of the conversation up to a certain point.
    /// If a summary doesn't exist, it creates one using the LLM.
    async fn get_or_create_summary(&self, messages: &[ChatMessage], max_tokens: usize) -> Result<String> {
        if messages.is_empty() {
            return Ok("".to_string());
        }

        // The key for the summary will be the timestamp of the last message being summarized.
        let last_message_timestamp = messages.last().unwrap().timestamp.to_rfc3339();

        if let Some(summary_bytes) = self.background_storage_tree.get(last_message_timestamp.as_bytes())? {
            return Ok(String::from_utf8(summary_bytes.to_vec())?);
        }

        // No summary found, so we generate one.
        let conversation_text: String = messages.iter()
            .map(|msg| format!("{}: {}", msg.role.as_str(), msg.content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Please summarize the following conversation concisely. Focus on key facts, user intentions, and important outcomes. The summary will be used as context for a continuing conversation. Do not exceed {} tokens.\n\nCONVERSATION:\n---\n{}\n---\n\nSUMMARY:",
            max_tokens,
            conversation_text
        );

        let summary = gemini::query_gemini(&prompt, &self.config).await?;

        // Store the new summary
        self.background_storage_tree.insert(last_message_timestamp.as_bytes(), summary.as_bytes())?;
        self.db.flush_async().await?;

        Ok(summary)
    }
    
    // --- Memory Management Methods ---
    
    fn load_metadata(metadata_tree: &Tree) -> Result<(DateTime<Utc>, u64)> {
        let last_background_move = if let Some(data) = metadata_tree.get("last_background_move")? {
            let timestamp_str = String::from_utf8(data.to_vec())?;
            DateTime::parse_from_rfc3339(&timestamp_str)?.with_timezone(&Utc)
        } else {
            Utc::now() // First time setup
        };
        
        let message_count = if let Some(data) = metadata_tree.get("message_count")? {
            let count_str = String::from_utf8(data.to_vec())?;
            count_str.parse::<u64>().unwrap_or(0)
        } else {
            0
        };
        
        Ok((last_background_move, message_count))
    }
    
    fn save_metadata(&self) -> Result<()> {
        let last_move = self.last_background_move.lock().unwrap();
        self.metadata_tree.insert("last_background_move", last_move.to_rfc3339().as_bytes())?;
        self.metadata_tree.insert("message_count", self.message_count.load(Ordering::Relaxed).to_string().as_bytes())?;
        Ok(())
    }
    
    async fn should_compact(&self) -> Result<bool> {
        let message_count = self.message_count.load(Ordering::Relaxed);
        // Force compaction if conversation has become unwieldy
        if message_count >= FORCE_COMPACTION_LIMIT {
            return Ok(true);
        }
        
        // Consider compaction if we've hit the soft threshold and it's been a while
        if message_count >= COMPACTION_THRESHOLD {
            // Only compact if it's been at least 2 hours since last compaction
            // This prevents frequent compaction during active sessions
            let last_move = self.last_background_move.lock().unwrap();
            let time_since_compaction = Utc::now() - *last_move;
            if time_since_compaction > ChronoDuration::hours(2) {
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    async fn compact_memory(&self) -> Result<()> {
        
        let message_count = self.message_count.load(Ordering::Relaxed);
        let reason = if message_count >= FORCE_COMPACTION_LIMIT {
            "conversation has grown very long"
        } else {
            "optimizing for better performance"
        };
        
        // Send compacting message to status bar instead of direct println to avoid UI scrambling
        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!(
            "🗜️ Compacting conversation memory ({} messages → {} focused + {} truncated + summary): {}",
            message_count, WORKING_CONTEXT_FOCUSED, 
            WORKING_CONTEXT_TRUNCATED - WORKING_CONTEXT_FOCUSED, reason
        ));
        
        // Get all messages
        let all_messages: Vec<ChatMessage> = self.working_memory_tree.iter()
            .filter_map(Result::ok)
            .filter_map(|(_, v)| serde_json::from_slice(&v).ok())
            .collect();
        
        if all_messages.len() <= WORKING_CONTEXT_TRUNCATED {
            crate::thinking_display::PersistentStatusBar::set_ai_thinking("Memory already optimal, skipping compaction");
            return Ok(());
        }
        
        // Sort by timestamp
        let mut sorted_messages = all_messages;
        sorted_messages.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        // Smart compaction: preserve recent context for ongoing tasks
        let split_point = sorted_messages.len().saturating_sub(WORKING_CONTEXT_TRUNCATED);
        let (messages_to_summarize, recent_messages) = sorted_messages.split_at(split_point.max(1));
        
        if !messages_to_summarize.is_empty() {
            // Create comprehensive summary
            let summary = self.create_comprehensive_summary(messages_to_summarize).await?;
            
            // Clear old conversation data
            self.working_memory_tree.clear()?;
            
            // Store summary as a system message
            let summary_message = ChatMessage {
                role: MessageRole::System,
                content: summary,
                timestamp: if let Some(last_old_msg) = messages_to_summarize.last() {
                    last_old_msg.timestamp + chrono::Duration::seconds(1)
                } else {
                    Utc::now()
                },
                importance: MessageImportance::Important,
                metadata: MessageMetadata::default(),
            };
            
            let key = summary_message.timestamp.to_rfc3339();
            let value = serde_json::to_vec(&summary_message)?;
            self.working_memory_tree.insert(key.as_bytes(), value)?;
            
            // Re-add recent messages
            for message in recent_messages {
                let key = message.timestamp.to_rfc3339();
                let value = serde_json::to_vec(message)?;
                self.working_memory_tree.insert(key.as_bytes(), value)?;
            }
        }
        
        // Update metadata
        {
            let mut last_move = self.last_background_move.lock().unwrap();
            *last_move = Utc::now();
        }
        self.message_count.store((recent_messages.len() + 1) as u64, Ordering::Relaxed); // +1 for summary
        self.save_metadata()?;
        
        // Flush database
        self.db.flush_async().await?;
        
        crate::thinking_display::PersistentStatusBar::set_ai_thinking("🗜️ Memory compacted successfully - ready for continued productivity");
        
        Ok(())
    }
    
    async fn create_comprehensive_summary(&self, messages: &[ChatMessage]) -> Result<String> {
        if messages.is_empty() {
            return Ok("".to_string());
        }
        
        let conversation_text: String = messages.iter()
            .map(|msg| format!("[{}] {}: {}", 
                msg.timestamp.format("%H:%M"), 
                msg.role.as_str(), 
                msg.content))
            .collect::<Vec<_>>()
            .join("\n");
        
        let prompt = format!(
            "Create a comprehensive but concise summary of this conversation history. Focus on:\n\
            1. Key topics discussed\n\
            2. Important decisions made\n\
            3. Files or code worked on\n\
            4. User preferences learned\n\
            5. Context that would be valuable for continuing the conversation\n\
            \n\
            Keep the summary under 2000 words but capture all essential information.\n\
            \n\
            CONVERSATION HISTORY:\n\
            ---\n\
            {}\n\
            ---\n\
            \n\
            COMPREHENSIVE SUMMARY:",
            conversation_text
        );
        
        let summary = gemini::query_gemini(&prompt, &self.config).await?;
        Ok(format!("**Conversation Summary** ({})\n\n{}", 
            Utc::now().format("%Y-%m-%d %H:%M UTC"), 
            summary))
    }
    
    // Force compaction (for manual triggering)
    pub async fn force_compact(&self) -> Result<()> {
        self.compact_memory().await
    }
    
    /// Search historical context (NOT included in working context window)
    /// For queries like "what did we change about this project last week"
    pub async fn search_historical_context(&self, query: &str, days_back: i64) -> Result<Vec<ChatMessage>> {
        let cutoff_date = Utc::now() - ChronoDuration::days(days_back);
        let query_lower = query.to_lowercase();
        
        let mut matches = Vec::new();
        
        // Search working memory tree (includes compacted summaries)
        for item in self.working_memory_tree.iter() {
            if let Ok((_, value)) = item {
                if let Ok(message) = serde_json::from_slice::<ChatMessage>(&value) {
                    // Check if message is within date range
                    if message.timestamp >= cutoff_date {
                        // Check if content matches search query
                        let content_lower = message.content.to_lowercase();
                        if content_lower.contains(&query_lower) ||
                           self.is_relevant_to_query(&message, &query_lower) {
                            matches.push(message);
                        }
                    }
                }
            }
        }
        
        // Sort by relevance/timestamp
        matches.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        matches.truncate(10); // Limit to top 10 matches
        
        Ok(matches)
    }
    
    /// Check if message is relevant to search query (beyond simple text matching)
    fn is_relevant_to_query(&self, message: &ChatMessage, query: &str) -> bool {
        let content_lower = message.content.to_lowercase();
        
        // Look for code changes if query mentions "change", "edit", "modify"
        if query.contains("change") || query.contains("edit") || query.contains("modify") {
            return content_lower.contains("```") || 
                   content_lower.contains("diff") ||
                   content_lower.contains("modified") ||
                   content_lower.contains("updated");
        }
        
        // Look for file operations if query mentions "file", "create", "delete"
        if query.contains("file") || query.contains("create") || query.contains("delete") {
            return self.contains_file_references(&content_lower) ||
                   content_lower.contains("created") ||
                   content_lower.contains("deleted");
        }
        
        false
    }
    
    /// Smart file detection - recognizes any file based on common patterns
    fn contains_file_references(&self, content: &str) -> bool {
        // Look for file extensions (dot followed by 2-4 alphanumeric characters)
        if content.matches(|c: char| c == '.').count() > 0 {
            // Pattern: word.ext where ext is 2-4 characters
            let words: Vec<&str> = content.split_whitespace().collect();
            for word in words {
                if let Some(dot_pos) = word.rfind('.') {
                    let extension = &word[dot_pos + 1..];
                    // Valid file extension: 2-4 alphanumeric characters
                    if extension.len() >= 2 && extension.len() <= 4 && 
                       extension.chars().all(|c| c.is_alphanumeric()) {
                        return true;
                    }
                }
            }
        }
        
        // Look for path separators indicating file paths
        content.contains("/") || content.contains("\\") ||
        
        // Look for common file operation keywords
        content.contains("file") || content.contains("directory") || 
        content.contains("folder") || content.contains("path") ||
        
        // Look for programming-specific file indicators
        content.contains("main.") || content.contains("index.") || 
        content.contains("config.") || content.contains("package.") ||
        content.contains("readme") || content.contains("makefile") ||
        content.contains("dockerfile") || content.contains("cargo.toml") ||
        content.contains("package.json") || content.contains("requirements.txt") ||
        
        // Look for source code indicators (regardless of extension)
        content.contains("src/") || content.contains("lib/") || 
        content.contains("bin/") || content.contains("tests/") ||
        content.contains("build/") || content.contains("dist/") ||
        content.contains("node_modules/") || content.contains("target/")
    }
    
    // Get memory statistics
    pub async fn get_memory_stats(&self) -> Result<MemoryStats> {
        let total_messages = self.working_memory_tree.len();
        let total_background = self.background_storage_tree.len();
        let db_size = self.db.size_on_disk()?;
        let message_count = self.message_count.load(Ordering::Relaxed);
        let last_compaction = *self.last_background_move.lock().unwrap();
        
        Ok(MemoryStats {
            total_messages,
            message_count,
            total_summaries: total_background,
            last_compaction,
            db_size_bytes: db_size,
            needs_compaction: self.should_compact().await?,
        })
    }

    /// Background compaction method (static to avoid self reference issues)
    async fn compact_memory_background(
        db: sled::Db,
        working_memory_tree: sled::Tree,
        _background_storage_tree: sled::Tree,
        config: Config,
        metadata_tree: sled::Tree,
        last_background_move: Arc<Mutex<DateTime<Utc>>>,
        message_count: Arc<AtomicU64>,
    ) -> Result<()> {
        
        let count = message_count.load(Ordering::Relaxed);
        let reason = if count >= FORCE_COMPACTION_LIMIT {
            "conversation has grown very long"
        } else {
            "optimizing for better performance"
        };
        
        // Show compaction progress in status bar instead of direct print
        crate::thinking_display::PersistentStatusBar::set_ai_thinking(&format!("🗜️ Background compacting conversation memory ({} messages): {}", count, reason));
        
        // Get all messages
        let all_messages: Vec<ChatMessage> = working_memory_tree.iter()
            .filter_map(Result::ok)
            .filter_map(|(_, v)| serde_json::from_slice(&v).ok())
            .collect();
        
        if all_messages.len() <= WORKING_CONTEXT_TRUNCATED {
            // Memory optimal - show in status bar
            crate::thinking_display::PersistentStatusBar::set_ai_thinking("Memory already optimal, skipping compaction");
            return Ok(());
        }
        
        // Sort by timestamp
        let mut sorted_messages = all_messages;
        sorted_messages.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        // Smart compaction: preserve recent context for ongoing tasks
        let split_point = sorted_messages.len().saturating_sub(WORKING_CONTEXT_TRUNCATED);
        let (messages_to_summarize, recent_messages) = sorted_messages.split_at(split_point.max(1));
        
        if !messages_to_summarize.is_empty() {
            // Create comprehensive summary
            let summary = Self::create_comprehensive_summary_static(messages_to_summarize, &config).await?;
            
            // Clear old conversation data
            working_memory_tree.clear()?;
            
            // Store summary as a system message
            let summary_message = ChatMessage {
                role: MessageRole::System,
                content: summary,
                timestamp: if let Some(last_old_msg) = messages_to_summarize.last() {
                    last_old_msg.timestamp + chrono::Duration::seconds(1)
                } else {
                    Utc::now()
                },
                importance: MessageImportance::Important,
                metadata: MessageMetadata::default(),
            };
            
            let key = summary_message.timestamp.to_rfc3339();
            let value = serde_json::to_vec(&summary_message)?;
            working_memory_tree.insert(key.as_bytes(), value)?;
            
            // Re-add recent messages
            for message in recent_messages {
                let key = message.timestamp.to_rfc3339();
                let value = serde_json::to_vec(message)?;
                working_memory_tree.insert(key.as_bytes(), value)?;
            }
        }
        
        // Update metadata
        {
            let mut last_move = last_background_move.lock().unwrap();
            *last_move = Utc::now();
        }
        message_count.store((recent_messages.len() + 1) as u64, Ordering::Relaxed); // +1 for summary
        
        // Save metadata (scope the mutex guard properly)
        let last_move_str = {
            let last_move = last_background_move.lock().unwrap();
            last_move.to_rfc3339()
        };
        metadata_tree.insert("last_background_move", last_move_str.as_bytes())?;
        metadata_tree.insert("message_count", message_count.load(Ordering::Relaxed).to_string().as_bytes())?;
        
        // Flush database
        db.flush_async().await?;
        
        // Compaction completed - show in status bar
        crate::thinking_display::PersistentStatusBar::set_ai_thinking("🗜️ Background compaction completed");
        
        Ok(())
    }
    
    /// Static version of create_comprehensive_summary for background use
    async fn create_comprehensive_summary_static(messages: &[ChatMessage], config: &Config) -> Result<String> {
        if messages.is_empty() {
            return Ok("".to_string());
        }
        
        let conversation_text: String = messages.iter()
            .map(|msg| format!("[{}] {}: {}", 
                msg.timestamp.format("%H:%M"), 
                msg.role.as_str(), 
                msg.content))
            .collect::<Vec<_>>()
            .join("\n");
        
        let prompt = format!(
            "Create a comprehensive but concise summary of this conversation history. Focus on:\n\
            1. Key topics discussed\n\
            2. Important decisions made\n\
            3. Files or code worked on\n\
            4. User preferences learned\n\
            5. Context that would be valuable for continuing the conversation\n\
            \n\
            Keep the summary under 2000 words but capture all essential information.\n\
            \n\
            CONVERSATION HISTORY:\n\
            ---\n\
            {}\n\
            ---\n\
            \n\
            COMPREHENSIVE SUMMARY:",
            conversation_text
        );
        
        let summary = crate::gemini::query_gemini(&prompt, config).await?;
        Ok(format!("**Conversation Summary** ({})\n\n{}", 
            Utc::now().format("%Y-%m-%d %H:%M UTC"), 
            summary))
    }
    
    /// Calculate memory usage percentage until compaction is needed
    pub fn get_memory_percentage(&self) -> u8 {
        let message_count = self.message_count.load(Ordering::Relaxed);
        if message_count >= FORCE_COMPACTION_LIMIT {
            100
        } else if message_count >= COMPACTION_THRESHOLD {
            let progress_in_compaction_range = message_count - COMPACTION_THRESHOLD;
            let compaction_range_size = FORCE_COMPACTION_LIMIT - COMPACTION_THRESHOLD;
            let percentage = 50 + ((progress_in_compaction_range * 50) / compaction_range_size);
            percentage.min(100) as u8
        } else {
            let percentage = (message_count * 50) / COMPACTION_THRESHOLD;
            percentage.min(50) as u8
        }
    }
    
    /// Smart context retrieval for current task - leverages metadata for better reasoning
    pub async fn get_smart_context_for_task(&self, current_request: &str) -> Result<String> {
        let mut context_parts = Vec::new();
        
        // Get recent messages with rich metadata
        let recent_messages = self.get_recent_messages(10).await?;
        
        // Extract file-related context if current request involves files
        if current_request.to_lowercase().contains("file") || current_request.contains(".") {
            let file_context = self.extract_file_context(&recent_messages, current_request);
            if !file_context.is_empty() {
                context_parts.push(format!("RECENT FILE OPERATIONS:\n{}", file_context));
            }
        }
        
        // Extract error context for debugging tasks
        if current_request.to_lowercase().contains("fix") || current_request.to_lowercase().contains("error") {
            let error_context = self.extract_error_context(&recent_messages);
            if !error_context.is_empty() {
                context_parts.push(format!("RECENT ERRORS/ISSUES:\n{}", error_context));
            }
        }
        
        // Extract success patterns for similar tasks
        let success_context = self.extract_success_patterns(&recent_messages, current_request);
        if !success_context.is_empty() {
            context_parts.push(format!("SUCCESSFUL APPROACHES:\n{}", success_context));
        }
        
        // Standard conversation context (condensed)
        let conversation_context = recent_messages.iter()
            .take(5)
            .map(|msg| format!("{}: {}", msg.role, self.truncate_message_content(&msg.content)))
            .collect::<Vec<_>>()
            .join("\n");
        
        if !conversation_context.is_empty() {
            context_parts.push(format!("RECENT CONVERSATION:\n{}", conversation_context));
        }
        
        Ok(context_parts.join("\n\n"))
    }
    
    /// Extract file-related context from message metadata
    fn extract_file_context(&self, messages: &[ChatMessage], current_request: &str) -> String {
        let mut file_operations = Vec::new();
        
        for msg in messages.iter().rev().take(5) {
            if !msg.metadata.files_mentioned.is_empty() {
                let files_str = msg.metadata.files_mentioned.join(", ");
                if msg.metadata.success_indicator {
                    file_operations.push(format!("✓ Worked on: {}", files_str));
                } else {
                    file_operations.push(format!("⚠ Attempted: {}", files_str));
                }
            }
            
            if !msg.metadata.code_changes.is_empty() {
                let changes_str = msg.metadata.code_changes.join(", ");
                file_operations.push(format!("📝 Changes: {}", changes_str));
            }
        }
        
        // Find mentioned files in current request
        let request_lower = current_request.to_lowercase();
        for msg in messages {
            for file in &msg.metadata.files_mentioned {
                if request_lower.contains(&file.to_lowercase()) || current_request.contains(file) {
                    file_operations.push(format!("🎯 Previously worked on: {}", file));
                    break;
                }
            }
        }
        
        file_operations.join("\n")
    }
    
    /// Extract error context for debugging assistance
    fn extract_error_context(&self, messages: &[ChatMessage]) -> String {
        let mut error_info = Vec::new();
        
        for msg in messages.iter().rev().take(8) {
            if let Some(error_type) = &msg.metadata.error_type {
                error_info.push(format!("⚠ {}: {}", error_type, 
                    msg.content.lines().next().unwrap_or("").chars().take(100).collect::<String>()));
            }
            
            if msg.content.to_lowercase().contains("error") || msg.content.to_lowercase().contains("failed") {
                let error_line = msg.content.lines()
                    .find(|line| line.to_lowercase().contains("error") || line.to_lowercase().contains("failed"))
                    .unwrap_or("")
                    .chars().take(150).collect::<String>();
                if !error_line.is_empty() {
                    error_info.push(format!("❌ {}", error_line));
                }
            }
        }
        
        error_info.join("\n")
    }
    
    /// Extract successful patterns for similar task types
    fn extract_success_patterns(&self, messages: &[ChatMessage], current_request: &str) -> String {
        let mut success_patterns = Vec::new();
        let request_lower = current_request.to_lowercase();
        
        // Categorize current request
        let task_category = if request_lower.contains("edit") || request_lower.contains("modify") {
            "editing"
        } else if request_lower.contains("create") || request_lower.contains("build") {
            "creation"
        } else if request_lower.contains("fix") || request_lower.contains("debug") {
            "debugging"
        } else if request_lower.contains("analyze") || request_lower.contains("explain") {
            "analysis"
        } else {
            "general"
        };
        
        for msg in messages.iter().rev().take(10) {
            if msg.metadata.success_indicator {
                let content_lower = msg.content.to_lowercase();
                let matches_category = match task_category {
                    "editing" => content_lower.contains("edit") || content_lower.contains("modif"),
                    "creation" => content_lower.contains("creat") || content_lower.contains("build"),
                    "debugging" => content_lower.contains("fix") || content_lower.contains("debug"),
                    "analysis" => content_lower.contains("analyz") || content_lower.contains("explain"),
                    _ => true,
                };
                
                if matches_category {
                    if let Some(task_type) = &msg.metadata.task_type {
                        success_patterns.push(format!("✓ {}: {}", task_type, 
                            msg.content.lines().next().unwrap_or("").chars().take(120).collect::<String>()));
                    }
                }
            }
        }
        
        success_patterns.join("\n")
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_messages: usize,
    pub message_count: u64,
    pub total_summaries: usize,
    pub last_compaction: DateTime<Utc>,
    pub db_size_bytes: u64,
    pub needs_compaction: bool,
}

impl MessageRole {
    pub fn as_str(&self) -> &str {
        match self {
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            MessageRole::System => "System",
        }
    }
}
