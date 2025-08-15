use anyhow::{anyhow, Context, Result};
use kuchiki::{parse_html, traits::*};
use chrono::Datelike;
use html2text::from_read;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use url::Url;
use crate::config::Config;
use crate::gemini;
use crate::cli::colors::{EMERALD_BRIGHT, BLUE_BRIGHT, RESET};

// Research system for intelligent web scraping and information gathering
#[derive(Debug, Clone)]
pub struct ResearchEngine {
    client: Client,
    config: Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    pub url: String,
    pub title: String,
    pub summary: String,
    pub content_type: ContentType,
    pub confidence: f32,
    pub extracted_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Documentation,
    Tutorial,
    Reference,
    BlogPost,
    Repository,
    Unknown,
}

impl ResearchEngine {
    pub fn new(config: &Config) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent("Glimmer Research Bot 1.0")
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            config: config.clone(),
        }
    }

    /// Intelligent research trigger - determines if we need to lookup information
    pub async fn should_research(&self, input: &str, context: &str) -> bool {
        // Patterns that indicate missing information
        let research_indicators = [
            "i don't know", "not sure", "tell me about", "what is", "how does",
            "explain", "documentation for", "how to", "tutorial", "guide",
            "latest", "current", "recent", "new features", "changes"
        ];
        
        let input_lower = input.to_lowercase();
        
        // Check for explicit research requests
        if research_indicators.iter().any(|&pattern| input_lower.contains(pattern)) {
            return true;
        }
        
        // Check for URLs in the input
        if input_lower.contains("http") || input_lower.contains("github.com") {
            return true;
        }
        
        // Check if context suggests we need more information
        if context.contains("error") || context.contains("failed") {
            return input_lower.contains("fix") || input_lower.contains("solve");
        }
        
        false
    }

    /// Extract URLs from user input
    pub fn extract_urls(&self, input: &str) -> Vec<String> {
        let mut urls = Vec::new();
        
        // Simple URL extraction
        for word in input.split_whitespace() {
            if let Ok(url) = Url::parse(word) {
                if url.scheme() == "http" || url.scheme() == "https" {
                    urls.push(url.to_string());
                }
            }
        }
        
        urls
    }

    /// Perform intelligent research on a topic or URL
    pub async fn research(&self, query: &str) -> Result<ResearchResult> {
        // Research progress shown via thinking display
        
        // Check if it's a URL
        if let Ok(url) = Url::parse(query) {
            self.scrape_url(&url).await
        } else {
            // Perform topic search
            self.search_and_research_topic(query).await
        }
    }

    /// Search for topic and research best result
    async fn search_and_research_topic(&self, topic: &str) -> Result<ResearchResult> {
        // Progress shown via thinking display
        
        // Get the best URL for this topic
        let search_url = self.get_best_search_result(topic).await?;
        
        // Research the found URL
        // Progress shown via thinking display
        self.scrape_url(&search_url).await
    }
    
    /// Find the best search result for a topic
    async fn get_best_search_result(&self, topic: &str) -> Result<Url> {
        // Use a combination of knowledge-based suggestions and web search
        let reliable_sources = self.get_reliable_sources_for_topic(topic);
        
        if let Some(source) = reliable_sources.first() {
            return Url::parse(source).context("Invalid URL from reliable source");
        }
        
        // Fallback: create a comprehensive search result
        self.create_knowledge_based_result(topic).await
    }
    
    /// Get reliable sources for common topics
    fn get_reliable_sources_for_topic(&self, topic: &str) -> Vec<String> {
        let topic_lower = topic.to_lowercase();
        let mut sources = Vec::new();
        
        // Programming and development topics
        if topic_lower.contains("rust") {
            sources.push("https://doc.rust-lang.org/book/".to_string());
            sources.push("https://forge.rust-lang.org/".to_string());
        } else if topic_lower.contains("javascript") || topic_lower.contains("js") {
            sources.push("https://developer.mozilla.org/en-US/docs/Web/JavaScript".to_string());
        } else if topic_lower.contains("python") {
            sources.push("https://docs.python.org/3/".to_string());
        } else if topic_lower.contains("react") {
            sources.push("https://react.dev/".to_string());
        } else if topic_lower.contains("nodejs") || topic_lower.contains("node") {
            sources.push("https://nodejs.org/docs/latest/api/".to_string());
        }
        
        // Academic and reference topics
        else if topic_lower.contains("gematria") || topic_lower.contains("numerology") {
            sources.push("https://en.wikipedia.org/wiki/Gematria".to_string());
        } else if topic_lower.contains("mathematics") || topic_lower.contains("math") {
            sources.push("https://en.wikipedia.org/wiki/Mathematics".to_string());
        } else if topic_lower.contains("physics") {
            sources.push("https://en.wikipedia.org/wiki/Physics".to_string());
        }
        
        // Technology and tools
        else if topic_lower.contains("docker") {
            sources.push("https://docs.docker.com/".to_string());
        } else if topic_lower.contains("kubernetes") || topic_lower.contains("k8s") {
            sources.push("https://kubernetes.io/docs/".to_string());
        } else if topic_lower.contains("git") && !topic_lower.contains("github") {
            sources.push("https://git-scm.com/docs".to_string());
        }
        
        sources
    }
    
    /// Create a knowledge-based research result when no URL is found
    async fn create_knowledge_based_result(&self, topic: &str) -> Result<Url> {
        // Instead of guessing Wikipedia URLs, use Wikipedia search API
        let search_query = urlencoding::encode(topic);
        let _wiki_search_url = format!("https://en.wikipedia.org/w/api.php?action=opensearch&search={}&limit=1&format=json", search_query);
        
        // Try to get a real Wikipedia page from search results
        match self.get_wikipedia_page_from_search(&search_query).await {
            Ok(url) => Ok(url),
            Err(_) => {
                // Ultimate fallback: use a generic search engine
                let search_url = format!("https://duckduckgo.com/?q={}", search_query);
                Url::parse(&search_url).context("Failed to create search URL")
            }
        }
    }
    
    /// Get actual Wikipedia page from search API
    async fn get_wikipedia_page_from_search(&self, query: &str) -> Result<Url> {
        let api_url = format!(
            "https://en.wikipedia.org/w/api.php?action=opensearch&search={}&limit=1&format=json&redirects=resolve",
            query
        );
        
        let response = self.client.get(&api_url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .context("Failed to query Wikipedia API")?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Wikipedia API returned error: {}", response.status()));
        }
        
        let json_text = response.text().await
            .context("Failed to read Wikipedia API response")?;
        
        // Parse the OpenSearch JSON format: [query, [titles], [descriptions], [urls]]
        let parsed: serde_json::Value = serde_json::from_str(&json_text)
            .context("Failed to parse Wikipedia API response")?;
        
        if let Some(urls_array) = parsed.get(3).and_then(|v| v.as_array()) {
            if let Some(first_url) = urls_array.first().and_then(|v| v.as_str()) {
                return Url::parse(first_url).context("Invalid URL from Wikipedia API");
            }
        }
        
        Err(anyhow!("No Wikipedia page found for query: {}", query))
    }

    /// Scrape and analyze a URL
    async fn scrape_url(&self, url: &Url) -> Result<ResearchResult> {
        let response = self.client.get(url.as_str())
            .send()
            .await
            .context("Failed to fetch URL")?;

        if !response.status().is_success() {
            return Err(anyhow!("HTTP error: {}", response.status()));
        }

        let html = response.text().await.context("Failed to read response body")?;
        self.extract_information(url, &html).await
    }

    /// Extract and analyze information from HTML
    async fn extract_information(&self, url: &Url, html: &str) -> Result<ResearchResult> {
        let (title, content) = {
            let document = parse_html().one(html);
            
            // Extract title
            let title: String = document
                .select("title")
                .unwrap()
                .next()
                .map(|node| node.text_contents().trim().to_string())
                .unwrap_or_else(|| "Untitled".to_string());

            // Extract main content using kuchiki
            let content = self.extract_main_content(&document);
            (title, content)
        }; // `document` (which is !Send) is dropped here

        let text_content = from_read(content.as_bytes(), 120); // 120 chars per line
        
        // Determine content type
        let content_type = self.classify_content(&title, &text_content, url);
        
        // Create intelligent summary
        let summary = self.create_intelligent_summary(&title, &text_content, &content_type).await?;
        
        Ok(ResearchResult {
            url: url.to_string(),
            title: title.clone(),
            summary,
            content_type: content_type.clone(),
            confidence: self.calculate_confidence_score(url, &title, &text_content, &content_type),
            extracted_at: chrono::Utc::now(),
        })
    }

    /// Extract main content from the document
    fn extract_main_content(&self, document: &kuchiki::NodeRef) -> String {
        // Priority order for content extraction
        let content_selectors = [
            "main", "article", ".content", ".post-content", 
            ".entry-content", "#content", ".markdown-body", 
            ".readme", ".documentation"
        ];
        
        for selector in &content_selectors {
            if let Ok(mut selection) = document.select(selector) {
                if let Some(node) = selection.next() {
                    return node.text_contents();
                }
            }
        }
        
        // Fallback to body content, excluding navigation and footer
        let body = document.select("body").unwrap().next()
            .map(|node| node.text_contents())
            .unwrap_or_default();
            
        // Remove script and style content
        body.lines()
            .filter(|line| !line.trim().is_empty())
            .take(100) // Limit to first 100 lines
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Classify the type of content
    fn classify_content(&self, title: &str, content: &str, url: &Url) -> ContentType {
        let combined = format!("{} {}", title, content).to_lowercase();
        let url_str = url.to_string().to_lowercase();
        
        if url_str.contains("github.com") {
            return ContentType::Repository;
        }
        
        if combined.contains("documentation") || combined.contains("api reference") {
            return ContentType::Documentation;
        }
        
        if combined.contains("tutorial") || combined.contains("how to") || combined.contains("guide") {
            return ContentType::Tutorial;
        }
        
        if combined.contains("reference") || combined.contains("manual") {
            return ContentType::Reference;
        }
        
        if url_str.contains("blog") || combined.contains("posted by") || combined.contains("author") {
            return ContentType::BlogPost;
        }
        
        ContentType::Unknown
    }

    /// Calculate confidence score based on source quality and content
    fn calculate_confidence_score(&self, url: &Url, title: &str, content: &str, content_type: &ContentType) -> f32 {
        let mut confidence: f32 = 0.0;
        let url_str = url.to_string().to_lowercase();
        
        // Source reliability scoring (40% of total confidence)
        let source_score = if url_str.contains("wikipedia.org") || url_str.contains("github.com") {
            0.35 // High reliability
        } else if url_str.contains(".edu") || url_str.contains(".gov") {
            0.40 // Very high reliability
        } else if url_str.contains("docs.") || url_str.contains("documentation") {
            0.35 // High for official docs
        } else if url_str.contains("stackoverflow.com") || url_str.contains("developer.mozilla.org") {
            0.30 // Good community sources
        } else if url_str.contains("medium.com") || url_str.contains("dev.to") {
            0.20 // Medium reliability blogs
        } else {
            0.15 // Unknown sources
        };
        confidence += source_score;
        
        // Content quality scoring (30% of total confidence)
        let content_quality = if content.len() > 2000 && title.len() > 10 {
            0.25 // Good length content
        } else if content.len() > 500 {
            0.20 // Adequate content
        } else {
            0.10 // Short content
        };
        confidence += content_quality;
        
        // Content type bonus (20% of total confidence)
        let type_bonus = match content_type {
            ContentType::Documentation => 0.20,
            ContentType::Reference => 0.18,
            ContentType::Tutorial => 0.15,
            ContentType::Repository => 0.12,
            ContentType::BlogPost => 0.10,
            ContentType::Unknown => 0.05,
        };
        confidence += type_bonus;
        
        // Freshness indicator (10% of total confidence)
        // If URL contains recent year indicators
        let current_year = chrono::Utc::now().year();
        let freshness_bonus = if url_str.contains(&current_year.to_string()) || 
                                url_str.contains(&(current_year - 1).to_string()) {
            0.10
        } else {
            0.05
        };
        confidence += freshness_bonus;
        
        confidence.min(1.0) // Cap at 1.0
    }

    /// Create an intelligent summary using Gemini
    async fn create_intelligent_summary(&self, title: &str, content: &str, content_type: &ContentType) -> Result<String> {
        // Limit content size for summarization
        let truncated_content = if content.len() > 8000 {
            format!("{}...", &content[..8000])
        } else {
            content.to_string()
        };
        
        let content_type_str = match content_type {
            ContentType::Documentation => "documentation",
            ContentType::Tutorial => "tutorial",
            ContentType::Reference => "reference material", 
            ContentType::BlogPost => "blog post",
            ContentType::Repository => "repository",
            ContentType::Unknown => "web content",
        };
        
        let prompt = format!(
            "Analyze and summarize this {} concisely. Focus on:\n\
            1. Main purpose and key points\n\
            2. Important technical details\n\
            3. Practical applications\n\
            4. Key takeaways for a developer\n\
            \n\
            Keep the summary under 500 words and make it actionable.\n\
            \n\
            TITLE: {}\n\
            \n\
            CONTENT:\n\
            {}\n\
            \n\
            SUMMARY:",
            content_type_str, title, truncated_content
        );
        
        // Progress shown via thinking display
        gemini::query_gemini(&prompt, &self.config).await
    }

    /// Get research suggestions for a query
    pub fn get_research_suggestions(&self, input: &str) -> Vec<String> {
        let input_lower = input.to_lowercase();
        let mut suggestions = Vec::new();
        
        // Technology-specific suggestions
        if input_lower.contains("rust") {
            suggestions.push("https://doc.rust-lang.org/std/".to_string());
            suggestions.push("https://docs.rs/".to_string());
        }
        
        if input_lower.contains("javascript") || input_lower.contains("js") {
            suggestions.push("https://developer.mozilla.org/en-US/docs/Web/JavaScript".to_string());
        }
        
        if input_lower.contains("python") {
            suggestions.push("https://docs.python.org/3/".to_string());
        }
        
        suggestions
    }
}

/// Auto-research capability that can be triggered during conversations
pub async fn auto_research_if_needed(input: &str, context: &str, config: &Config) -> Option<ResearchResult> {
    let research = ResearchEngine::new(config);
    
    if !research.should_research(input, context).await {
        return None;
    }
    
    // Extract URLs from input
    let urls = research.extract_urls(input);
    if let Some(url) = urls.first() {
        println!("{}🤖 Auto-research triggered for: {}{}", EMERALD_BRIGHT, url, RESET);
        return research.research(url).await.ok();
    }
    
    // Check for research suggestions
    let suggestions = research.get_research_suggestions(input);
    if let Some(suggestion) = suggestions.first() {
        println!("{}💡 Research suggestion: {}{}", BLUE_BRIGHT, suggestion, RESET);
        return research.research(suggestion).await.ok();
    }
    
    None
}

/// Check if the query should trigger research
pub fn should_perform_research(input: &str) -> bool {
    let input_lower = input.to_lowercase();
    
    let research_indicators = [
        "how does", "how do", "explain how", "what are the differences",
        "compare", "vs", "versus", "alternatives to", "best practices for",
        "pros and cons", "advantages", "disadvantages", "history of",
        "evolution of", "trends in", "future of", "state of", "research",
        "documentation for", "docs for", "tell me about"
    ];
    
    let complex_topics = [
        "programming", "software", "technology", "algorithm", "framework",
        "language", "database", "architecture", "design pattern", "api",
        "machine learning", "ai", "blockchain", "cloud", "security"
    ];
    
    let has_research_indicator = research_indicators.iter().any(|&indicator| input_lower.contains(indicator));
    let has_complex_topic = complex_topics.iter().any(|&topic| input_lower.contains(topic));
    
    has_research_indicator || (has_complex_topic && input_lower.contains("?"))
}

/// Perform intelligent research for complex queries
pub async fn perform_intelligent_research(input: &str, config: &Config) -> anyhow::Result<String> {
    let research_engine = ResearchEngine::new(config);
    
    // Extract URLs from input if any
    let urls = research_engine.extract_urls(input);
    
    let research_result = if let Some(url) = urls.first() {
        // Research the specific URL
        research_engine.research(url).await?
    } else {
        // Research the topic
        research_engine.research(input).await?
    };
    
    // Format the research result for display
    let response = format!(
        "## Research Result: {}\n\n\
        **Source**: {}\n\n\
        **Summary**:\n{}\n\n\
        **Content Type**: {:?}\n\
        **Extracted**: {}",
        research_result.title,
        research_result.url,
        research_result.summary,
        research_result.content_type,
        research_result.extracted_at.format("%Y-%m-%d %H:%M UTC")
    );
    
    Ok(response)
}