use anyhow::Result;
use std::path::Path;
use imara_diff::{
    Algorithm, Diff, InternedInput,
};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};
use crate::cli::colors::*;

/// Code diff display system using imara-diff for diffing and syntect for syntax highlighting
pub struct CodeDiffDisplay {
    syntax_set: SyntaxSet,
    theme: Theme,
}

impl CodeDiffDisplay {
    pub fn new() -> Result<Self> {
        let syntax_set = SyntaxSet::load_defaults_newlines();
        
        // Use built-in dark theme (base16-eighties.dark) instead of custom theme
        let theme_set = ThemeSet::load_defaults();
        let theme = theme_set.themes.get("base16-eighties.dark")
            .or_else(|| theme_set.themes.get("Solarized (dark)"))
            .or_else(|| theme_set.themes.get("InspiredGitHub"))
            .unwrap_or(theme_set.themes.values().next().unwrap())
            .clone();

        Ok(Self {
            syntax_set,
            theme,
        })
    }

    /// Display a clean diff showing only the changes between original and modified code
    /// Format diff content for ratatui display (returns Vec<String> instead of printing)
    pub fn format_diff_for_ratatui(
        &self,
        original: &str,
        modified: &str,
        file_path: &Path,
    ) -> Result<Vec<String>> {
        let language = self.detect_language(file_path);
        let syntax = self.syntax_set.find_syntax_by_name(&language)
            .or_else(|| self.syntax_set.find_syntax_by_extension(&language))
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());

        let mut output = Vec::new();
        output.push(format!("📝 Code Changes in {}", file_path.display()));
        output.push("─────────────────────────────────────────────────".to_string());

        // Using new imara-diff 0.2.0 API
        let input = InternedInput::new(original, modified);
        let diff = Diff::compute(Algorithm::Histogram, &input);
        
        // Get the original lines for reference
        let original_lines: Vec<&str> = original.lines().collect();
        let modified_lines: Vec<&str> = modified.lines().collect();
        
        for hunk in diff.hunks() {
            // Process removed lines
            for idx in hunk.before.clone() {
                if diff.is_removed(idx) {
                    if let Some(line) = original_lines.get(idx as usize) {
                        output.push(self.format_removed_line(line, (idx + 1) as usize, syntax)?);
                    }
                }
            }
            
            // Process added lines
            for idx in hunk.after.clone() {
                if diff.is_added(idx) {
                    if let Some(line) = modified_lines.get(idx as usize) {
                        output.push(self.format_added_line(line, (idx + 1) as usize, syntax)?);
                    }
                }
            }
            
            // Process unchanged lines in context (optional, for better context)
            for idx in hunk.before.clone() {
                if !diff.is_removed(idx) {
                    if let Some(line) = original_lines.get(idx as usize) {
                        output.push(self.format_unchanged_line(line, (idx + 1) as usize, syntax)?);
                    }
                }
            }
        }

        output.push("─────────────────────────────────────────────────".to_string());
        
        Ok(output)
    }

    pub fn display_diff(&self, _original: &str, _modified: &str, _file_path: &Path) -> Result<()> {
        // Silent method - no printing to avoid interfering with ratatui display
        // The diff will be shown in the function result summary instead
        Ok(())
    }

    fn format_added_line(&self, line: &str, line_num: usize, syntax: &syntect::parsing::SyntaxReference) -> Result<String> {
        let mut highlighter = HighlightLines::new(syntax, &self.theme);
        let ranges = highlighter.highlight_line(line, &self.syntax_set)?;
        let highlighted = as_24_bit_terminal_escaped(&ranges[..], false);
        
        Ok(format!("+  {:4} {}", line_num, highlighted.trim_end()))
    }

    fn display_added_line(&self, line: &str, line_num: usize, syntax: &syntect::parsing::SyntaxReference) -> Result<()> {
        // Legacy method - uses the formatter
        let formatted = self.format_added_line(line, line_num, syntax)?;
        println!("{}", formatted);
        Ok(())
    }

    fn format_removed_line(&self, line: &str, line_num: usize, syntax: &syntect::parsing::SyntaxReference) -> Result<String> {
        let mut highlighter = HighlightLines::new(syntax, &self.theme);
        let ranges = highlighter.highlight_line(line, &self.syntax_set)?;
        let highlighted = as_24_bit_terminal_escaped(&ranges[..], false);
        
        Ok(format!("-  {:4} {}", line_num, highlighted.trim_end()))
    }

    fn format_unchanged_line(&self, line: &str, line_num: usize, syntax: &syntect::parsing::SyntaxReference) -> Result<String> {
        let mut highlighter = HighlightLines::new(syntax, &self.theme);
        let ranges = highlighter.highlight_line(line, &self.syntax_set)?;
        let highlighted = as_24_bit_terminal_escaped(&ranges[..], false);
        
        Ok(format!("   {:4} {}", line_num, highlighted.trim_end()))
    }

    fn display_removed_line(&self, line: &str, line_num: usize, syntax: &syntect::parsing::SyntaxReference) -> Result<()> {
        let formatted = self.format_removed_line(line, line_num, syntax)?;
        println!("{}", formatted);
        Ok(())
    }

    fn display_unchanged_line(&self, line: &str, line_num: usize, syntax: &syntect::parsing::SyntaxReference) -> Result<()> {
        let formatted = self.format_unchanged_line(line, line_num, syntax)?;
        println!("{}", formatted);
        Ok(())
    }

    /// Detect programming language from file extension
    fn detect_language(&self, file_path: &Path) -> String {
        if let Some(extension) = file_path.extension().and_then(|e| e.to_str()) {
            match extension.to_lowercase().as_str() {
                "rs" => "Rust".to_string(),
                "js" => "JavaScript".to_string(),
                "jsx" => "JavaScript (JSX)".to_string(),
                "ts" => "TypeScript".to_string(),
                "tsx" => "TypeScript (TSX)".to_string(),
                "py" => "Python".to_string(),
                "html" | "htm" => "HTML".to_string(),
                "css" => "CSS".to_string(),
                "json" => "JSON".to_string(),
                "toml" => "TOML".to_string(),
                "yaml" | "yml" => "YAML".to_string(),
                "md" => "Markdown".to_string(),
                "sh" => "Bash".to_string(),
                "go" => "Go".to_string(),
                "c" => "C".to_string(),
                "cpp" | "cc" | "cxx" => "C++".to_string(),
                "java" => "Java".to_string(),
                "php" => "PHP".to_string(),
                "rb" => "Ruby".to_string(),
                "cs" => "C#".to_string(),
                "xml" => "XML".to_string(),
                "sql" => "SQL".to_string(),
                _ => "Plain Text".to_string(),
            }
        } else {
            "Plain Text".to_string()
        }
    }
}

/// Display a simple code file with syntax highlighting
pub fn display_code_file(file_path: &Path, content: &str) -> Result<()> {
    let display = CodeDiffDisplay::new()?;
    let language = display.detect_language(file_path);
    let syntax = display.syntax_set.find_syntax_by_name(&language)
        .or_else(|| display.syntax_set.find_syntax_by_extension(&language))
        .unwrap_or_else(|| display.syntax_set.find_syntax_plain_text());

    println!("\n{}📄 {}{}", EMERALD_BRIGHT, file_path.display(), RESET);
    println!("{}─────────────────────────────────────────────────{}", GRAY_DIM, RESET);
    
    let mut highlighter = HighlightLines::new(syntax, &display.theme);
    for (i, line) in LinesWithEndings::from(content).enumerate() {
        let ranges = highlighter.highlight_line(line, &display.syntax_set)?;
        let highlighted = as_24_bit_terminal_escaped(&ranges[..], false);
        println!("{}{:4}{} {}", 
                GRAY_DIM, i + 1, RESET, 
                highlighted.trim_end());
    }
    
    println!("{}─────────────────────────────────────────────────{}", GRAY_DIM, RESET);
    println!();
    
    Ok(())
}

/// Display only a summary of what a file does, not its content
pub fn display_file_summary(file_path: &Path, content: &str) -> Result<String> {
    // This function should not print directly to stdout as it interferes with the ratatui UI.
    // It should return the formatted summary string for the caller to display.
    let summary = analyze_file_purpose(file_path, content)?;
    Ok(format!("📖 **File Summary for {}**\n{}", file_path.display(), summary))
}

/// Analyze the purpose of a file based on its content and structure
fn analyze_file_purpose(file_path: &Path, content: &str) -> Result<String> {
    let extension = file_path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "html" | "htm" => analyze_html_file(content),
        "js" | "jsx" | "ts" | "tsx" => analyze_javascript_file(content),
        "py" => analyze_python_file(content),
        "rs" => analyze_rust_file(content),
        "css" => analyze_css_file(content),
        "json" => analyze_json_file(content),
        "md" => analyze_markdown_file(content),
        _ => analyze_generic_file(content),
    }
}

fn analyze_html_file(content: &str) -> Result<String> {
    let mut features = Vec::new();
    
    // Extract title
    if let Some(title_start) = content.find("<title>") {
        if let Some(title_end) = content[title_start + 7..].find("</title>") {
            let title = &content[title_start + 7..title_start + 7 + title_end];
            features.push(format!("**Title**: {}", title));
        }
    }
    
    // Check for common frameworks/libraries
    if content.contains("react") || content.contains("React") {
        features.push("Uses **React**".to_string());
    }
    if content.contains("vue") || content.contains("Vue") {
        features.push("Uses **Vue**".to_string());
    }
    if content.contains("angular") || content.contains("Angular") {
        features.push("Uses **Angular**".to_string());
    }
    if content.contains("bootstrap") {
        features.push("Uses **Bootstrap**".to_string());
    }
    
    // Check for interactive features
    if content.contains("canvas") || content.contains("<canvas") {
        features.push("Contains **Canvas** graphics".to_string());
    }
    if content.contains("webgl") || content.contains("WebGL") {
        features.push("Uses **WebGL**".to_string());
    }
    
    // Count script tags
    let script_count = content.matches("<script").count();
    if script_count > 0 {
        features.push(format!("Contains **{}** script sections", script_count));
    }
    
    let purpose = if features.is_empty() {
        "Basic HTML document".to_string()
    } else {
        format!("HTML document with: {}", features.join(", "))
    };
    
    Ok(purpose)
}

fn analyze_javascript_file(content: &str) -> Result<String> {
    let mut features = Vec::new();
    
    // Check for functions
    let function_count = content.matches("function ").count() + content.matches("=> ").count();
    if function_count > 0 {
        features.push(format!("**{}** functions", function_count));
    }
    
    // Check for classes
    let class_count = content.matches("class ").count();
    if class_count > 0 {
        features.push(format!("**{}** classes", class_count));
    }
    
    // Check for common patterns
    if content.contains("import ") || content.contains("require(") {
        features.push("**ES6/CommonJS** modules".to_string());
    }
    if content.contains("async ") || content.contains("await ") {
        features.push("**Async/await** patterns".to_string());
    }
    if content.contains("fetch(") || content.contains("axios") {
        features.push("**API calls**".to_string());
    }
    
    let purpose = if features.is_empty() {
        "JavaScript code".to_string()
    } else {
        format!("JavaScript with: {}", features.join(", "))
    };
    
    Ok(purpose)
}

fn analyze_python_file(content: &str) -> Result<String> {
    let mut features = Vec::new();
    
    // Check for functions and classes
    let def_count = content.matches("def ").count();
    let class_count = content.matches("class ").count();
    
    if def_count > 0 {
        features.push(format!("**{}** functions", def_count));
    }
    if class_count > 0 {
        features.push(format!("**{}** classes", class_count));
    }
    
    // Check for common imports
    if content.contains("import requests") || content.contains("from requests") {
        features.push("**HTTP requests** (requests library)".to_string());
    }
    if content.contains("import pandas") || content.contains("from pandas") {
        features.push("**Data analysis** (pandas)".to_string());
    }
    if content.contains("import numpy") || content.contains("from numpy") {
        features.push("**Numerical computing** (numpy)".to_string());
    }
    if content.contains("import flask") || content.contains("from flask") {
        features.push("**Web framework** (Flask)".to_string());
    }
    if content.contains("import django") || content.contains("from django") {
        features.push("**Web framework** (Django)".to_string());
    }
    
    let purpose = if features.is_empty() {
        "Python script".to_string()
    } else {
        format!("Python script with: {}", features.join(", "))
    };
    
    Ok(purpose)
}

fn analyze_rust_file(content: &str) -> Result<String> {
    let mut features = Vec::new();
    
    // Check for functions, structs, enums
    let fn_count = content.matches("fn ").count();
    let struct_count = content.matches("struct ").count();
    let enum_count = content.matches("enum ").count();
    
    if fn_count > 0 {
        features.push(format!("**{}** functions", fn_count));
    }
    if struct_count > 0 {
        features.push(format!("**{}** structs", struct_count));
    }
    if enum_count > 0 {
        features.push(format!("**{}** enums", enum_count));
    }
    
    // Check for common patterns
    if content.contains("async fn") || content.contains("await") {
        features.push("**Async** functionality".to_string());
    }
    if content.contains("tokio") {
        features.push("**Tokio** async runtime".to_string());
    }
    if content.contains("serde") {
        features.push("**Serialization** (serde)".to_string());
    }
    
    let purpose = if features.is_empty() {
        "Rust code".to_string()
    } else {
        format!("Rust code with: {}", features.join(", "))
    };
    
    Ok(purpose)
}

fn analyze_css_file(content: &str) -> Result<String> {
    let selector_count = content.matches('{').count();
    let media_queries = content.matches("@media").count();
    let animations = content.matches("@keyframes").count();
    
    let mut features = Vec::new();
    features.push(format!("**{}** CSS rules", selector_count));
    
    if media_queries > 0 {
        features.push(format!("**{}** media queries (responsive)", media_queries));
    }
    if animations > 0 {
        features.push(format!("**{}** animations", animations));
    }
    
    Ok(format!("CSS stylesheet with: {}", features.join(", ")))
}

fn analyze_json_file(content: &str) -> Result<String> {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content) {
        match &parsed {
            serde_json::Value::Object(obj) => {
                Ok(format!("JSON object with **{}** keys", obj.len()))
            }
            serde_json::Value::Array(arr) => {
                Ok(format!("JSON array with **{}** items", arr.len()))
            }
            _ => Ok("JSON data".to_string())
        }
    } else {
        Ok("JSON file (invalid/malformed)".to_string())
    }
}

fn analyze_markdown_file(content: &str) -> Result<String> {
    let heading_count = content.lines().filter(|line| line.starts_with('#')).count();
    let link_count = content.matches("](").count();
    let code_blocks = content.matches("```").count() / 2;
    
    let mut features = Vec::new();
    if heading_count > 0 {
        features.push(format!("**{}** headings", heading_count));
    }
    if link_count > 0 {
        features.push(format!("**{}** links", link_count));
    }
    if code_blocks > 0 {
        features.push(format!("**{}** code blocks", code_blocks));
    }
    
    let purpose = if features.is_empty() {
        "Markdown document".to_string()
    } else {
        format!("Markdown document with: {}", features.join(", "))
    };
    
    Ok(purpose)
}

fn analyze_generic_file(content: &str) -> Result<String> {
    let lines = content.lines().count();
    let words = content.split_whitespace().count();
    let chars = content.chars().count();
    
    Ok(format!("Text file with **{}** lines, **{}** words, **{}** characters", 
              lines, words, chars))
}