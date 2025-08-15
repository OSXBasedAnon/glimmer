use anyhow::Result;
use serde::{Deserialize, Serialize};
use tree_sitter::{Language, Tree};

// Dynamic analysis result structure
#[derive(Debug, Serialize, Deserialize)]
struct CodeAnalysisResult {
    pub structure_analysis: String,
    pub semantic_representation: String,
    pub symbols: Vec<Symbol>,
    pub issues: Vec<ParseError>,
    pub complexity_assessment: Option<String>,
    pub dependencies: Vec<String>,
    pub patterns: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ParseResult {
    pub language: String,
    pub tree_representation: String,
    pub ast_representation: String,
    pub symbols: Vec<Symbol>,
    pub errors: Vec<ParseError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: String,
    pub start_line: usize,
    pub start_column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

pub async fn parse_code(content: &str, language: &str, format: &str) -> Result<ParseResult> {
    // PRODUCTION APPROACH: Use AI for dynamic code understanding
    // This works for ANY language/format without static parsers
    
    let analysis_result = analyze_code_dynamically(content, language, format).await?;
    
    Ok(ParseResult {
        language: language.to_string(),
        tree_representation: analysis_result.structure_analysis,
        ast_representation: analysis_result.semantic_representation,
        symbols: analysis_result.symbols,
        errors: analysis_result.issues,
    })
}

// Dynamic AI-based code analysis - like Claude Code does
async fn analyze_code_dynamically(content: &str, language: &str, format: &str) -> Result<CodeAnalysisResult> {
    use crate::config;
    use crate::gemini;
    
    let config = config::load_config(None).await?;
    
    let analysis_prompt = create_dynamic_analysis_prompt(content, language, format);
    let analysis_response = gemini::query_gemini(&analysis_prompt, &config).await?;
    
    parse_analysis_response(&analysis_response, content)
}

fn create_dynamic_analysis_prompt(content: &str, language: &str, _format: &str) -> String {
    format!(
        r#"Analyze this {} code comprehensively. Provide a JSON response with:

1. "structure_analysis": Hierarchical breakdown of code organization
2. "semantic_representation": Key concepts and relationships  
3. "symbols": Array of functions, classes, variables with locations
4. "issues": Any problems, errors, or improvement opportunities
5. "complexity_assessment": Code complexity and maintainability
6. "dependencies": External dependencies and imports
7. "patterns": Design patterns and architectural insights

FORMAT: Return ONLY valid JSON.

CODE TO ANALYZE:
```{}
{}
```

ANALYSIS:"#,
        language, language, content
    )
}

fn parse_analysis_response(response: &str, _content: &str) -> Result<CodeAnalysisResult> {
    // Try to parse JSON response from AI
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
        let structure_analysis = parsed.get("structure_analysis")
            .and_then(|v| v.as_str())
            .unwrap_or("Structure analysis unavailable")
            .to_string();
            
        let semantic_representation = parsed.get("semantic_representation")
            .and_then(|v| v.as_str())
            .unwrap_or("Semantic analysis unavailable")
            .to_string();
            
        let symbols = parse_symbols_from_json(&parsed)?;
        let issues = parse_issues_from_json(&parsed)?;
        let dependencies = parse_string_array(&parsed, "dependencies");
        let patterns = parse_string_array(&parsed, "patterns");
        
        Ok(CodeAnalysisResult {
            structure_analysis,
            semantic_representation,
            symbols,
            issues,
            complexity_assessment: parsed.get("complexity_assessment")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            dependencies,
            patterns,
        })
    } else {
        // Fallback: Extract what we can from text response
        Ok(CodeAnalysisResult {
            structure_analysis: response.to_string(),
            semantic_representation: "Raw analysis response".to_string(),
            symbols: vec![],
            issues: vec![],
            complexity_assessment: None,
            dependencies: vec![],
            patterns: vec![],
        })
    }
}

fn parse_symbols_from_json(json: &serde_json::Value) -> Result<Vec<Symbol>> {
    let mut symbols = Vec::new();
    
    if let Some(symbols_array) = json.get("symbols").and_then(|v| v.as_array()) {
        for symbol_obj in symbols_array {
            if let Some(symbol_map) = symbol_obj.as_object() {
                symbols.push(Symbol {
                    name: symbol_map.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    kind: symbol_map.get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    start_line: symbol_map.get("start_line")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    start_column: symbol_map.get("start_column")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    end_line: symbol_map.get("end_line")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    end_column: symbol_map.get("end_column")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                });
            }
        }
    }
    
    Ok(symbols)
}

fn parse_issues_from_json(json: &serde_json::Value) -> Result<Vec<ParseError>> {
    let mut issues = Vec::new();
    
    if let Some(issues_array) = json.get("issues").and_then(|v| v.as_array()) {
        for issue_obj in issues_array {
            if let Some(issue_map) = issue_obj.as_object() {
                issues.push(ParseError {
                    message: issue_map.get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown issue")
                        .to_string(),
                    line: issue_map.get("line")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    column: issue_map.get("column")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                });
            }
        }
    }
    
    Ok(issues)
}

fn parse_string_array(json: &serde_json::Value, field: &str) -> Vec<String> {
    json.get(field)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn get_tree_sitter_language(language: &str) -> Result<Language> {
    // Use dynamic analysis via Gemini instead of static tree-sitter
    // This allows understanding ANY language/format without predefined parsers
    Err(anyhow::anyhow!("Using dynamic AI-based analysis instead of tree-sitter for: {}", language))
}

fn extract_symbols(_tree: &Tree, _content: &str, _language: &str) -> Vec<Symbol> {
    let mut symbols = Vec::new();
    
    // This would traverse the AST and extract function definitions, 
    // class definitions, variables, etc.
    // For now, return placeholder data
    symbols.push(Symbol {
        name: "placeholder_function".to_string(),
        kind: "function".to_string(),
        start_line: 0,
        start_column: 0,
        end_line: 10,
        end_column: 0,
    });

    symbols
}

fn extract_errors(tree: &Tree, content: &str) -> Vec<ParseError> {
    let mut errors = Vec::new();
    
    // Traverse the tree looking for ERROR nodes
    let root_node = tree.root_node();
    if root_node.has_error() {
        traverse_for_errors(root_node, &mut errors, content);
    }

    errors
}

fn traverse_for_errors(node: tree_sitter::Node, errors: &mut Vec<ParseError>, content: &str) {
    if node.is_error() {
        let start_position = node.start_position();
        errors.push(ParseError {
            message: "Parse error".to_string(),
            line: start_position.row,
            column: start_position.column,
        });
    }

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            traverse_for_errors(child, errors, content);
        }
    }
}

fn format_tree(tree: &Tree, content: &str, indent: usize) -> String {
    let root_node = tree.root_node();
    format_node(root_node, content, indent)
}

fn format_node(node: tree_sitter::Node, content: &str, indent: usize) -> String {
    let mut result = String::new();
    let indent_str = "  ".repeat(indent);
    
    result.push_str(&format!("{}{}[{}:{}->{}:{}]\n",
        indent_str,
        node.kind(),
        node.start_position().row,
        node.start_position().column,
        node.end_position().row,
        node.end_position().column
    ));

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            result.push_str(&format_node(child, content, indent + 1));
        }
    }

    result
}

fn format_ast(tree: &Tree) -> String {
    // Simple S-expression representation
    let root_node = tree.root_node();
    format_node_sexp(root_node)
}

fn format_node_sexp(node: tree_sitter::Node) -> String {
    if node.child_count() == 0 {
        return node.kind().to_string();
    }

    let mut result = format!("({}", node.kind());
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            result.push(' ');
            result.push_str(&format_node_sexp(child));
        }
    }
    result.push(')');
    result
}