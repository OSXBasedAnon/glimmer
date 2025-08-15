use anyhow::{Context, Result};
use std::path::PathBuf;
use crate::file_io;
use crate::parser;
use crate::{emerald_println};

pub async fn handle_parse(
    file: PathBuf,
    lang: Option<String>,
    format: String,
) -> Result<()> {
    emerald_println!("Parsing file: {}", file.display());

    // Read the file content
    let content = file_io::read_file(&file).await
        .with_context(|| format!("Failed to read file: {}", file.display()))?;

    // Detect language if not provided
    let language = lang.unwrap_or_else(|| {
        file_io::detect_language(&file)
    });

    emerald_println!("Language: {}", language);

    // Parse the code
    let parse_result = parser::parse_code(&content, &language, &format).await
        .context("Failed to parse code")?;

    // Output the result
    match format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&parse_result)?),
        "tree" => println!("{}", parse_result.tree_representation),
        "ast" => println!("{}", parse_result.ast_representation),
        _ => unreachable!(), // clap validates this
    }

    Ok(())
}