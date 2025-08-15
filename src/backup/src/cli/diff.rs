use anyhow::{Context, Result};
use std::path::PathBuf;
use crate::file_io;
use crate::differ;
use crate::{emerald_println};

pub async fn handle_diff(
    file1: PathBuf,
    file2: PathBuf,
    format: String,
) -> Result<()> {
    emerald_println!("Comparing {} and {}", file1.display(), file2.display());

    // Read both files
    let content1 = file_io::read_file(&file1).await
        .with_context(|| format!("Failed to read file: {}", file1.display()))?;

    let content2 = file_io::read_file(&file2).await
        .with_context(|| format!("Failed to read file: {}", file2.display()))?;

    // Create diff based on format
    let diff_result = match format.as_str() {
        "unified" => differ::create_diff(&content1, &content2, 
            file1.to_string_lossy().as_ref(), 
            file2.to_string_lossy().as_ref())?,
        "split" => differ::create_split_diff(&content1, &content2, 
            file1.to_string_lossy().as_ref(), 
            file2.to_string_lossy().as_ref())?,
        "json" => {
            let changes = differ::get_changes(&content1, &content2)?;
            serde_json::to_string_pretty(&changes)?
        },
        _ => unreachable!(), // clap validates this
    };

    println!("{}", diff_result);
    Ok(())
}