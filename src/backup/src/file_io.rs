use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::fs;
use ropey::Rope;
use memmap2::MmapOptions;
use std::fs::File;
use crossterm::{event::{self, Event, KeyCode}, terminal::{disable_raw_mode, enable_raw_mode}};
use crate::permissions;

pub async fn read_file(path: &Path) -> Result<String> {
    // Check permissions first
    if !permissions::verify_path_access(path).await? {
        return Err(anyhow::anyhow!("Access denied to path: {}", path.display()));
    }

    // For large files, use memory mapping
    if let Ok(metadata) = fs::metadata(path).await {
        if metadata.len() > 10_000_000 { // 10MB threshold
            return read_large_file(path).await;
        }
    }

    // For regular files, use tokio fs
    fs::read_to_string(path).await
        .with_context(|| format!("Failed to read file: {}", path.display()))
}

async fn read_large_file(path: &Path) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;
    
    let mmap = unsafe {
        MmapOptions::new().map(&file)
            .with_context(|| format!("Failed to memory map file: {}", path.display()))?
    };

    std::str::from_utf8(&mmap)
        .with_context(|| format!("File contains invalid UTF-8: {}", path.display()))
        .map(|s| s.to_string())
}

pub async fn write_file(path: &Path, content: &str) -> Result<()> {
    // Check permissions first
    if !permissions::verify_path_access(path).await? {
        return Err(anyhow::anyhow!("Access denied to path: {}", path.display()));
    }

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    fs::write(path, content).await
        .with_context(|| format!("Failed to write file: {}", path.display()))
}

pub fn detect_language(path: &Path) -> String {
    if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
        match extension.to_lowercase().as_str() {
            "rs" => "rust".to_string(),
            "py" => "python".to_string(),
            "js" | "mjs" => "javascript".to_string(),
            "ts" => "typescript".to_string(),
            "go" => "go".to_string(),
            "java" => "java".to_string(),
            "cpp" | "cc" | "cxx" => "cpp".to_string(),
            "c" => "c".to_string(),
            "cs" => "csharp".to_string(),
            "php" => "php".to_string(),
            "rb" => "ruby".to_string(),
            "swift" => "swift".to_string(),
            "kt" => "kotlin".to_string(),
            "scala" => "scala".to_string(),
            "sh" | "bash" => "bash".to_string(),
            "ps1" => "powershell".to_string(),
            "html" | "htm" => "html".to_string(),
            "css" => "css".to_string(),
            "scss" | "sass" => "scss".to_string(),
            "json" => "json".to_string(),
            "xml" => "xml".to_string(),
            "yaml" | "yml" => "yaml".to_string(),
            "toml" => "toml".to_string(),
            "md" => "markdown".to_string(),
            "sql" => "sql".to_string(),
            _ => "text".to_string(),
        }
    } else {
        "text".to_string()
    }
}

pub struct RopeEditor {
    rope: Rope,
    path: PathBuf,
}

impl RopeEditor {
    pub async fn new(path: PathBuf) -> Result<Self> {
        // Check permissions first
        if !permissions::verify_path_access(&path).await? {
            return Err(anyhow::anyhow!("Access denied to path: {}", path.display()));
        }

        let content = if path.exists() {
            // Use fs::read_to_string directly here since we already checked permissions
            fs::read_to_string(&path).await
                .with_context(|| format!("Failed to read file: {}", path.display()))?
        } else {
            String::new()
        };

        let rope = Rope::from_str(&content);
        
        Ok(Self { rope, path })
    }

    pub fn insert(&mut self, char_idx: usize, text: &str) {
        self.rope.insert(char_idx, text);
    }

    pub fn remove(&mut self, start_idx: usize, end_idx: usize) {
        self.rope.remove(start_idx..end_idx);
    }

    pub fn replace(&mut self, start_idx: usize, end_idx: usize, text: &str) {
        self.rope.remove(start_idx..end_idx);
        self.rope.insert(start_idx, text);
    }

    pub fn get_line(&self, line_idx: usize) -> Option<String> {
        if line_idx < self.rope.len_lines() {
            Some(self.rope.line(line_idx).to_string())
        } else {
            None
        }
    }

    pub fn to_string(&self) -> String {
        self.rope.to_string()
    }

    pub async fn save(&self) -> Result<()> {
        write_file(&self.path, &self.to_string()).await
    }
}

// Safety confirmation system for destructive operations
pub async fn delete_file_with_confirmation(path: &Path) -> Result<bool> {
    use crate::cli::colors::{BLUE_BRIGHT, GRAY_DIM, RED_ERROR, EMERALD_BRIGHT, RESET};
    
    if !path.exists() {
        println!("{}File does not exist: {}{}", RED_ERROR, path.display(), RESET);
        return Ok(false);
    }
    
    // Show file info
    let metadata = fs::metadata(path).await?;
    let size = metadata.len();
    let size_str = if size < 1024 {
        format!("{} bytes", size)
    } else if size < 1024 * 1024 {
        format!("{:.1} KB", size as f64 / 1024.0)
    } else {
        format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
    };
    
    println!("\n{}⚠️  Destructive Operation Warning{}", RED_ERROR, RESET);
    println!("{}File: {}{}", GRAY_DIM, path.display(), RESET);
    println!("{}Size: {}{}", GRAY_DIM, size_str, RESET);
    println!();
    
    // Blue prompt as requested
    println!("{}Press [Enter] to delete the file{}", BLUE_BRIGHT, RESET);
    println!("{}Press [Esc] to cancel{}", GRAY_DIM, RESET);
    
    // Wait for user confirmation using raw mode
    enable_raw_mode().unwrap_or(());
    let mut confirmed = false;
    let mut cancelled = false;
    
    loop {
        if let Ok(Event::Key(key_event)) = event::read() {
            match key_event.code {
                KeyCode::Enter => {
                    confirmed = true;
                    break;
                }
                KeyCode::Esc => {
                    cancelled = true;
                    break;
                }
                _ => continue,
            }
        }
    }
    
    disable_raw_mode().unwrap_or(());
    println!();
    
    if cancelled {
        println!("{}❌ File deletion cancelled{}", GRAY_DIM, RESET);
        return Ok(false);
    }
    
    if confirmed {
        fs::remove_file(path).await.with_context(|| format!("Failed to delete file: {}", path.display()))?;
        println!("{}✅ File deleted: {}{}", EMERALD_BRIGHT, path.display(), RESET);
        return Ok(true);
    }
    
    Ok(false)
}

pub async fn overwrite_file_with_confirmation(path: &Path, content: &str) -> Result<bool> {
    use crate::cli::colors::{BLUE_BRIGHT, GRAY_DIM, YELLOW_WARN, EMERALD_BRIGHT, RESET};
    
    if !path.exists() {
        // File doesn't exist, safe to write
        return write_file(path, content).await.map(|_| true);
    }
    
    // File exists, ask for confirmation
    let metadata = fs::metadata(path).await?;
    let size = metadata.len();
    let size_str = if size < 1024 {
        format!("{} bytes", size)
    } else if size < 1024 * 1024 {
        format!("{:.1} KB", size as f64 / 1024.0)
    } else {
        format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
    };
    
    println!("\n{}⚠️  File Overwrite Warning{}", YELLOW_WARN, RESET);
    println!("{}File: {}{}", GRAY_DIM, path.display(), RESET);
    println!("{}Current size: {}{}", GRAY_DIM, size_str, RESET);
    println!("{}New content size: {} bytes{}", GRAY_DIM, content.len(), RESET);
    println!();
    
    // Blue prompt as requested
    println!("{}Press [Enter] to overwrite the file{}", BLUE_BRIGHT, RESET);
    println!("{}Press [Esc] to cancel{}", GRAY_DIM, RESET);
    
    // Wait for user confirmation
    enable_raw_mode().unwrap_or(());
    let mut confirmed = false;
    let mut cancelled = false;
    
    loop {
        if let Ok(Event::Key(key_event)) = event::read() {
            match key_event.code {
                KeyCode::Enter => {
                    confirmed = true;
                    break;
                }
                KeyCode::Esc => {
                    cancelled = true;
                    break;
                }
                _ => continue,
            }
        }
    }
    
    disable_raw_mode().unwrap_or(());
    println!();
    
    if cancelled {
        println!("{}❌ File overwrite cancelled{}", GRAY_DIM, RESET);
        return Ok(false);
    }
    
    if confirmed {
        write_file(path, content).await.with_context(|| format!("Failed to overwrite file: {}", path.display()))?;
        println!("{}✅ File overwritten: {}{}", EMERALD_BRIGHT, path.display(), RESET);
        return Ok(true);
    }
    
    Ok(false)
}

// Safe write - asks for confirmation if file exists
pub async fn safe_write_file(path: &Path, content: &str) -> Result<()> {
    if path.exists() {
        if !overwrite_file_with_confirmation(path, content).await? {
            return Err(anyhow::anyhow!("File write cancelled by user"));
        }
    } else {
        write_file(path, content).await?;
    }
    Ok(())
}