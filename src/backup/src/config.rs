use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use crate::cli::colors::ColorScheme;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub gemini: GeminiConfig,
    pub editor: EditorConfig,
    pub cache: CacheConfig,
    pub lsp: LspConfig,
    pub workspace: WorkspaceConfig,
    pub colors: ColorScheme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    pub api_key: Option<String>,
    pub default_model: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorConfig {
    pub auto_save: bool,
    pub backup_enabled: bool,
    pub tab_size: usize,
    pub max_line_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub max_size_mb: usize,
    pub ttl_minutes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspConfig {
    pub enabled: bool,
    pub port: u16,
    pub auto_start: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    pub last_working_directory: Option<PathBuf>,
    pub remember_last_directory: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gemini: GeminiConfig {
                api_key: None,
                default_model: "gemini-2.5-flash".to_string(),
                timeout_seconds: 30,
                max_retries: 3,
            },
            editor: EditorConfig {
                auto_save: false,
                backup_enabled: true,
                tab_size: 4,
                max_line_length: 100,
            },
            cache: CacheConfig {
                enabled: true,
                max_size_mb: 100,
                ttl_minutes: 60,
            },
            lsp: LspConfig {
                enabled: false,
                port: 9257,
                auto_start: false,
            },
            workspace: WorkspaceConfig {
                last_working_directory: None,
                remember_last_directory: true,
            },
            colors: ColorScheme::default(),
        }
    }
}

impl Config {
    pub fn update_last_working_directory(&mut self, dir: PathBuf) {
        if self.workspace.remember_last_directory {
            self.workspace.last_working_directory = Some(dir);
        }
    }

    pub async fn get_intelligent_directory(&self) -> Option<PathBuf> {
        // First try last working directory if it exists and we have access
        if let Some(last_dir) = &self.workspace.last_working_directory {
            if last_dir.exists() {
                if let Ok(has_access) = crate::permissions::verify_path_access(last_dir).await {
                    if has_access {
                        return Some(last_dir.clone());
                    }
                }
            }
        }

        // Try to get first allowed directory from permissions
        if let Ok(manager) = crate::permissions::PermissionManager::new().await {
            let allowed_paths = manager.list_allowed_paths();
            for path in allowed_paths {
                if path.exists() {
                    return Some(path.clone());
                }
            }
        }

        // Fallback to current directory (will prompt for permission if needed)
        std::env::current_dir().ok()
    }
}

pub async fn load_config(config_path: Option<&Path>) -> Result<Config> {
    let config_file = match config_path {
        Some(path) => path.to_path_buf(),
        None => get_default_config_path()?,
    };

    // Load from .env file first for API keys
    let mut config = if config_file.exists() {
        let content = fs::read_to_string(&config_file)
            .await
            .with_context(|| format!("Failed to read config file: {}", config_file.display()))?;
        
        toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", config_file.display()))?
    } else {
        Config::default()
    };

    // Try to load API key from .env file in the glimmer root directory
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let env_path = exe_dir.join(".env");
            if env_path.exists() {
                if let Ok(env_content) = fs::read_to_string(&env_path).await {
                    for line in env_content.lines() {
                        if let Some(key_value) = line.strip_prefix("GEMINI_API_KEY=") {
                            config.gemini.api_key = Some(key_value.trim().to_string());
                            break;
                        }
                    }
                }
            }
        }
    }

    // Fallback to environment variable
    if config.gemini.api_key.is_none() {
        config.gemini.api_key = std::env::var("GEMINI_API_KEY").ok();
    }

    Ok(config)
}

fn get_default_config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir()
        .context("Failed to get user config directory")?;
    
    Ok(config_dir.join("glimmer").join("config.toml"))
}

pub async fn save_config(config: &Config, config_path: Option<&Path>) -> Result<()> {
    let config_file = match config_path {
        Some(path) => path.to_path_buf(),
        None => get_default_config_path()?,
    };

    // Ensure parent directory exists
    if let Some(parent) = config_file.parent() {
        fs::create_dir_all(parent).await
            .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
    }

    let content = toml::to_string_pretty(config)
        .context("Failed to serialize configuration")?;

    fs::write(&config_file, content).await
        .with_context(|| format!("Failed to write config file: {}", config_file.display()))?;

    Ok(())
}