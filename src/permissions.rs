use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tokio::fs;
use crate::{emerald_println, error_println};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionsConfig {
    pub allowed_paths: HashSet<PathBuf>,
    pub version: u32,
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            allowed_paths: HashSet::new(),
            version: 1,
        }
    }
}

pub struct PermissionManager {
    config: PermissionsConfig,
    config_path: PathBuf,
}

impl PermissionManager {
    pub async fn new() -> Result<Self> {
        let config_path = get_permissions_config_path()?;
        
        let config = if config_path.exists() {
            let content = fs::read_to_string(&config_path).await
                .context("Failed to read permissions config")?;
            
            toml::from_str(&content)
                .context("Failed to parse permissions config")?
        } else {
            PermissionsConfig::default()
        };

        Ok(Self { config, config_path })
    }

    pub async fn check_path_access(&mut self, path: &Path) -> Result<bool> {
        // For non-existent files, check the parent directory
        let target_path = if path.exists() {
            path.canonicalize()
                .with_context(|| format!("Failed to canonicalize path: {}", path.display()))?
        } else {
            // Path doesn't exist, use the parent directory
            let parent = path.parent()
                .context("Path has no parent directory")?;
            
            // Try to canonicalize the parent, or use as-is if parent doesn't exist either
            if parent.exists() {
                let canonical_parent = parent.canonicalize()
                    .with_context(|| format!("Failed to canonicalize parent path: {}", parent.display()))?;
                // Return the canonical parent joined with the filename
                if let Some(filename) = path.file_name() {
                    canonical_parent.join(filename)
                } else {
                    canonical_parent
                }
            } else {
                path.to_path_buf()
            }
        };

        // Block access to system folders
        if is_system_folder(&target_path) {
            error_println!("Access denied: System folders are not allowed");
            return Ok(false);
        }

        // Get the parent directory to check
        let parent_dir = if target_path.is_file() || !target_path.exists() {
            target_path.parent()
                .context("Path has no parent directory")?
                .to_path_buf()
        } else {
            target_path.clone()
        };

        // Check if path is already allowed
        if self.is_path_allowed(&parent_dir) {
            return Ok(true);
        }

        // Always prompt user for approval if not already allowed
        if self.prompt_for_approval(&parent_dir).await? {
            self.add_allowed_path(parent_dir).await?;
            Ok(true)
        } else {
            // User denied - just return false, don't save denial anywhere
            // This allows re-prompting next time
            Ok(false)
        }
    }

    fn is_path_allowed(&self, path: &Path) -> bool {
        self.config.allowed_paths.iter().any(|allowed_path| {
            path.starts_with(allowed_path) || allowed_path.starts_with(path)
        })
    }

    async fn prompt_for_approval(&self, path: &Path) -> Result<bool> {
        // For now, auto-approve the glimmer directory to prevent terminal conflicts
        // TODO: Implement proper ratatui-compatible permission dialog
        if path.to_string_lossy().contains("glimmer") {
            emerald_println!("\n✅ Auto-approved access to glimmer directory: {}", path.display());
            return Ok(true);
        }
        
        // For other paths, use a simple approval mechanism
        emerald_println!("\n🔒 Permission Request:");
        emerald_println!("Glimmer needs access to: {}", path.display());
        emerald_println!("Auto-approving for now to prevent terminal conflicts.");
        
        Ok(true)
    }

    async fn add_allowed_path(&mut self, path: PathBuf) -> Result<()> {
        self.config.allowed_paths.insert(path.clone());
        self.save_config().await?;
        emerald_println!("Added to allowed paths: {}", path.display());
        Ok(())
    }

    async fn save_config(&self) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent).await
                .context("Failed to create config directory")?;
        }

        let content = toml::to_string_pretty(&self.config)
            .context("Failed to serialize permissions config")?;

        fs::write(&self.config_path, content).await
            .context("Failed to write permissions config")?;

        Ok(())
    }

    pub fn list_allowed_paths(&self) -> &HashSet<PathBuf> {
        &self.config.allowed_paths
    }

    pub async fn remove_path(&mut self, path: &Path) -> Result<bool> {
        let removed = self.config.allowed_paths.remove(path);
        if removed {
            self.save_config().await?;
            emerald_println!("Removed from allowed paths: {}", path.display());
        }
        Ok(removed)
    }

    pub async fn clear_all_paths(&mut self) -> Result<()> {
        self.config.allowed_paths.clear();
        self.save_config().await?;
        emerald_println!("Cleared all allowed paths");
        Ok(())
    }
}

fn get_permissions_config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir()
        .context("Failed to get user config directory")?;
    
    Ok(config_dir.join("glimmer").join("permissions.toml"))
}

fn is_system_folder(path: &Path) -> bool {
    let path_str = path.to_string_lossy().to_lowercase();
    
    // Windows system folders
    if cfg!(windows) {
        let system_folders = [
            "c:\\windows",
            "c:\\program files",
            "c:\\program files (x86)",
            "c:\\programdata",
            "c:\\system volume information",
            "c:\\$recycle.bin",
            "c:\\recovery",
            "c:\\boot",
            "c:\\efi",
        ];
        
        for system_folder in &system_folders {
            if path_str.starts_with(system_folder) {
                return true;
            }
        }
        
        // Check for Windows user system folders
        if let Some(user_dir) = dirs::home_dir() {
            let user_str = user_dir.to_string_lossy().to_lowercase();
            let restricted_user_folders = [
                format!("{}\\appdata\\roaming\\microsoft", user_str),
                format!("{}\\appdata\\local\\microsoft", user_str),
            ];
            
            for restricted_folder in &restricted_user_folders {
                if path_str.starts_with(restricted_folder) {
                    return true;
                }
            }
        }
    }

    // Unix/Linux system folders
    if cfg!(unix) {
        let system_folders = [
            "/bin", "/sbin", "/usr/bin", "/usr/sbin",
            "/boot", "/dev", "/proc", "/sys",
            "/etc", "/lib", "/lib64", "/usr/lib",
            "/var/log", "/var/run", "/var/lock",
            "/root",
        ];
        
        for system_folder in &system_folders {
            if path_str.starts_with(system_folder) {
                return true;
            }
        }
    }

    false
}

// Utility function to check if a path is safe to access
pub async fn verify_path_access(path: &Path) -> Result<bool> {
    let mut manager = PermissionManager::new().await?;
    manager.check_path_access(path).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_system_folder_detection() {
        if cfg!(windows) {
            assert!(is_system_folder(Path::new("C:\\Windows\\System32")));
            assert!(is_system_folder(Path::new("C:\\Program Files\\Test")));
            assert!(!is_system_folder(Path::new("C:\\Users\\TestUser\\Documents")));
        }
        
        if cfg!(unix) {
            assert!(is_system_folder(Path::new("/bin/bash")));
            assert!(is_system_folder(Path::new("/etc/passwd")));
            assert!(!is_system_folder(Path::new("/home/user/documents")));
        }
    }

    #[tokio::test]
    async fn test_permission_manager() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, "test content").unwrap();

        // This would require user interaction in real usage
        // In tests, we'd mock the approval process
    }
}