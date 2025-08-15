use anyhow::Result;
use crate::permissions::PermissionManager;
use crate::{emerald_println, warn_println};

pub async fn handle_permissions(action: crate::PermissionAction) -> Result<()> {
    let mut manager = PermissionManager::new().await?;

    match action {
        crate::PermissionAction::List => {
            let allowed_paths = manager.list_allowed_paths();
            
            if allowed_paths.is_empty() {
                warn_println!("No allowed paths configured");
                emerald_println!("Use 'glimmer permissions add <path>' to add a path");
            } else {
                emerald_println!("Allowed paths:");
                for path in allowed_paths {
                    emerald_println!("  {}", path.display());
                }
            }
        }
        crate::PermissionAction::Add { path } => {
            let canonical_path = path.canonicalize()?;
            let parent_dir = if canonical_path.is_file() {
                canonical_path.parent().unwrap().to_path_buf()
            } else {
                canonical_path
            };

            if manager.check_path_access(&parent_dir).await? {
                emerald_println!("Path access granted and saved: {}", parent_dir.display());
            } else {
                warn_println!("Path access denied: {}", parent_dir.display());
            }
        }
        crate::PermissionAction::Remove { path } => {
            let canonical_path = path.canonicalize()?;
            let parent_dir = if canonical_path.is_file() {
                canonical_path.parent().unwrap().to_path_buf()
            } else {
                canonical_path
            };

            if manager.remove_path(&parent_dir).await? {
                emerald_println!("Removed path: {}", parent_dir.display());
            } else {
                warn_println!("Path not found in allowed list: {}", parent_dir.display());
            }
        }
        crate::PermissionAction::Clear => {
            manager.clear_all_paths().await?;
            emerald_println!("All allowed paths have been cleared");
        }
    }

    Ok(())
}