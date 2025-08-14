// Emerald green color theme for terminal output
use serde::{Deserialize, Serialize};

pub const RESET: &str = "\x1b[0m";

// Emerald green shades - Rich professional emerald palette
pub const EMERALD_BRIGHT: &str = "\x1b[38;2;16;185;129m";    // RGB(16, 185, 129) - Emerald 500
pub const EMERALD_BOLD: &str = "\x1b[1;38;2;5;150;105m";     // RGB(5, 150, 105) - Emerald 600 Bold
pub const EMERALD_DIM: &str = "\x1b[2;38;2;52;211;153m";     // RGB(52, 211, 153) - Emerald 400 Dim  
pub const EMERALD_UNDERLINE: &str = "\x1b[4;38;2;16;185;129m"; // Emerald 500 Underlined
pub const EMERALD_LIGHT: &str = "\x1b[38;2;110;231;183m";    // RGB(110, 231, 183) - Emerald 300
pub const EMERALD_DARK: &str = "\x1b[38;2;4;120;87m";        // RGB(4, 120, 87) - Emerald 700
pub const EMERALD_VERY_LIGHT: &str = "\x1b[38;2;167;243;208m"; // RGB(167, 243, 208) - Emerald 200

// Additional colors for UI elements
pub const WHITE_BRIGHT: &str = "\x1b[97m";      // Bright white
pub const GRAY_DIM: &str = "\x1b[90m";          // Dim gray
pub const RED_ERROR: &str = "\x1b[91m";         // Bright red for errors
pub const GREEN_SUCCESS: &str = "\x1b[32m";     // Green for success/additions
pub const YELLOW_WARN: &str = "\x1b[93m";       // Bright yellow for warnings
pub const ORANGE_EDIT: &str = "\x1b[38;2;255;175;0m";      // #ffaf00 for editing
pub const YELLOW_ANALYZE: &str = "\x1b[38;2;255;204;92m";   // #ffcc5c for analyzing
pub const GREEN_COMPLETE: &str = "\x1b[38;2;159;239;0m";    // #9fef00 for task completion
pub const PURPLE_BOLD: &str = "\x1b[1;95m";     // Bold bright magenta/purple
pub const PURPLE_BRIGHT: &str = "\x1b[95m";     // Bright magenta/purple

// Blue shades for modern UI elements
pub const BLUE_BRIGHT: &str = "\x1b[38;2;59;130;246m";      // RGB(59, 130, 246) - Blue 500
pub const BLUE_LIGHT: &str = "\x1b[38;2;147;197;253m";      // RGB(147, 197, 253) - Blue 300
pub const BLUE_LIGHTER: &str = "\x1b[38;2;191;219;254m";    // RGB(191, 219, 254) - Blue 200
pub const BLUE_DARK: &str = "\x1b[38;2;37;99;235m";         // RGB(37, 99, 235) - Blue 600
pub const BLUE_DIM: &str = "\x1b[2;38;2;59;130;246m";       // Blue 500 dimmed

// Enhanced grey shades for better hierarchy
pub const GRAY_BRIGHT: &str = "\x1b[38;2;156;163;175m";     // RGB(156, 163, 175) - Gray 400
pub const GRAY_MEDIUM: &str = "\x1b[38;2;107;114;128m";     // RGB(107, 114, 128) - Gray 500
pub const GRAY_DARK: &str = "\x1b[38;2;75;85;99m";          // RGB(75, 85, 99) - Gray 600
pub const GRAY_DARKER: &str = "\x1b[38;2;55;65;81m";        // RGB(55, 65, 81) - Gray 700

// Utility macros for colored output
#[macro_export]
macro_rules! emerald_print {
    ($($arg:tt)*) => {
        print!("{}{}{}", $crate::cli::colors::EMERALD_BRIGHT, format_args!($($arg)*), $crate::cli::colors::RESET)
    };
}

#[macro_export]
macro_rules! emerald_println {
    ($($arg:tt)*) => {
        println!("{}{}{}", $crate::cli::colors::EMERALD_BRIGHT, format_args!($($arg)*), $crate::cli::colors::RESET)
    };
}

#[macro_export]
macro_rules! error_println {
    ($($arg:tt)*) => {
        println!("{}{}{}", $crate::cli::colors::RED_ERROR, format_args!($($arg)*), $crate::cli::colors::RESET)
    };
}

#[macro_export]
macro_rules! warn_println {
    ($($arg:tt)*) => {
        println!("{}{}{}", $crate::cli::colors::YELLOW_WARN, format_args!($($arg)*), $crate::cli::colors::RESET)
    };
}

pub fn format_success(text: &str) -> String {
    format!("{}{}{}", EMERALD_BRIGHT, text, RESET)
}

pub fn format_error(text: &str) -> String {
    format!("{}{}{}", RED_ERROR, text, RESET)
}

pub fn format_warning(text: &str) -> String {
    format!("{}{}{}", YELLOW_WARN, text, RESET)
}

pub fn format_dim(text: &str) -> String {
    format!("{}{}{}", GRAY_DIM, text, RESET)
}

// Emerald-specific formatters
pub fn format_emerald(text: &str) -> String {
    format!("{}{}{}", EMERALD_BRIGHT, text, RESET)
}

pub fn format_emerald_bold(text: &str) -> String {
    format!("{}{}{}", EMERALD_BOLD, text, RESET)
}

pub fn format_emerald_light(text: &str) -> String {
    format!("{}{}{}", EMERALD_LIGHT, text, RESET)
}

pub fn format_emerald_dark(text: &str) -> String {
    format!("{}{}{}", EMERALD_DARK, text, RESET)
}

// Blue formatters for modern interactive prompts
pub fn format_blue(text: &str) -> String {
    format!("{}{}{}", BLUE_BRIGHT, text, RESET)
}

pub fn format_blue_light(text: &str) -> String {
    format!("{}{}{}", BLUE_LIGHT, text, RESET)
}

pub fn format_blue_lighter(text: &str) -> String {
    format!("{}{}{}", BLUE_LIGHTER, text, RESET)
}

pub fn format_blue_dark(text: &str) -> String {
    format!("{}{}{}", BLUE_DARK, text, RESET)
}

// Grey formatters for better text hierarchy
pub fn format_gray_bright(text: &str) -> String {
    format!("{}{}{}", GRAY_BRIGHT, text, RESET)
}

pub fn format_gray_medium(text: &str) -> String {
    format!("{}{}{}", GRAY_MEDIUM, text, RESET)
}

pub fn format_gray_dark(text: &str) -> String {
    format!("{}{}{}", GRAY_DARK, text, RESET)
}

pub fn format_gray_darker(text: &str) -> String {
    format!("{}{}{}", GRAY_DARKER, text, RESET)
}

// JSON-configurable color scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub accent: String,
    pub success: String,
    pub warning: String,
    pub error: String,
    pub dim: String,
    pub bright: String,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self::emerald_scheme()
    }
}

impl ColorScheme {
    pub fn emerald_scheme() -> Self {
        Self {
            primary: EMERALD_BRIGHT.to_string(),
            secondary: EMERALD_LIGHT.to_string(),
            accent: EMERALD_DARK.to_string(),
            success: EMERALD_BOLD.to_string(),
            warning: YELLOW_WARN.to_string(),
            error: RED_ERROR.to_string(),
            dim: GRAY_DIM.to_string(),
            bright: WHITE_BRIGHT.to_string(),
        }
    }
    
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
    
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    pub fn apply_to_text(&self, text: &str, color_type: ColorType) -> String {
        let color = match color_type {
            ColorType::Primary => &self.primary,
            ColorType::Secondary => &self.secondary,
            ColorType::Accent => &self.accent,
            ColorType::Success => &self.success,
            ColorType::Warning => &self.warning,
            ColorType::Error => &self.error,
            ColorType::Dim => &self.dim,
            ColorType::Bright => &self.bright,
        };
        format!("{}{}{}", color, text, RESET)
    }
}

#[derive(Debug, Clone)]
pub enum ColorType {
    Primary,
    Secondary,
    Accent,
    Success,
    Warning,
    Error,
    Dim,
    Bright,
}

// Advanced emerald macros with different shades
#[macro_export]
macro_rules! emerald_light_print {
    ($($arg:tt)*) => {
        print!("{}{}{}", $crate::cli::colors::EMERALD_LIGHT, format_args!($($arg)*), $crate::cli::colors::RESET)
    };
}

#[macro_export]
macro_rules! emerald_dark_print {
    ($($arg:tt)*) => {
        print!("{}{}{}", $crate::cli::colors::EMERALD_DARK, format_args!($($arg)*), $crate::cli::colors::RESET)
    };
}

#[macro_export]
macro_rules! emerald_bold_println {
    ($($arg:tt)*) => {
        println!("{}{}{}", $crate::cli::colors::EMERALD_BOLD, format_args!($($arg)*), $crate::cli::colors::RESET)
    };
}

// Progress indicator with emerald theme
pub fn create_emerald_progress_bar(width: usize, progress: f32) -> String {
    let filled = (width as f32 * progress) as usize;
    let empty = width - filled;
    
    format!(
        "{}{}{}{}{}",
        EMERALD_BRIGHT,
        "█".repeat(filled),
        EMERALD_DIM,
        "░".repeat(empty),
        RESET
    )
}