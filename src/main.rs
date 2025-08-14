use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::fmt::format::FmtSpan;

mod cli;
mod config;
mod file_io;
mod parser;
mod gemini;
mod differ;
mod lsp;
mod permissions;
mod research;
mod function_calling;
mod code_display;
mod progress_display;
mod thinking_display;
mod reasoning_engine;
mod input_handler;

use cli::colors::*;

#[derive(Parser)]
#[command(
    name = "glimmer",
    version = "0.1.0",
    about = "A blazingly fast, local Claude-style AI assistant",
    long_about = "Glimmer is a modular AI assistant that runs locally and integrates with the Gemini protocol.\nBuilt for speed, elegance, and developer productivity.\n\nRun without arguments to start interactive chat mode.",
    help_template = r#"{before-help}{name} {version}
{about}

{usage-heading} {usage}

{all-args}{after-help}
"#,
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Configuration file path
    #[arg(short, long, global = true, value_name = "FILE")]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Edit files with AI assistance
    Edit {
        /// File to edit
        #[arg(short, long, value_name = "FILE")]
        file: PathBuf,

        /// Programming language (auto-detected if not specified)
        #[arg(short, long)]
        lang: Option<String>,

        /// Query/instruction for the AI
        #[arg(short, long)]
        query: String,

        /// Output file (overwrites input if not specified)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },

    /// Parse and analyze code structure
    Parse {
        /// File to parse
        #[arg(short, long, value_name = "FILE")]
        file: PathBuf,

        /// Programming language (auto-detected if not specified)
        #[arg(short, long)]
        lang: Option<String>,

        /// Output format
        #[arg(long, default_value = "tree", value_parser = ["tree", "json", "ast"])]
        format: String,
    },

    /// Interact with Gemini protocol
    Gemini {
        /// Gemini URL to fetch
        #[arg(value_name = "URL")]
        url: String,

        /// Save response to file
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },

    /// Show differences between files or versions
    Diff {
        /// Original file
        #[arg(value_name = "FILE1")]
        file1: PathBuf,

        /// Modified file
        #[arg(value_name = "FILE2")]
        file2: PathBuf,

        /// Output format
        #[arg(long, default_value = "unified", value_parser = ["unified", "split", "json"])]
        format: String,
    },

    /// Watch files for changes and auto-process
    Watch {
        /// Directory or file to watch
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// File pattern to watch (e.g., "*.rs", "*.py")
        #[arg(short, long, default_value = "*")]
        pattern: String,

        /// Command to run when files change
        #[arg(short, long)]
        command: Option<String>,
    },

    /// Start Language Server Protocol (LSP) server
    Lsp {
        /// Port to bind LSP server
        #[arg(short, long, default_value = "9257")]
        port: u16,
    },

    /// Manage folder permissions
    Permissions {
        #[command(subcommand)]
        action: PermissionAction,
    },

    /// Interactive chat mode
    Chat {
        /// Save conversation to file
        #[arg(short, long, value_name = "FILE")]
        save: Option<PathBuf>,

        /// Load previous conversation
        #[arg(short, long, value_name = "FILE")]
        load: Option<PathBuf>,
    },

    /// Export conversation or project data
    Export {
        /// Type of export
        #[arg(value_parser = ["conversation", "config", "permissions"])]
        export_type: String,

        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,

        /// Export format
        #[arg(long, default_value = "json", value_parser = ["json", "markdown", "txt"])]
        format: String,
    },

    /// Compact output mode for scripts and automation
    Compact {
        #[command(subcommand)]
        action: CompactAction,
    },
}

#[derive(Subcommand)]
enum PermissionAction {
    /// List all allowed paths
    List,
    /// Add a path to allowed list
    Add {
        /// Path to allow
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },
    /// Remove a path from allowed list
    Remove {
        /// Path to remove
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },
    /// Clear all allowed paths
    Clear,
}

#[derive(Subcommand)]
enum CompactAction {
    /// Edit file with minimal output
    Edit {
        /// File to edit
        #[arg(short, long, value_name = "FILE")]
        file: PathBuf,
        /// Query/instruction
        #[arg(short, long)]
        query: String,
    },
    /// Parse file and output JSON only
    Parse {
        /// File to parse
        #[arg(short, long, value_name = "FILE")]
        file: PathBuf,
    },
    /// Check file permissions (returns exit code only)
    Check {
        /// File to check
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Remove Ctrl+C handler - let it work normally for copying

    let cli = Cli::parse();

    // Initialize logging with emerald green theme
    init_logging(cli.verbose)?;

    // Print startup banner
    print_banner();

    // Load configuration
    let config = config::load_config(cli.config.as_deref()).await?;
    info!("Configuration loaded successfully");

    // Initialize modern realtime UI for Claude Code-style output
    progress_display::init_realtime_ui();

    // Execute command or start interactive chat if no command provided
    match cli.command {
        Some(Commands::Edit { file, lang, query, output }) => {
            info!("Starting edit command for file: {}", file.display());
            cli::edit::handle_edit(file, lang, query, output, &config).await?;
        }
        Some(Commands::Parse { file, lang, format }) => {
            info!("Starting parse command for file: {}", file.display());
            cli::parse::handle_parse(file, lang, format).await?;
        }
        Some(Commands::Gemini { url, output }) => {
            info!("Starting Gemini fetch for URL: {}", url);
            cli::gemini::handle_gemini(url, output, &config).await?;
        }
        Some(Commands::Diff { file1, file2, format }) => {
            info!("Starting diff between {} and {}", file1.display(), file2.display());
            cli::diff::handle_diff(file1, file2, format).await?;
        }
        Some(Commands::Watch { path, pattern, command }) => {
            info!("Starting watch on path: {} with pattern: {}", path.display(), pattern);
            cli::watch::handle_watch(path, pattern, command).await?;
        }
        Some(Commands::Lsp { port }) => {
            info!("Starting LSP server on port: {}", port);
            cli::lsp::handle_lsp(port).await?;
        }
        Some(Commands::Permissions { action }) => {
            cli::permissions::handle_permissions(action).await?;
        }
        Some(Commands::Chat { save, load }) => {
            cli::chat::handle_chat(save, load, &config).await?;
        }
        Some(Commands::Export { export_type, output, format }) => {
            cli::export::handle_export(export_type, output, format).await?;
        }
        Some(Commands::Compact { action }) => {
            cli::compact::handle_compact(action, &config).await?;
        }
        None => {
            // No subcommand provided - start interactive chat mode
            info!("Starting interactive chat mode");
            let save_path = Some(std::env::current_dir()?.join("glimmer-conversation.json"));
            cli::chat::handle_chat(save_path, None, &config).await?;
        }
    }

    Ok(())
}

fn init_logging(verbose: bool) -> Result<()> {
    let level = if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .with_thread_ids(true)
        .with_ansi(true)
        .init();

    Ok(())
}

fn print_banner() {
    println!("{}", EMERALD_BOLD);
    println!("    {}GLIMMER{}", EMERALD_BRIGHT, RESET);
    println!("    {bright}A blazingly fast AI assistant{reset}", 
        bright = EMERALD_BRIGHT, reset = RESET);
    println!("    {dim}Built with Rust{reset}", 
        dim = EMERALD_DIM, reset = RESET);
    println!();
}