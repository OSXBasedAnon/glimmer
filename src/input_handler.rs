use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{self},
    execute,
    style::Print,
};
use std::io::{self, Write, stdout};
use std::time::Duration;
use tokio::sync::mpsc;
use crate::cli::colors::*;

/// Enhanced input handler with bracketed paste mode and ESC interrupt support
pub struct InputHandler {
    paste_buffer: String,
    in_paste_mode: bool,
    interrupt_sender: Option<mpsc::UnboundedSender<InterruptSignal>>,
}

/// Signals for interrupting operations
#[derive(Debug, Clone)]
pub enum InterruptSignal {
    EscapePressed,
    CtrlC,
    PasteStarted,
    PasteCompleted(String),
}

impl InputHandler {
    pub fn new() -> Self {
        Self {
            paste_buffer: String::new(),
            in_paste_mode: false,
            interrupt_sender: None,
        }
    }

    /// Set interrupt sender for communicating with running processes
    pub fn set_interrupt_sender(&mut self, sender: mpsc::UnboundedSender<InterruptSignal>) {
        self.interrupt_sender = Some(sender);
    }

    /// Enable bracketed paste mode in terminal
    pub fn enable_bracketed_paste() -> Result<()> {
        // Enable bracketed paste mode
        execute!(stdout(), Print("\x1b[?2004h"))?;
        Ok(())
    }

    /// Disable bracketed paste mode in terminal
    pub fn disable_bracketed_paste() -> Result<()> {
        // Disable bracketed paste mode
        execute!(stdout(), Print("\x1b[?2004l"))?;
        Ok(())
    }

    /// Read input with enhanced paste handling and ESC interrupt support
    pub async fn read_input_enhanced(&mut self, prompt: &str) -> Result<InputResult> {
        print!("{}", prompt);
        io::stdout().flush()?;

        // Enable raw mode for better key handling
        terminal::enable_raw_mode()?;
        
        let result = self.read_input_loop().await;
        
        // Always clean up
        terminal::disable_raw_mode()?;
        
        result
    }

    /// Main input reading loop with all enhancements
    async fn read_input_loop(&mut self) -> Result<InputResult> {
        let mut input_buffer = String::new();
        let mut cursor_pos = 0;

        loop {
            // Check for events with timeout to allow for interrupts
            if event::poll(Duration::from_millis(100))? {
                match event::read()? {
                    Event::Key(key_event) => {
                        match self.handle_key_event(key_event, &mut input_buffer, &mut cursor_pos).await? {
                            Some(result) => return Ok(result),
                            None => continue,
                        }
                    }
                    Event::Paste(text) => {
                        // Handle paste events directly from crossterm
                        return Ok(InputResult::Input(self.handle_large_paste(&text)?));
                    }
                    _ => continue,
                }
            }

            // Check for external interrupt signals
            if let Some(ref _sender) = self.interrupt_sender {
                // This allows other parts of the system to signal interrupts
                // The actual interrupt handling is done in the key event handler
            }
        }
    }

    /// Handle individual key events
    async fn handle_key_event(
        &mut self,
        key_event: KeyEvent,
        input_buffer: &mut String,
        cursor_pos: &mut usize,
    ) -> Result<Option<InputResult>> {
        match key_event.code {
            // Handle Enter key
            KeyCode::Enter => {
                println!(); // Move to next line
                if input_buffer.is_empty() {
                    return Ok(Some(InputResult::Empty));
                }
                return Ok(Some(InputResult::Input(input_buffer.clone())));
            }

            // Handle ESC key - interrupt signal
            KeyCode::Esc => {
                println!("\n{}🛑 Operation interrupted by ESC key{}", EMERALD_BRIGHT, RESET);
                if let Some(ref sender) = self.interrupt_sender {
                    let _ = sender.send(InterruptSignal::EscapePressed);
                }
                return Ok(Some(InputResult::Interrupted));
            }

            // Handle Ctrl+C
            KeyCode::Char('c') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                println!("\n{}🛑 Operation interrupted by Ctrl+C{}", EMERALD_BRIGHT, RESET);
                if let Some(ref sender) = self.interrupt_sender {
                    let _ = sender.send(InterruptSignal::CtrlC);
                }
                return Ok(Some(InputResult::Interrupted));
            }

            // Handle Backspace
            KeyCode::Backspace => {
                if *cursor_pos > 0 && !input_buffer.is_empty() {
                    let char_index = self.get_char_index_at_cursor(input_buffer, *cursor_pos - 1);
                    input_buffer.remove(char_index);
                    *cursor_pos -= 1;
                    self.redraw_line(input_buffer, *cursor_pos)?;
                }
            }

            // Handle Delete
            KeyCode::Delete => {
                if *cursor_pos < input_buffer.chars().count() {
                    let char_index = self.get_char_index_at_cursor(input_buffer, *cursor_pos);
                    input_buffer.remove(char_index);
                    self.redraw_line(input_buffer, *cursor_pos)?;
                }
            }

            // Handle Left Arrow
            KeyCode::Left => {
                if *cursor_pos > 0 {
                    *cursor_pos -= 1;
                    execute!(stdout(), crossterm::cursor::MoveLeft(1))?;
                }
            }

            // Handle Right Arrow
            KeyCode::Right => {
                if *cursor_pos < input_buffer.chars().count() {
                    *cursor_pos += 1;
                    execute!(stdout(), crossterm::cursor::MoveRight(1))?;
                }
            }

            // Handle Home
            KeyCode::Home => {
                execute!(stdout(), crossterm::cursor::MoveToColumn(0))?;
                *cursor_pos = 0;
            }

            // Handle End
            KeyCode::End => {
                let line_length = input_buffer.chars().count();
                execute!(stdout(), crossterm::cursor::MoveToColumn(line_length as u16))?;
                *cursor_pos = line_length;
            }

            // Handle Ctrl+V for paste (if bracketed paste isn't working)
            KeyCode::Char('v') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                // Try to get clipboard content using crossterm's paste detection
                println!("{}📋 Paste your content and press Ctrl+D when done:{}", EMERALD_BRIGHT, RESET);
                let pasted_content = self.read_multiline_paste().await?;
                return Ok(Some(InputResult::Input(self.handle_large_paste(&pasted_content)?)));
            }

            // Handle Ctrl+D to finish multiline paste
            KeyCode::Char('d') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                if !self.paste_buffer.is_empty() {
                    let content = self.paste_buffer.clone();
                    self.paste_buffer.clear();
                    return Ok(Some(InputResult::Input(self.handle_large_paste(&content)?)));
                }
            }

            // Handle regular character input
            KeyCode::Char(c) => {
                let char_index = self.get_char_index_at_cursor(input_buffer, *cursor_pos);
                input_buffer.insert(char_index, c);
                *cursor_pos += 1;

                // Always redraw the line to show the new character and keep cursor position correct
                self.redraw_line(input_buffer, *cursor_pos)?;
            }

            _ => {} // Ignore other keys
        }

        Ok(None)
    }

    /// Read multiline paste content
    async fn read_multiline_paste(&mut self) -> Result<String> {
        let mut paste_content = String::new();
        
        loop {
            if event::poll(Duration::from_millis(100))? {
                match event::read()? {
                    Event::Key(KeyEvent { code: KeyCode::Char('d'), modifiers, .. }) 
                        if modifiers.contains(KeyModifiers::CONTROL) => {
                        break;
                    }
                    Event::Key(KeyEvent { code: KeyCode::Char(c), .. }) => {
                        paste_content.push(c);
                        print!("{}", c);
                        io::stdout().flush()?;
                    }
                    Event::Key(KeyEvent { code: KeyCode::Enter, .. }) => {
                        paste_content.push('\n');
                        println!();
                    }
                    _ => continue,
                }
            }
        }

        Ok(paste_content)
    }

    /// Handle large paste content with preprocessing
    fn handle_large_paste(&self, content: &str) -> Result<String> {
        let lines: Vec<&str> = content.lines().collect();
        
        if lines.len() > 10 || content.len() > 1000 {
            println!("\n{}📋 Large paste detected ({} lines, {} characters){}", 
                    EMERALD_BRIGHT, lines.len(), content.len(), RESET);
            
            // Show preview of first few lines
            println!("{}Preview:{}", BLUE_BRIGHT, RESET);
            for (i, line) in lines.iter().take(3).enumerate() {
                println!("{}{}: {}{}", GRAY_DIM, i + 1, line, RESET);
            }
            
            if lines.len() > 3 {
                println!("{}... ({} more lines){}", GRAY_DIM, lines.len() - 3, RESET);
            }
            
            // Ask for confirmation
            print!("\n{}Process this large paste? (y/N): {}", EMERALD_BRIGHT, RESET);
            io::stdout().flush()?;
            
            // Read confirmation in raw mode
            loop {
                if event::poll(Duration::from_millis(100))? {
                    match event::read()? {
                        Event::Key(KeyEvent { code: KeyCode::Char('y'), .. }) |
                        Event::Key(KeyEvent { code: KeyCode::Char('Y'), .. }) => {
                            println!("y");
                            break;
                        }
                        Event::Key(KeyEvent { code: KeyCode::Enter, .. }) |
                        Event::Key(KeyEvent { code: KeyCode::Char('n'), .. }) |
                        Event::Key(KeyEvent { code: KeyCode::Char('N'), .. }) => {
                            println!("n");
                            return Ok("".to_string()); // Return empty string to ignore paste
                        }
                        Event::Key(KeyEvent { code: KeyCode::Esc, .. }) => {
                            println!("ESC");
                            return Ok("".to_string());
                        }
                        _ => continue,
                    }
                }
            }
        }

        // Process the content (remove excessive whitespace, etc.)
        let processed = self.preprocess_paste_content(content);
        Ok(processed)
    }

    /// Preprocess paste content to clean it up
    fn preprocess_paste_content(&self, content: &str) -> String {
        // Remove trailing whitespace from each line
        let lines: Vec<String> = content
            .lines()
            .map(|line| line.trim_end().to_string())
            .collect();
        
        // Remove excessive empty lines (more than 2 consecutive empty lines become 2)
        let mut processed_lines = Vec::new();
        let mut empty_count = 0;
        
        for line in lines {
            if line.is_empty() {
                empty_count += 1;
                if empty_count <= 2 {
                    processed_lines.push(line);
                }
            } else {
                empty_count = 0;
                processed_lines.push(line);
            }
        }
        
        processed_lines.join("\n")
    }

    /// Get character index from cursor position (handles multi-byte chars)
    fn get_char_index_at_cursor(&self, text: &str, cursor_pos: usize) -> usize {
        text.char_indices().nth(cursor_pos).map(|(i, _)| i).unwrap_or(text.len())
    }

    /// Redraw the entire input line
    fn redraw_line(&self, input_buffer: &str, cursor_pos: usize) -> Result<()> {
        // Move to beginning of line
        execute!(stdout(), crossterm::cursor::MoveToColumn(0))?;
        // Clear line
        execute!(stdout(), crossterm::terminal::Clear(crossterm::terminal::ClearType::CurrentLine))?;
        // Print prompt + input
        print!("> {}", input_buffer);
        // Move cursor to correct position
        execute!(stdout(), crossterm::cursor::MoveToColumn((cursor_pos + 2) as u16))?;
        io::stdout().flush()?;
        Ok(())
    }
}

/// Result of input reading operation
#[derive(Debug, Clone)]
pub enum InputResult {
    Input(String),
    Empty,
    Interrupted,
}

/// Simple input reader with transparent large paste handling
pub fn read_simple_input(prompt: &str) -> Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    
    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(_) => {
            let trimmed = input.trim();
            
            // Handle large pastes transparently (like Claude Code does)
            if is_likely_large_paste(trimmed) {
                handle_large_paste_simple(trimmed)
            } else {
                Ok(trimmed.to_string())
            }
        },
        Err(e) => Err(anyhow::anyhow!("Failed to read input: {}", e)),
    }
}

/// Read input with prompt positioned above persistent status bar
pub fn read_input_above_status_bar(prompt: &str) -> Result<String> {
    // Position cursor above status bar (second to last line)
    print!("\x1b7");  // Save current cursor position
    print!("\x1b[999;1H");  // Move to bottom of terminal
    print!("\x1b[1A");  // Move up one line (above status bar)
    print!("\r\x1B[K");  // Clear the line
    
    // Display prompt
    print!("{}", prompt);
    io::stdout().flush()?;
    
    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(_) => {
            let trimmed = input.trim();
            
            // Clear the input line after reading
            print!("\x1b[999;1H");  // Move to bottom
            print!("\x1b[1A");  // Move up one line
            print!("\r\x1B[K");  // Clear the line
            
            // Restore original cursor position
            print!("\x1b8");
            io::stdout().flush()?;
            
            // Handle large pastes transparently (like Claude Code does)
            if is_likely_large_paste(trimmed) {
                handle_large_paste_simple(trimmed)
            } else {
                Ok(trimmed.to_string())
            }
        },
        Err(e) => {
            // Clean up and restore cursor position on error
            print!("\x1b[999;1H");
            print!("\x1b[1A");
            print!("\r\x1B[K");
            print!("\x1b8");
            io::stdout().flush().ok();
            Err(anyhow::anyhow!("Failed to read input: {}", e))
        }
    }
}

/// Read input integrated with the persistent status bar
pub fn read_input_in_status_bar() -> Result<String> {
    use crate::thinking_display::PersistentStatusBar;
    
    // Show chat input prompt
    PersistentStatusBar::show_chat_input();
    
    // Give the status bar display loop time to show the prompt
    std::thread::sleep(std::time::Duration::from_millis(50));
    
    // Position cursor on the input line (above status bar)
    let term_height = if let Ok((_, height)) = crossterm::terminal::size() {
        height as usize
    } else {
        24
    };
    print!("\x1b[{};3H", term_height - 1);  // Move to input line after "> "
    io::stdout().flush()?;
    
    let mut input = String::new();
    let result = match io::stdin().read_line(&mut input) {
        Ok(_) => {
            let trimmed = input.trim();
            
            // Handle large pastes transparently
            if is_likely_large_paste(trimmed) {
                handle_large_paste_simple(trimmed)
            } else {
                Ok(trimmed.to_string())
            }
        },
        Err(e) => Err(anyhow::anyhow!("Failed to read input: {}", e))
    };
    
    // Hide chat input and return to status display
    PersistentStatusBar::hide_chat_input();
    
    // Clear the input line after reading
    let term_height = if let Ok((_, height)) = crossterm::terminal::size() {
        height as usize
    } else {
        24
    };
    print!("\x1b[{};1H", term_height - 1);  // Move to input line
    print!("\r\x1B[K");  // Clear the line
    io::stdout().flush()?;
    
    result
}

/// Check for ESC key during operations (simple polling version)
pub async fn check_for_escape_key() -> bool {
    // Use crossterm to check for ESC without blocking
    if crossterm::event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
        if let Ok(event) = crossterm::event::read() {
            if let crossterm::event::Event::Key(key_event) = event {
                return key_event.code == crossterm::event::KeyCode::Esc;
            }
        }
    }
    false
}

/// Check if input looks like a large paste (simple detection)
fn is_likely_large_paste(input: &str) -> bool {
    // Large pastes typically have:
    // - More than 500 characters, OR
    // - More than 5 lines, OR  
    // - Contains code-like patterns
    let char_count = input.len();
    let line_count = input.lines().count();
    let has_code_patterns = input.contains("function ") || input.contains("def ") || 
                           input.contains("class ") || input.contains("import ") ||
                           input.contains("<?php") || input.contains("<!DOCTYPE") ||
                           input.contains("#include") || input.contains("use std::");
    
    char_count > 500 || line_count > 5 || (char_count > 200 && has_code_patterns)
}

/// Handle large paste transparently 
fn handle_large_paste_simple(content: &str) -> Result<String> {
    let lines = content.lines().count();
    let chars = content.len();
    
    // Just give a brief, unobtrusive notification
    println!("{}📋 Processing large paste ({} lines, {} chars){}", 
            crate::cli::colors::GRAY_DIM, lines, chars, crate::cli::colors::RESET);
    
    // Clean up the content (remove excessive whitespace)
    let cleaned = clean_pasted_content(content);
    Ok(cleaned)
}

/// Clean pasted content (remove excessive whitespace, normalize line endings)
fn clean_pasted_content(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    
    // Remove trailing whitespace from each line and excessive empty lines
    let mut cleaned_lines = Vec::new();
    let mut consecutive_empty = 0;
    
    for line in lines {
        let trimmed = line.trim_end();
        
        if trimmed.is_empty() {
            consecutive_empty += 1;
            // Allow max 2 consecutive empty lines
            if consecutive_empty <= 2 {
                cleaned_lines.push(trimmed.to_string());
            }
        } else {
            consecutive_empty = 0;
            cleaned_lines.push(trimmed.to_string());
        }
    }
    
    // Remove trailing empty lines
    while let Some(last) = cleaned_lines.last() {
        if last.is_empty() {
            cleaned_lines.pop();
        } else {
            break;
        }
    }
    
    cleaned_lines.join("\n")
}

/// Interrupt-aware operation runner
pub struct InterruptibleOperation {
    receiver: mpsc::UnboundedReceiver<InterruptSignal>,
}

impl InterruptibleOperation {
    pub fn new() -> (Self, mpsc::UnboundedSender<InterruptSignal>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (Self { receiver }, sender)
    }

    /// Check if operation should be interrupted
    pub fn check_interrupt(&mut self) -> Option<InterruptSignal> {
        self.receiver.try_recv().ok()
    }

    /// Async check for interrupt (doesn't block)
    pub async fn check_interrupt_async(&mut self) -> Option<InterruptSignal> {
        match self.receiver.try_recv() {
            Ok(signal) => Some(signal),
            Err(_) => None,
        }
    }
}

/// Initialize terminal for enhanced input handling
pub fn init_enhanced_terminal() -> Result<()> {
    InputHandler::enable_bracketed_paste()?;
    Ok(())
}

/// Cleanup terminal after enhanced input handling
pub fn cleanup_enhanced_terminal() -> Result<()> {
    InputHandler::disable_bracketed_paste()?;
    // Ensure we're not in raw mode
    let _ = terminal::disable_raw_mode();
    Ok(())
}