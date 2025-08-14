![0](https://github.com/user-attachments/assets/7810225d-7f83-4255-b71a-2c83fac79ddd)

# Glimmer: A High-Performance AI Assistant CLI

`Glimmer` is a modular, high-performance command-line interface (CLI) built entirely in Rust. It serves as a powerful and responsive AI assistant for developers, integrating directly with the Gemini Protocol to handle a diverse range of coding tasks, from atomic file edits to multi-step architectural refactoring.

## Technical Features

* **Intelligent Conversational State Management:** A persistent conversation state, enabling multi-turn, contextual interactions. The system synthesizes prior messages with new prompts to maintain a coherent dialogue and provide highly-relevant responses.

* **Adaptive Task Analysis and Decomposition:** Glimmer's core intelligence is driven by analyzing task complexity and using adaptive metrics to form deep responses related to the query. This function dynamically assesses the complexity of a user's request and generates a structured `TaskComplexity` object. This object informs the execution strategy and, for complex tasks, provides a pre-computed vector of sub-steps.

    * **Low-Complexity Execution:** Trivial tasks are handled via a single, optimized API call using Gemini to minimize latency and token consumption.

    * **High-Complexity Orchestration:** For tasks requiring a multi-step approach, the system autonomously iterates through the pre-planned steps, executing each as an independent prompt-response loop. Intermediate results are cached and fed back into subsequent steps for enhanced context.

* **Project Context Synthesis:** The system intelligently aggregates crucial project context, including an analysis of the local directory structure and file contents. This proactive approach allows Glimmer to operate on the most relevant files without explicit user direction.

* **Transparent "Chain-of-Thought" Output:** Glimmer provides a detailed "thinking" trace for all multi-step tasks. This output documents the AI's internal reasoning process, showcasing each sub-step and the corresponding partial results, which is instrumental for debugging and building trust in the system's decisions.

* **Rust-Powered Performance:** By leveraging Rust's ownership model, zero-cost abstractions, and robust concurrency primitives (`tokio`, `rayon`), Glimmer achieves exceptional speed and memory safety. Dependencies such as `clap` for argument parsing and `reqwest` for asynchronous HTTP requests are carefully selected to ensure a performant and reliable architecture.

---

## Architecture and Workflow

Glimmer's architecture is a modular pipeline for intelligent task execution. The primary components and their interactions are as follows:

1.  **CLI Command Parsing (`main.rs`):** The program entry point uses the `clap` crate to parse command-line arguments. The `main` function dispatches commands such as `edit` and `chat` to their respective modules. The interactive chat mode is initialized by the `cli::chat::handle_chat` function.

2.  **Interactive Chat Loop (`chat.rs`):** The chat handler establishes a persistent session. It reads user input, manages the conversation history, and forwards requests to the core AI engine. The `ChatUI` struct provides a clean, user-friendly interface for displaying messages and system feedback.

3.  **AI Engine (`gemini.rs`):** This is the core logical unit.

    * The `gemini::analyze_task_complexity` function is the first point of contact for a user request. It makes a structured API call to the Gemini model to determine the task's complexity, providing the `steps` vector if required.

    * The `gemini::query_gemini` and `gemini::query_gemini_with_thinking` functions handle the low-level API communication, including request serialization and response deserialization. They are responsible for communicating with the Gemini Protocol endpoints.

4.  **Execution Strategy (`chat.rs`):** Based on the output of `analyze_task_complexity`, the chat handler either sends a single API request for simple tasks or orchestrates a sequence of API calls for complex tasks, managing the flow of information between each step.

5.  **Output Presentation (`chat.rs`):** The final output, along with any "thinking" trace, is formatted and displayed to the user. This ensures transparency and allows the user to inspect the AI's entire problem-solving process.

---

## Installation and Setup

### Prerequisites

* Rust toolchain (version 1.70.0 or newer)

* Git

### Build Instructions

```bash
# Clone the Glimmer repository
git clone https://github.com/OSXBasedAnon/glimmer.git
cd glimmer

# Build the project in release mode for optimal performance
cargo build --release

# The executable will be located at target/release/glimmer
./target/release/glimmer
```

### Configuration

For security, it is highly recommended to set your Gemini API key as an environment variable. Create a `.env` file in your project root with the following content:

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

And ensure you are loading this file in your application at startup. 

https://aistudio.google.com/

You can sign in there with your Google Account to get an API key for the free tier of the Gemini API.

---

## Usage

### Interactive Chat Mode

Execute Glimmer without any arguments to enter the interactive chat session, managed by the `cli::chat` module.

```bash
./target/release/glimmer
# >> Glimmer: A blazingly fast AI assistant is ready.
# >> > Refactor the `process_chat_input` function in `chat.rs` to use the new `ChatUI` struct.
```

### Command-Line Arguments

The `glimmer` executable supports several subcommands for specific tasks:

```
A high-performance, local Gemini-based CLI assistant

Usage: glimmer [OPTIONS] [COMMAND]

Commands:
  edit    Modifies files with AI assistance. Accepts a file path and a natural language instruction.
  clear   Clears the chat from context memory.
  export  Exports a conversation history to a specified format.
  compact Compacts project files for a focused context window.
  help    Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose           Enables verbose logging via the tracing crate.
  -c, --config <FILE>     Specifies an alternative path to the configuration file.
  -h, --help              Print help










