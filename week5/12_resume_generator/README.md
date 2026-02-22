# Resume Generation Using LLaMA3 Locally

An AI-powered resume generator that creates professional resumes using LLaMA3 locally, ensuring data privacy and offline capabilities.

## Objectives

- Develop an AI-powered agent that generates professional resumes based on user input
- Utilize LLaMA 3 locally to ensure data privacy and offline capabilities
- Implement prompt engineering for different types of resumes
- Generate well-structured Markdown resumes

## Problem Statement

Job seekers often struggle to create tailored resumes that highlight their strengths and align with specific job roles. This solution builds an AI agent that:

- Collects user information from structured data
- Generates well-structured, professional resumes
- Supports different resume types (technical, entry-level, general)

## Features

- **Local LLaMA3 Integration**: Runs entirely offline for data privacy
- **Object-Oriented Design**: Clean, maintainable code structure
- **Multiple Resume Types**: Technical, Entry-level, and General resume templates
- **Prompt Engineering**: Specialized prompts for different user profiles
- **Markdown Output**: Professional resumes in Markdown format
- **Error Handling**: Comprehensive exception handling throughout
- **Batch Processing**: Generate multiple resumes from sample data

## Project Structure

```txt
resume_generator/
├── resume_generator.py    # Main script with LlamaModel class
├── requirements.txt       # Python dependencies
├── prompts.txt           # Resume generation prompts
├── input.json            # Sample user data (3 profiles)
├── outputs/              # Generated resume files
│   ├── resume_alex_chen.md
│   ├── resume_sarah_johnson.md
│   └── resume_michael_rodriguez.md
└── README.md             # This file
```

## Installation

1. **Clone or navigate to the project directory:**

   ```bash
   cd /home/nggocnn/ai-dev/week5/resume_generator
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download a LLaMA model** (choose one):

    **Option A: Llama-2-7B-Chat (Recommended - ~4GB)**

   ```bash
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
   ```

   **Option B: Code Llama 7B (~4GB)**

   ```bash
   wget https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf
   ```

   **Option C: Llama-2-13B-Chat (Better quality - ~7GB)**

   ```bash
   wget https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf
   ```

4. **Update model path** in `resume_generator.py` if needed:

   ```python
   model_path = "your-downloaded-model.gguf"
   ```

## Usage

### Quick Start

Run the main script to generate resumes for all sample users:

```bash
python resume_generator.py
```

### Sample Output

The script will:

1. Load 3 sample user profiles from `input.json`
2. Generate appropriate resumes using LLaMA3
3. Save resumes as Markdown files in the `outputs/` directory

### Expected Console Output

```txt
Resume Generation Using LLaMA3 Locally
==================================================
Model found at: llama-2-7b-chat.Q4_K_M.gguf
Loaded 3 user profiles from input.json
Loaded 3 prompts from prompts.txt
Loading LLaMA model from llama-2-7b-chat.Q4_K_M.gguf...
LLaMA model loaded successfully!

Generating resumes for 3 users...
--------------------------------------------------

Processing User 1: Alex Chen
Generating resume for Alex Chen...
Resume generated successfully!
Resume saved to: outputs/resume_alex_chen.md
Resume 1 completed: outputs/resume_alex_chen.md

Processing User 2: Sarah Johnson
Generating resume for Sarah Johnson...
Resume generated successfully!
Resume saved to: outputs/resume_sarah_johnson.md
Resume 2 completed: outputs/resume_sarah_johnson.md

Processing User 3: Michael Rodriguez
Generating resume for Michael Rodriguez...
Resume generated successfully!
Resume saved to: outputs/resume_michael_rodriguez.md
Resume 3 completed: outputs/resume_michael_rodriguez.md

Resume generation complete!
Successfully generated: 3/3 resumes
Output directory: outputs
```

## Sample Data

The `input.json` contains 3 diverse user profiles:

1. **Alex Chen**: Senior Software Engineer (Technical Resume)
2. **Sarah Johnson**: Recent CS Graduate (Entry-level Resume)
3. **Michael Rodriguez**: Marketing Manager (General Resume)

## Resume Types

### Technical Resume

- Emphasizes programming skills and technical projects
- Highlights frameworks, tools, and technologies
- Includes GitHub links and technical achievements

### Entry-Level Resume

- Focuses on education and academic projects
- Highlights internships and coursework
- Emphasizes potential and learning ability

### General Resume

- Professional format for non-technical roles
- Emphasizes achievements and leadership
- Industry-agnostic approach

## Configuration

### Prompts Customization

Edit `prompts.txt` to customize resume generation prompts:

- `RESUME_GENERATION_PROMPT`: General purpose prompt
- `TECHNICAL_RESUME_PROMPT`: For software engineers
- `ENTRY_LEVEL_RESUME_PROMPT`: For recent graduates

### Model Parameters

Adjust model parameters in the `LlamaModel` class:

- `n_ctx`: Context window size (default: 4096)
- `temperature`: Creativity level (default: 0.7)
- `max_tokens`: Maximum response length (default: 2000)

## Expected Outcomes

- Complete, professional resumes generated based on user input
- Enhanced understanding of integrating LLaMA 3 with Python
- Experience in prompt engineering and handling model outputs
- Local model deployment for data privacy

## Concepts Covered

- **AI-Powered Resume Generation**: Leveraging LLaMA 3 for structured resume creation
- **Local Model Deployment**: Running LLaMA 3 locally for privacy and offline capability
- **Prompt Engineering**: Crafting effective prompts for different resume types
- **Object-Oriented Programming**: Clean, maintainable code structure
- **Exception Handling**: Robust error handling throughout the application
