# AI Development Playground

Welcome to the AI Development Playground! This repository contains a comprehensive collection of AI projects and tools exploring various aspects of artificial intelligence, machine learning, natural language processing, computer vision, vector databases, and document processing. Each project is organized by week and focuses on specific applications ranging from chatbots to image analysis and document generation.

## Highlights

- **15 Complete Projects** across 5 weeks of development
- **Diverse AI Applications**: Chatbots, TTS, semantic search, image analysis, vector databases, document generation, RAG systems
- **Modern Tech Stack**: LangChain, HuggingFace, Pinecone, ChromaDB, Azure OpenAI, FastAPI, Next.js
- **Production-Ready**: Complete with documentation, requirements, and setup instructions
- **Interactive Examples**: Console interfaces, web applications, API endpoints, and document processing pipelines

## Repository Structure

The repository is structured as follows:

### Week 1

#### 1. Task Manager

- **Description**: A tool to manage and organize tasks.
- **Files**: `task_manager.py`, `tasks.json`
- **Usage**: Refer to the [README.md](week1/1_task_manager/README.md) in the `1_task_manager` folder for details.

#### 2. Instruction Generator

- **Description**: Generates instructions based on input data.
- **Files**: `instruction_generator.py`, `instructions.csv`, `tasks.csv`
- **Usage**: Refer to the [README.md](week1/2_instruction_generator/README.md) in the `2_instruction_generator` folder for details.

#### 3. Meeting Summarizer

- **Description**: Summarizes meeting transcripts into concise summaries.
- **Files**: `meeting_summarizer.py`, `inputs/`, `outputs/`
- **Usage**: Refer to the [README.md](week1/3_meeting_summarizer/README.md) in the `3_meeting_summarizer` folder for details.

### Week 2

#### 4. AI Chatbot

- **Description**: A chatbot powered by AI to handle various queries.
- **Files**: `ai_chatbot.py`
- **Usage**: Refer to the [README.md](week2/4_ai_chatbot/README.md) in the `4_ai_chatbot` folder for details.

#### 5. Function Calling

- **Description**: Demonstrates function calling mechanisms in Python.
- **Files**: `function_calling.py`
- **Usage**: Refer to the [README.md](week2/5_function_calling/README.md) in the `5_function_calling` folder for details.

#### 6. Maintenance Logs

- **Description**: Processes and analyzes maintenance logs.
- **Files**: `maintenance_logs.py`, `maintenance_logs.csv`
- **Usage**: Refer to the [README.md](week2/6_maintaince_logs/README.md) in the `6_maintaince_logs` folder for details.

### Week 3

#### 7. HuggingFace TTS

- **Description**: Text-to-Speech (TTS) system using HuggingFace models.
- **Files**: `hunggingface_tts.py`, `requirements.txt`, `inputs/`, `outputs/`
- **Usage**: Refer to the [README.md](week3/7_huggingface_tts/README.md) in the `7_huggingface_tts` folder for details.

#### 8. Semantic Search

- **Description**: Implements semantic search on a dataset of clothing products.
- **Files**: `semantic_search.py`, `requirements.txt`, `clothing_products.csv`, `queries.txt`, `search_results.csv`
- **Usage**: Refer to the [README.md](week3/8_semantic_search/README.md) in the `8_semantic_search` folder for details.

#### 9. Consultant Chatbot

- **Description**: A chatbot designed to assist with laptop recommendations.
- **Files**: `consultant_chatbot.py`, `laptops.csv`, `queries.txt`, `outputs.txt`, `requirements.txt`
- **Usage**: Refer to the [README.md](week3/9_consultant_chatbot/README.md) in the `9_consultant_chatbot` folder for details.

### Week 4

#### 10. Simple Pinecone Query

- **Description**: Product similarity search using Pinecone vector database to retrieve the top 3 most similar products for a given query.
- **Files**: `simple_pinecone_query.py`, `products.json`, `queries.txt`, `requirements.txt`
- **Usage**: Refer to the [README.md](week4/10_simple_pinecone_query/README.md) in the `10_simple_pinecone_query` folder for details.

#### 11. AI Weather & Search Agent

- **Description**: A powerful AI assistant built with Langchain that handles real-time weather queries and web searches with intelligent routing.
- **Files**: `weather_agent.py`, `requirements.txt`
- **Features**: Weather queries, web search, conversation history, interactive and demo modes
- **Usage**: Refer to the [README.md](week4/11_weather_agent/README.md) in the `11_weather_agent` folder for details.

#### 12. Satellite Cloud Detection

- **Description**: AI-powered satellite image cloud detection using Azure OpenAI's GPT-4 Vision model with FastAPI backend and Next.js frontend.
- **Files**: `backend/`, `frontend/`
- **Features**: Image upload, URL analysis, cloud classification with confidence scores
- **Usage**: Refer to the [README.md](week4/12_statellite_cloud_detection/README.md) in the `12_statellite_cloud_detection` folder for details.

### Week 5

#### 13. Resume Generator

- **Description**: AI-powered resume generator that creates professional resumes from structured input data using customizable templates and prompts.
- **Files**: `resume_generator.py`, `input.json`, `sample_data.json`, `prompts.txt`, `outputs/`
- **Features**: Template-based generation, structured input processing, multiple output formats
- **Usage**: Refer to the [README.md](week5/13_resume_generator/README.md) in the `13_resume_generator` folder for details.

#### 14. Patient Info Collector

- **Description**: Healthcare information system that collects, processes, and stores patient information using AI-powered document analysis and vector storage.
- **Files**: `patient_info_collector.py`, `documents/`, `reports/`, `chroma_db/`
- **Features**: Document processing, vector storage with ChromaDB, patient data management
- **Usage**: Refer to the [README.md](week5/14_patient_info_collector/README.md) in the `14_patient_info_collector` folder for details.

#### 15. Retail RAG Chatbot

- **Description**: Retrieval-Augmented Generation (RAG) chatbot for retail applications with product knowledge base and intelligent query processing.
- **Files**: `retail_chatbot.py`, `products/`, `chroma_db/`
- **Features**: Product knowledge retrieval, context-aware responses, vector-based search
- **Usage**: Refer to the [README.md](week5/15_retail_rag/README.md) in the `15_retail_rag` folder for details.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ (for Next.js projects)
- Git

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nggocnn/ai-dev-playground.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd ai-dev-playground
   ```

3. **Set up a Python virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies for specific projects:**

   Each project has its own `requirements.txt` file. Navigate to the project folder and install:

   ```bash
   cd week4/11_weather_agent
   pip install -r requirements.txt
   ```

### API Keys Setup

Many projects require API keys. Check each project's `.env.sample` file for required environment variables:

- **Azure OpenAI**: For LLM capabilities
- **OpenWeatherMap**: For weather data
- **Tavily Search**: For web search functionality
- **Pinecone**: For vector database operations
- **HuggingFace**: For model access

## Technologies Used

- **AI/ML**: LangChain, HuggingFace Transformers, Azure OpenAI, RAG (Retrieval-Augmented Generation)
- **Vector Databases**: Pinecone, ChromaDB
- **Web Frameworks**: FastAPI, Next.js
- **Data Processing**: Pandas, NumPy
- **Document Processing**: PDF parsing, text extraction, structured data generation
- **APIs**: OpenWeatherMap, Tavily Search
- **Frontend**: React, TypeScript, Tailwind CSS

## Learning Path

1. **Week 1**: Basic task management and text processing
2. **Week 2**: Chatbots and function calling
3. **Week 3**: Advanced NLP with TTS and semantic search
4. **Week 4**: Vector databases, multi-modal AI, and full-stack applications
5. **Week 5**: Document generation, healthcare AI, and RAG systems

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the projects.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.

---

Happy coding!
