# Patient Information Collection and Advisory Chatbot Agent

## Overview

This project implements an AI-powered chatbot agent that interactively collects patient information and provides preliminary health advice using LangGraph for conversation flow management and Azure OpenAI for natural language processing. The system features persistent vector storage, comprehensive report generation, and multiple interaction modes.

## Features

- **Interactive Patient Data Collection**: Collects patient name, age, symptoms, and medical history
- **Preliminary Health Advice**: Provides relevant health recommendations based on patient symptoms
- **Real-time Information Retrieval**: Uses Tavily API to fetch current health information
- **Conversational AI**: Natural dialogue flow managed by LangGraph
- **Persistent Vector Storage**: ChromaDB-based retrieval system with document persistence
- **Comprehensive Report Generation**: Saves detailed consultation reports to files
- **Multiple Operation Modes**: CLI support for initialization, demo, and interactive modes
- **Structured Medical Knowledge Base**: JSON-based medical conditions database

## Architecture

The chatbot uses:

- **LangGraph**: For managing conversation state and flow
- **Azure OpenAI**: For natural language understanding and generation (GPT-4o-mini)
- **ChromaDB**: For persistent vector-based similarity search of medical advice
- **Tavily**: For real-time web search capabilities
- **LangChain**: For tool integration and orchestration

## Setup Instructions

### 1. Environment Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the `.env.sample` file to `.env` and fill in your API credentials:

```bash
cp .env.sample .env
```

Required environment variables:

- `AZURE_OPENAI_LLM_ENDPOINT`: Your Azure OpenAI endpoint
- `AZURE_OPENAI_LLM_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_LLM_DEPLOYMENT`: Your deployment name (e.g., "GPT-4o-mini")
- `AZURE_OPENAI_EMBED_ENDPOINT`: Azure OpenAI embedding endpoint
- `AZURE_OPENAI_EMBED_API_KEY`: Azure OpenAI embedding API key
- `AZURE_OPENAI_EMBED_DEPLOYMENT`: Embedding deployment name (e.g., "text-embedding-3-small")
- `TAVILY_API_KEY`: Your Tavily API key (optional, for web search)

### 3. Initialize the Knowledge Base

Before first use, initialize the ChromaDB knowledge base:

```bash
python patient_info_collector.py --init
```

## Usage

### Command Line Arguments

The application supports multiple operation modes:

```bash
# Initialize ChromaDB with medical knowledge (run once)
python patient_info_collector.py --init

# Run demo mode with sample patients
python patient_info_collector.py --demo

# Start interactive chat mode
python patient_info_collector.py --interactive

# Default: Run demo mode
python patient_info_collector.py
```

### Interactive Chat Mode

Start a conversational session with the AI medical assistant:

```bash
python patient_info_collector.py --interactive
```

The interactive mode supports:

- Natural conversation flow
- Real-time symptom analysis
- Dynamic advice generation
- Session history management

## Code Structure

### Main Components

1. **PatientInfoCollector Class**: Main orchestrator with enhanced features
   - ChromaDB initialization and management
   - Report generation and file saving
   - Multi-mode operation support

2. **Knowledge Base**:

   - `medical_knowledge.json`: Structured medical conditions database
   - ChromaDB persistent vector store
   - Automatic document embedding and indexing

3. **Tools**:

   - `retrieve_advice`: Searches internal medical knowledge via ChromaDB
   - `search_web_info`: Retrieves real-time information via Tavily

4. **LangGraph Workflow**: Enhanced conversation flow with tool integration

5. **Report System**:

   - Console output formatting
   - File-based report persistence
   - Timestamped consultation records

### Project Structure

```txt
patient_info_collector/
├── patient_info_collector.py    # Main application
├── requirements.txt             # Dependencies
├── .env.sample                  # Environment template
├── README.md                    # Documentation
├── documents/
│   └── medical_knowledge.json   # Medical knowledge base
├── chroma_db/                   # ChromaDB persistent storage
└── reports/                     # Generated consultation reports
    ├── consultation_patient1_timestamp.txt
    └── consultation_patient2_timestamp.txt
```

### Sample Patient Data

The demo mode includes four sample patients with different symptoms:

1. **Alice Johnson** (28): Sore throat, mild fever, fatigue
2. **Bob Smith** (45): Persistent cough, shortness of breath (former smoker)
3. **Carol Davis** (35): Severe headache, dizziness, nausea (migraine history)
4. **David Wilson** (22): Stomach pain, loss of appetite, fatigue

## Report Generation

The system generates comprehensive consultation reports that are:

- **Displayed in console**: Real-time formatted output during consultation
- **Saved to files**: Persistent records in `reports/` directory
- **Timestamped**: Unique filenames with consultation date/time
- **Structured**: Consistent format with patient info, advice, and disclaimers

### Sample Report Format

```txt
PATIENT CONSULTATION REPORT
Generated on: 2025-09-14 16:23:07
============================================================

PATIENT INFORMATION:
Name: Alice Johnson
Age: 28
Symptoms: sore throat, mild fever, and fatigue
Medical History: no significant medical history

============================================================

CONSULTATION SUMMARY:
[Detailed AI-generated health advice and recommendations]

============================================================

DISCLAIMER:
This consultation report is generated by an AI chatbot and provides preliminary 
health advice only. This information should not replace professional medical 
consultation. Please consult with a qualified healthcare professional for 
proper diagnosis and treatment.
============================================================
```

## Technologies Used

- **Python 3.10+**: Core programming language
- **LangGraph 0.6.6**: Conversation flow and state management
- **Azure OpenAI**: LLM and embedding services
- **ChromaDB 1.0.20**: Persistent vector database
- **LangChain**: AI application framework and tool integration
- **Tavily API**: Real-time web search capabilities

## Features in Detail

### Medical Knowledge Base

- Comprehensive JSON database with 10+ medical conditions
- Structured symptom, treatment, and prevention information
- Automatic embedding and indexing in ChromaDB

### AI-Powered Consultation

- Natural language symptom analysis
- Context-aware medical advice generation
- Real-time information retrieval from web sources

### Report Management

- Dual output: console display + file persistence
- Timestamped consultation records
- Professional formatting with disclaimers

### CLI Interface

- Multiple operation modes (init, demo, interactive)
- Flexible configuration options
- Easy setup and deployment
