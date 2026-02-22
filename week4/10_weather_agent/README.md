# AI Weather & Search Agent

A powerful AI assistant built with Langchain that can handle real-time weather queries and web searches. The agent intelligently routes queries to the appropriate tools: OpenWeather API for weather information and Tavily Search API for web searches.

## Features

- **Weather Queries**: Get current weather information for any city worldwide
- **Web Search**: Search for latest news, information, and general queries
- **Intelligent Routing**: Automatically determines which tool to use based on the query
- **Conversation History**: Maintains context across multiple interactions
- **Interactive Chat**: Console-based chat interface
- **Demo Mode**: Predefined demo queries for testing

## Quick Start

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Azure OpenAI
  - OpenWeatherMap
  - Tavily Search

### Installation

1. **Clone or navigate to the project directory**:

   ```bash
   cd /path/to/weather_agent
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   - Copy `.env.sample` to `.env`
   - Fill in your API keys in the `.env` file:

   ```bash
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_DEPLOYMENT=GPT-4o-mini
   AZURE_OPENAI_API_VERSION=2024-07-01-preview
   OPENWEATHERMAP_API_KEY=your_openweather_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

### Usage

#### Interactive Mode (Default)

```bash
python weather_agent.py
```

#### Demo Mode

```bash
python weather_agent.py --demo
```

## API Keys Setup

### 1. Azure OpenAI

1. Go to [Azure Portal](https://portal.azure.com/)
2. Create an Azure OpenAI resource
3. Get your endpoint, API key, and deployment name

### 2. OpenWeatherMap API

1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Get your API key from the dashboard

### 3. Tavily Search API

1. Go to [Tavily](https://app.tavily.com/)
2. Create an account
3. Get your API key from the dashboard

## Example Queries

### Weather Queries

- "What's the weather in Paris today?"
- "How's the weather in Tokyo right now?"
- "Tell me the current weather in New York"

### Search Queries

- "Tell me about the latest news in AI"
- "Search for recent developments in renewable energy"
- "Who won the FIFA World Cup 2022?"

## Architecture

The agent uses the following components:

- **Langchain**: Framework for building LLM applications
- **LangGraph**: For creating React-style agents
- **Azure OpenAI**: Large language model for reasoning
- **OpenWeather API**: Weather data provider
- **Tavily Search**: Web search capabilities

### Agent Flow

1. User submits a query
2. Agent analyzes the query intent
3. Routes to appropriate tool:
   - Weather tool for weather-related queries
   - Search tool for general information queries
4. Processes the tool response
5. Returns formatted answer to the user

## Project Structure

```txt
weather_agent/
├── weather_agent.py     # Main agent implementation
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .env.sample          # Environment variables template
└── .env                 # Your actual environment variables (not in git)
```

## Code Structure

### WeatherSearchAgent Class

The main class that orchestrates the AI agent:

- `setup_environment()`: Loads and validates API keys
- `setup_tools()`: Initializes weather and search tools
- `setup_llm()`: Configures Azure OpenAI
- `setup_agent()`: Creates the Langchain agent
- `query()`: Processes user queries
- `run_demo()`: Runs predefined demo queries
- `run_interactive()`: Starts interactive chat mode

### Tools

1. **Weather Tool**: Custom Langchain tool wrapping OpenWeatherMap API
2. **Search Tool**: Tavily search tool for web queries

## Customization

### Adding New Tools

To add new tools, follow this pattern:

```python
@tool
def my_custom_tool(parameter: str) -> str:
    """Tool description for the LLM.
    
    Args:
        parameter (str): Description of the parameter
        
    Returns:
        str: Description of the return value
    """
    # Your tool implementation
    return result
```

### Modifying Agent Behavior

You can customize the agent by:

- Adjusting the LLM temperature in `setup_llm()`
- Modifying tool descriptions
- Adding system prompts
- Changing response formatting
