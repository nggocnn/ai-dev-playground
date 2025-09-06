import os
import sys
from typing import List, Dict
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent


class WeatherSearchAgent:
    """
    A conversational AI agent that can handle weather queries and web searches.
    """
    
    def __init__(self):
        """Initialize the agent with API keys and tools."""
        self.setup_environment()
        self.setup_tools()
        self.setup_llm()
        self.setup_agent()
        self.messages = []
    
    def setup_environment(self):
        """Setup environment variables for API keys."""
        # Load from .env file if it exists
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        # Validate required environment variables
        required_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT",
            "OPENWEATHERMAP_API_KEY",
            "TAVILY_API_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            print(f"Missing required environment variables: {', '.join(missing_vars)}")
            print("Please set these variables in your .env file.")
            sys.exit(1)
        
        print("Environment variables loaded successfully")
    
    def setup_tools(self):
        """Setup weather and search tools."""
        # Weather tool setup
        self.weather = OpenWeatherMapAPIWrapper()
        
        # Tavily search tool setup
        self.tavily_search_tool = TavilySearch(
            max_results=3,
            topic="general",
        )
        
        print("Tools initialized successfully")
    
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a given city.
        
        Args:
            city (str): The name of the city to get the weather for.
            
        Returns:
            str: A string describing the current weather in the specified city.
        """
        print(f" Getting weather for {city}...")
        weather = OpenWeatherMapAPIWrapper()
        try:
            result = weather.run(city)
            return result
        except Exception as e:
            return f"Sorry, I couldn't get the weather for {city}. Error: {str(e)}"
    
    def setup_llm(self):
        """Setup Azure OpenAI LLM."""
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.1,
        )
        print("Azure OpenAI LLM initialized successfully")
    
    def setup_agent(self):
        """Setup Langchain agent with tools."""
        tools = [self.get_weather, self.tavily_search_tool]
        
        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
        )
        print("Langchain agent created successfully")
    
    def format_response(self, response_content: str) -> str:
        """Format the agent response for better readability."""
        # Add some basic formatting
        if "weather" in response_content.lower():
            return f"{response_content}"
        elif "search" in response_content.lower() or "news" in response_content.lower():
            return f"{response_content}"
        else:
            return f"{response_content}"
    
    def query(self, user_input: str) -> str:
        """Process a user query and return the agent's response."""
        try:
            print(f"\nProcessing: {user_input}")
            
            # Add user message to conversation history
            self.messages.append({"role": "user", "content": user_input})
            
            # Get response from agent
            response = self.agent.invoke({"messages": self.messages})
            
            # Extract the assistant's response
            assistant_message = response["messages"][-1].content
            
            # Add assistant response to conversation history
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            return self.format_response(assistant_message)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"Error: {error_msg}")
            return error_msg
    
    def run_demo(self):
        """Run a demonstration with predefined questions."""
        print("\n" + "="*60)
        print("AI Weather & Search Agent Demo")
        print("="*60)
        
        # Mock user questions for automatic input
        demo_questions = [
            "What's the weather in Hanoi today?",
            "What's the weather in Paris?",
            "Tell me about the latest news in artificial intelligence.",
            "Search for recent developments in renewable energy.",
            "Who won the FIFA World Cup 2022?",
            "What's the weather like in Tokyo right now?",
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\nDemo Query {i}: {question}")
            print("-" * 50)
            
            response = self.query(question)
            print(f"AI Assistant: {response}")
            
            # Add a small delay for readability
            import time
            time.sleep(1)
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
    
    def run_interactive(self):
        """Run interactive chat mode."""
        print("\n" + "="*60)
        print("AI Weather & Search Agent - Interactive Mode")
        print("="*60)
        print("Ask me about weather or search for information!")
        print("Type 'exit', 'quit', or 'bye' to stop.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye', '']:
                    print("\nGoodbye! Thanks for using the AI Weather & Search Agent!")
                    break
                
                response = self.query(user_input)
                print(f"AI Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using the AI Weather & Search Agent!")
                break
            except Exception as e:
                print(f"Unexpected error: {str(e)}")


def main():
    """Main function to run the weather agent."""
    try:
        # Create and initialize the agent
        agent = WeatherSearchAgent()
        
        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == '--demo':
            # Run demo mode
            agent.run_demo()
        else:
            # Run interactive mode
            agent.run_interactive()
            
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
        print("Please check your API keys and internet connection.")


if __name__ == "__main__":
    main()
