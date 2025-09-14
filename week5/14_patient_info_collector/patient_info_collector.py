import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_tavily import TavilySearch
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import chromadb
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

def initialize_chromadb():
    """Initialize ChromaDB with medical knowledge from JSON file. Run this once."""
    print("Initializing ChromaDB with medical knowledge...")
    
    # Load medical knowledge from JSON file
    json_file_path = os.path.join(os.path.dirname(__file__), "documents", "medical_knowledge.json")
    
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found!")
        return False
    
    with open(json_file_path, 'r') as f:
        medical_data = json.load(f)
    
    # Convert JSON data to documents
    documents = []
    for item in medical_data["medical_knowledge"]:
        content_parts = [
            f"Medical Condition: {item['title']}",
            f"Category: {item['category']}",
            f"Description: {item['content']}",
            f"Symptoms: {', '.join(item['symptoms'])}",
            f"Treatment: {'. '.join(item['treatment'])}",
            f"Prevention: {'. '.join(item['prevention'])}",
            f"When to see a doctor: {'. '.join(item['when_to_see_doctor'])}"
        ]
        
        full_content = "\n".join(content_parts)
        
        documents.append(Document(
            page_content=full_content,
            metadata={
                "id": item["id"],
                "title": item["title"],
                "category": item["category"]
            }
        ))
    
    # Setup embeddings
    embedding_model = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_EMBED_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBED_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview"),
    )
    
    # Create ChromaDB vector store
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    db = Chroma.from_documents(
        documents, 
        embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"ChromaDB initialized with {len(documents)} medical conditions")
    print(f"Vector database saved to: {persist_directory}")
    return True

class PatientInfoCollector:
    def __init__(self):
        """Initialize the patient information collector with Azure OpenAI and existing ChromaDB."""
        self.setup_environment()
        self.load_existing_knowledge_base()
        self.setup_llm()
        self.setup_tools()
        self.create_graph()

    def setup_environment(self):
        """Set up environment variables for Azure OpenAI and Tavily."""
        self.azure_llm_endpoint = os.getenv("AZURE_OPENAI_LLM_ENDPOINT")
        self.azure_llm_api_key = os.getenv("AZURE_OPENAI_LLM_API_KEY")
        self.azure_llm_deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")
        self.azure_embed_endpoint = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT")
        self.azure_embed_api_key = os.getenv("AZURE_OPENAI_EMBED_API_KEY")
        self.azure_embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

    def load_existing_knowledge_base(self):
        """Load existing ChromaDB vector store."""
        persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
        
        if not os.path.exists(persist_directory):
            print("ChromaDB not found. Please run: python patient_info_collector.py --init")
            raise FileNotFoundError("ChromaDB not initialized. Run with --init flag first.")
        
        # Setup embeddings
        self.embedding_model = AzureOpenAIEmbeddings(
            model=self.azure_embed_deployment,
            api_key=self.azure_embed_api_key,
            azure_endpoint=self.azure_embed_endpoint,
            api_version=self.api_version,
        )
        
        # Load existing ChromaDB vector store
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})

    def setup_llm(self):
        """Setup Azure OpenAI language model."""
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_llm_endpoint,
            api_key=self.azure_llm_api_key,
            azure_deployment=self.azure_llm_deployment,
            api_version=self.api_version,
            temperature=0.7,
        )

    def setup_tools(self):
        """Setup tools for retrieving advice and web search."""
        @tool
        def retrieve_advice(user_input: str) -> str:
            """Searches internal medical documents for relevant patient advice based on symptoms."""
            docs = self.retriever.invoke(user_input)
            advice = "\n".join(doc.page_content for doc in docs)
            return f"Internal medical advice: {advice}"

        @tool
        def search_web_info(query: str) -> str:
            """Searches the web for additional health information using Tavily."""
            if not self.tavily_api_key:
                return "Web search not available - no Tavily API key provided."
            
            try:
                tavily_search = TavilySearch(api_key=self.tavily_api_key)
                results = tavily_search.invoke(f"health advice {query}")
                if results:
                    return f"Web search results: {str(results)[:500]}..."
                return "No relevant web information found."
            except Exception as e:
                return f"Web search error: {str(e)}"

        self.tools = [retrieve_advice, search_web_info]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def create_graph(self):
        """Create the LangGraph workflow for patient information collection."""
        def call_model(state: MessagesState):
            """Call the language model with current conversation state."""
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: MessagesState):
            """Determine if tools should be called or conversation should end."""
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END

        # Create tool node
        tool_node = ToolNode(self.tools)

        # Build the graph
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("call_model", call_model)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_edge(START, "call_model")
        graph_builder.add_conditional_edges("call_model", should_continue, ["tools", END])
        graph_builder.add_edge("tools", "call_model")
        
        self.graph = graph_builder.compile()

    def print_consultation_report(self, patient_data: Dict[str, Any], result: Dict[str, Any]):
        """Print consultation report to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
{'='*80}
PATIENT CONSULTATION REPORT
Generated on: {timestamp}
{'='*80}

PATIENT INFORMATION:
Name: {patient_data.get('name', 'N/A')}
Age: {patient_data.get('age', 'N/A')}
Symptoms: {patient_data.get('symptoms', 'N/A')}
Medical History: {patient_data.get('medical_history', 'N/A')}

{'='*80}

CONSULTATION SUMMARY:
{result.get('final_advice', 'No advice generated')}

{'='*80}

DISCLAIMER:
This consultation report is generated by an AI chatbot and provides preliminary 
health advice only. This information should not replace professional medical 
consultation. Please consult with a qualified healthcare professional for 
proper diagnosis and treatment.

{'='*80}
"""
        print(report)

    def save_consultation_report(self, patient_data: Dict[str, Any], result: Dict[str, Any]) -> str:
        """
        Save consultation report to a file in the reports directory.
        
        Args:
            patient_data: Dictionary containing patient information
            result: Dictionary with conversation results and advice
            
        Returns:
            String path to the saved report file
        """
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_name = patient_data.get("name", "unknown").replace(" ", "_").lower()
        filename = f"consultation_{patient_name}_{timestamp}.txt"
        filepath = os.path.join(reports_dir, filename)
        
        # Create report content
        report_content = f"""
PATIENT CONSULTATION REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*60}

PATIENT INFORMATION:
Name: {patient_data.get('name', 'N/A')}
Age: {patient_data.get('age', 'N/A')}
Symptoms: {patient_data.get('symptoms', 'N/A')}
Medical History: {patient_data.get('medical_history', 'N/A')}

{'='*60}

CONSULTATION SUMMARY:
{result.get('final_advice', 'No advice generated')}

{'='*60}

DISCLAIMER:
This consultation report is generated by an AI chatbot and provides preliminary 
health advice only. This information should not replace professional medical 
consultation. Please consult with a qualified healthcare professional for 
proper diagnosis and treatment.

Report generated by Patient Information Collection and Advisory Chatbot
{'='*60}
"""
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return filepath

    def collect_patient_info_and_advise(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process patient information and provide health advice.
        
        Args:
            patient_data: Dictionary containing patient information
                         Expected keys: name, age, symptoms, medical_history (optional)
        
        Returns:
            Dictionary with conversation results and advice
        """
        # Create conversation context
        name = patient_data.get("name", "Patient")
        age = patient_data.get("age", "unknown")
        symptoms = patient_data.get("symptoms", "no specific symptoms")
        medical_history = patient_data.get("medical_history", "none reported")
        
        # System message for the medical assistant
        system_message = SystemMessage(
            content="""You are a helpful medical assistant chatbot. Your role is to:
1. Collect and acknowledge patient information
2. Provide preliminary health advice based on symptoms
3. Use available tools to search for relevant medical guidance
4. Remind patients that this is preliminary advice and they should consult healthcare professionals for serious concerns
5. Be empathetic and professional in your responses

Guidelines:
- Always acknowledge the patient by name
- Provide clear, actionable advice
- Suggest when to seek immediate medical attention
- Use tools to enhance your responses with relevant information"""
        )
        
        # Human message with patient information
        user_message = HumanMessage(
            content=f"""Patient Information:
Name: {name}
Age: {age}
Current Symptoms: {symptoms}
Medical History: {medical_history}

Please provide preliminary health advice based on these symptoms and any relevant information you can find."""
        )
        
        # Process through the graph
        result = self.graph.invoke({
            "messages": [system_message, user_message]
        })
        
        # Prepare return data
        consultation_result = {
            "patient_info": patient_data,
            "conversation": result["messages"],
            "final_advice": result["messages"][-1].content if result["messages"] else "No advice generated"
        }
        
        return consultation_result

    def interactive_chat_mode(self):
        """Start an interactive chat session with the medical chatbot."""
        print("=" * 80)
        print("PATIENT INFORMATION COLLECTION AND ADVISORY CHATBOT")
        print("Interactive Chat Mode")
        print("=" * 80)
        print("Welcome! I'm your AI medical assistant.")
        print("I can help collect your information and provide preliminary health advice.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for available commands.")
        print("-" * 80)
        
        # Initialize conversation history
        conversation_history = [
            SystemMessage(
                content="""You are a helpful medical assistant chatbot. Your role is to:
1. Collect patient information (name, age, symptoms, medical history)
2. Provide preliminary health advice based on symptoms
3. Use available tools to search for relevant medical guidance
4. Remind patients that this is preliminary advice and they should consult healthcare professionals for serious concerns
5. Be empathetic and professional in your responses
6. Continue conversations naturally and remember previous context

Guidelines:
- Always acknowledge the patient by name if provided
- Provide clear, actionable advice
- Suggest when to seek immediate medical attention
- Use tools to enhance your responses with relevant information
- Maintain conversation context throughout the session"""
            )
        ]
        
        patient_info = {}
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nChatbot: Thank you for using our medical advisory service.")
                    print("Remember to consult with healthcare professionals for proper diagnosis.")
                    print("Take care and stay healthy!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'info':
                    if patient_info:
                        print(f"\nCurrent patient information:")
                        for key, value in patient_info.items():
                            print(f"  {key.replace('_', ' ').title()}: {value}")
                    else:
                        print("\nNo patient information collected yet.")
                    continue
                
                # Add user message to conversation
                conversation_history.append(HumanMessage(content=user_input))
                
                # Check if we're collecting basic patient info
                if not patient_info.get('name') and any(keyword in user_input.lower() for keyword in ['name is', 'i am', "i'm", 'call me']):
                    # Try to extract name
                    for phrase in ['name is', 'i am', "i'm", 'call me']:
                        if phrase in user_input.lower():
                            name = user_input.lower().split(phrase)[1].strip().split()[0]
                            patient_info['name'] = name.title()
                            break
                
                # Process through the graph
                result = self.graph.invoke({
                    "messages": conversation_history
                })
                
                # Get the response
                if result["messages"]:
                    response = result["messages"][-1].content
                    print(f"\nChatbot: {response}")
                    
                    # Add response to conversation history
                    conversation_history.append(result["messages"][-1])
                else:
                    print("\nChatbot: I apologize, but I couldn't generate a response. Please try again.")
                
            except KeyboardInterrupt:
                print("\n\nChatbot: Goodbye! Take care of your health.")
                break
            except Exception as e:
                print(f"\nChatbot: I encountered an error: {str(e)}")
                print("Please try again or type 'quit' to exit.")

    def _show_help(self):
        """Show available commands in interactive mode."""
        print("\nAvailable commands:")
        print("  help  - Show this help message")
        print("  info  - Show current patient information")
        print("  quit  - Exit the chat (also: exit, bye)")
        print("\nYou can ask me about:")
        print("  - Your symptoms and health concerns")
        print("  - General health advice")
        print("  - When to see a doctor")
        print("  - Treatment recommendations")
        print("\nExample: 'Hi, my name is John and I have a headache'")
        print("-" * 50)

def run_sample_consultations():
    """Run sample patient consultations with dummy data."""
    # Initialize the patient info collector
    collector = PatientInfoCollector()
    
    # Sample patient data
    sample_patients = [
        {
            "name": "Alice Johnson",
            "age": 28,
            "symptoms": "sore throat, mild fever, and fatigue",
            "medical_history": "no significant medical history"
        },
        {
            "name": "Bob Smith", 
            "age": 45,
            "symptoms": "persistent cough for 3 weeks, shortness of breath",
            "medical_history": "former smoker, quit 5 years ago"
        },
        {
            "name": "Carol Davis",
            "age": 35,
            "symptoms": "severe headache, dizziness, nausea",
            "medical_history": "history of migraines"
        },
        {
            "name": "David Wilson",
            "age": 22,
            "symptoms": "stomach pain, loss of appetite, feeling tired",
            "medical_history": "no significant medical history"
        }
    ]
    
    print("=" * 80)
    print("PATIENT INFORMATION COLLECTION AND ADVISORY CHATBOT")
    print("=" * 80)
    
    for i, patient in enumerate(sample_patients, 1):
        print(f"\n{'='*20} CONSULTATION {i} {'='*20}")
        print(f"Processing patient: {patient['name']}")
        print("-" * 60)
        
        try:
            # Get advice for the patient
            result = collector.collect_patient_info_and_advise(patient)
            
            # Print consultation report to console
            collector.print_consultation_report(patient, result)
            
            # Save consultation report to file
            report_path = collector.save_consultation_report(patient, result)
            print(f"Consultation report saved to: {report_path}")
            
        except Exception as e:
            print(f"Error processing patient {patient['name']}: {str(e)}")
        
        print("-" * 60)

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Patient Information Collection and Advisory Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python patient_info_collector.py --init          # Initialize ChromaDB (run once)
  python patient_info_collector.py --demo          # Run demo with sample patients
  python patient_info_collector.py --interactive   # Start interactive chat mode
        """
    )
    
    parser.add_argument(
        "--init", 
        action="store_true", 
        help="Initialize ChromaDB with medical knowledge (run this once)"
    )
    parser.add_argument(
        "--demo", 
        action="store_true", 
        help="Run demo mode with sample patients"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Start interactive chat mode"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    try:
        if args.init:
            print("=" * 80)
            print("INITIALIZING CHROMADB")
            print("=" * 80)
            if initialize_chromadb():
                print("Initialization complete!")
                print("You can now run:")
                print("  python patient_info_collector.py --demo")
                print("  python patient_info_collector.py --interactive")
            else:
                print("Initialization failed!")
                
        elif args.demo:
            print("=" * 80)
            print("DEMO MODE - SAMPLE PATIENT CONSULTATIONS")
            print("=" * 80)
            run_sample_consultations()
            
        elif args.interactive:
            print("=" * 80)
            print("STARTING INTERACTIVE MODE")
            print("=" * 80)
            collector = PatientInfoCollector()
            collector.interactive_chat_mode()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run: python patient_info_collector.py --init")
    except Exception as e:
        print(f"Application error: {str(e)}")
        print("Please ensure all environment variables are properly set and try again.")

if __name__ == "__main__":
    main()
