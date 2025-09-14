import os
import json
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RetailChatbot:
    """RAG Chatbot for retail policy and product support."""
    
    def __init__(self):
        """Initialize the chatbot with Azure OpenAI configuration and ChromaDB."""
        load_dotenv()
        
        # Azure OpenAI configuration
        self.embedding_endpoint = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT")
        self.embedding_api_key = os.getenv("AZURE_OPENAI_EMBED_API_KEY")
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
        
        self.llm_endpoint = os.getenv("AZURE_OPENAI_LLM_ENDPOINT")
        self.llm_api_key = os.getenv("AZURE_OPENAI_LLM_API_KEY")
        self.llm_model = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")
        
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        # Validate environment variables
        self._validate_environment()
        
        # Initialize Azure OpenAI components
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.embedding_endpoint,
            api_key=self.embedding_api_key,
            model=self.embedding_model,
            api_version=self.api_version,
        )
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.llm_endpoint,
            api_key=self.llm_api_key,
            deployment_name=self.llm_model,
            api_version=self.api_version,
            temperature=0.2,
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.retriever = None
        
        # Conversation history for multi-turn conversations
        self.conversation_history = []
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful Walmart customer support assistant. Use the provided 
                information to answer product and policy questions accurately and clearly. 
                Always cite the retrieved information in your answer and be specific about 
                Walmart policies.
                
                When answering:
                1. Be clear and concise
                2. Reference specific Walmart policies mentioned in the context
                3. If the information is not in the context, say so
                4. Be helpful and professional
                
                Context: {context}"""
            ),
            ("human", "{question}")
        ])
    
    def _validate_environment(self):
        """Validate that all required environment variables are set."""
        required_vars = [
            "AZURE_OPENAI_EMBED_ENDPOINT",
            "AZURE_OPENAI_EMBED_API_KEY", 
            "AZURE_OPENAI_EMBED_DEPLOYMENT",
            "AZURE_OPENAI_LLM_ENDPOINT",
            "AZURE_OPENAI_LLM_API_KEY",
            "AZURE_OPENAI_LLM_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    def load_documents(self, file_path: str = "products/walmart_policies.json") -> List[Document]:
        """Load documents from JSON file and convert to LangChain Document objects."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for item in data:
                doc = Document(
                    page_content=item["content"],
                    metadata={"id": item["id"], "source": "walmart_policies"}
                )
                documents.append(doc)
            
            print(f"Loaded {len(documents)} documents from {file_path}")
            return documents
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Document file not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
    
    def setup_vectorstore(self, documents: List[Document], refresh: bool = False):
        """Set up ChromaDB vector store with documents."""
        collection_name = "walmart_policies"
        
        # Check if collection already exists and refresh is not requested
        if not refresh:
            try:
                existing_collections = self.chroma_client.list_collections()
                collection_exists = any(col.name == collection_name for col in existing_collections)
                
                if collection_exists:
                    print(f"Found existing ChromaDB collection: {collection_name}")
                    
                    # Use existing collection
                    self.vectorstore = Chroma(
                        client=self.chroma_client,
                        collection_name=collection_name,
                        embedding_function=self.embeddings,
                    )
                    
                    # Create retriever
                    self.retriever = self.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    
                    print("Using existing vector store with persistent data")
                    return
                
            except Exception as e:
                print(f"Error checking existing collections: {e}")
        
        # Collection doesn't exist or refresh requested
        if refresh:
            print("Refreshing ChromaDB collection...")
            try:
                self.chroma_client.delete_collection(collection_name)
                print("Deleted existing collection")
            except Exception:
                pass  # Collection might not exist
        else:
            print("Creating new ChromaDB collection...")
        
        # Split documents if they are too large
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Create vector store using Chroma
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name=collection_name,
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        print(f"Vector store created with {len(split_docs)} document chunks")
    
    def retrieve_context(self, question: str) -> str:
        """Retrieve relevant context for the given question."""
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call setup_vectorstore first.")
        
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Azure OpenAI based on the question and context."""
        formatted_prompt = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)
        return response.content
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Process a single question through the RAG pipeline."""
        # Retrieve relevant context
        context = self.retrieve_context(question)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Store in conversation history
        conversation_turn = {
            "question": question,
            "context": context,
            "answer": answer
        }
        self.conversation_history.append(conversation_turn)
        
        return conversation_turn
    
    def run_demo(self):
        """Run demo mode with predefined questions."""
        print("=" * 60)
        print("RETAIL RAG CHATBOT - DEMO MODE")
        print("=" * 60)
        
        demo_questions = [
            "Can I return a bicycle if I've ridden it outside?",
            "What is the return policy for electronics?",
            "Do I need a receipt to return items?",
            "Can I return prescription medications?",
            "What are the benefits of Walmart Plus membership?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\nDemo Question {i}: {question}")
            print("-" * 50)
            
            result = self.chat(question)
            
            print(f"Retrieved Context:\n{result['context']}")
            print(f"\nGenerated Answer:\n{result['answer']}")
            print("\n" + "=" * 60)
    
    def run_interactive(self):
        """Run interactive mode with multi-turn conversations."""
        print("=" * 60)
        print("RETAIL RAG CHATBOT - INTERACTIVE MODE")
        print("=" * 60)
        print("Ask me anything about Walmart policies, returns, or products!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'history' to see conversation history.")
        print("Type 'clear' to clear conversation history.")
        print("-" * 60)
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("\nRetail Assistant: Thank you for using our support! Have a great day!")
                    break
                
                elif question.lower() == 'history':
                    self._show_history()
                    continue
                
                elif question.lower() == 'clear':
                    self.conversation_history.clear()
                    print("\nConversation history cleared.")
                    continue
                
                # Process the question
                print("\nRetail Assistant: Let me find that information for you...")
                result = self.chat(question)
                
                print(f"\nRetail Assistant: {result['answer']}")
                
            except KeyboardInterrupt:
                print("\n\nRetail Assistant: Thank you for using our support! Have a great day!")
                break
            except Exception as e:
                print(f"\nSorry, I encountered an error: {e}")
    
    def _show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("\nNo conversation history yet.")
            return
        
        print(f"\nConversation History ({len(self.conversation_history)} turns):")
        print("-" * 50)
        
        for i, turn in enumerate(self.conversation_history, 1):
            print(f"\n{i}. Q: {turn['question']}")
            print(f"   A: {turn['answer'][:100]}{'...' if len(turn['answer']) > 100 else ''}")


def main():
    """Main function to run the chatbot."""
    parser = argparse.ArgumentParser(description="Retail RAG Chatbot for Policy and Product Support")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with predefined questions")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--refresh", action="store_true", help="Refresh the vector store (rebuild from documents)")
    
    args = parser.parse_args()
    
    # If no arguments provided, default to demo mode
    if not args.demo and not args.interactive:
        args.demo = True
    
    try:
        # Initialize the chatbot
        chatbot = RetailChatbot()
        
        # Load documents and setup vector store
        documents = chatbot.load_documents()
        chatbot.setup_vectorstore(documents, refresh=args.refresh)
        
        # Run the appropriate mode
        if args.demo:
            chatbot.run_demo()
        
        if args.interactive:
            chatbot.run_interactive()
    
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
