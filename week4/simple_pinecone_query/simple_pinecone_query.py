import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI
from dotenv import load_dotenv


@dataclass
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str


def load_config() -> AzureOpenAIConfig:
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview").strip()

    missing = [
        k
        for k, v in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_KEY": api_key,
            "AZURE_OPENAI_DEPLOYMENT": deployment,
        }.items()
        if not v
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nPlease set them (e.g., in a .env file) and re-run."
        )

    return AzureOpenAIConfig(endpoint, api_key, deployment, api_version)


def get_azure_openai_client(cfg: AzureOpenAIConfig) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=cfg.api_key,
        azure_endpoint=cfg.endpoint,
        api_version=cfg.api_version,
    )


# Load configuration and create client
embedding_config = load_config()
embedding_client = get_azure_openai_client(embedding_config)


def load_products_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Load product data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default products.")
        return [
            {"id": "prod1", "title": "Red T-Shirt", "description": "Comfortable cotton t-shirt in bright red"},
            {"id": "prod2", "title": "Blue Jeans", "description": "Stylish denim jeans with relaxed fit"},
            {"id": "prod3", "title": "Black Leather Jacket", "description": "Genuine leather jacket with classic style"},
            {"id": "prod4", "title": "White Sneakers", "description": "Comfortable sneakers perfect for daily wear"},
            {"id": "prod5", "title": "Green Hoodie", "description": "Warm hoodie made of organic cotton"},
        ]


def get_embedding(text: str) -> List[float]:
    """Generate embedding for given text using Azure OpenAI."""
    response = embedding_client.embeddings.create(
        input=text, 
        model=embedding_config.deployment
    )
    return response.data[0].embedding


def load_pinecone_config():
    """Load and validate Pinecone configuration from environment variables."""
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "products").strip()
    namespace = os.getenv("PINECONE_NAMESPACE", "ai-dev").strip()
    
    missing = []
    if not api_key:
        missing.append("PINECONE_API_KEY")
    if not index_name:
        missing.append("PINECONE_INDEX_NAME")
    
    if missing:
        raise RuntimeError(
            "Missing required Pinecone environment variables: "
            + ", ".join(missing)
            + "\nPlease set them (e.g., in a .env file) and re-run."
        )
    
    return {
        "api_key": api_key,
        "index_name": index_name,
        "namespace": namespace if namespace else None,
        "top_k": int(os.getenv("TOP_K", "3"))
    }


def initialize_pinecone_index(config: dict, dimension: int = 1536) -> Any:
    """Initialize Pinecone client and create/connect to index."""
    pc = Pinecone(api_key=config["api_key"])
    
    # Check if index exists, create if not
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    if config["index_name"] not in existing_indexes:
        print(f"Creating new index: {config['index_name']}")
        # Try different regions for free tier
        try:
            pc.create_index(
                name=config["index_name"],
                dimension=dimension,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except Exception as e:
            print(f"Failed to create index with us-east-1, trying us-west-2: {e}")
            try:
                pc.create_index(
                    name=config["index_name"],
                    dimension=dimension,
                    spec=ServerlessSpec(cloud="aws", region="us-west-2"),
                )
            except Exception as e2:
                print(f"Failed to create index with us-west-2: {e2}")
                raise e2
    else:
        print(f"Using existing index: {config['index_name']}")
    
    return pc.Index(config["index_name"])


def upsert_products_to_pinecone(index: Any, products: List[Dict[str, Any]], namespace: str = None) -> None:
    """Upsert product embeddings to Pinecone index."""
    print("Generating embeddings and upserting to Pinecone...")
    
    vectors = []
    for product in products:
        text_to_embed = f"{product['title']} {product['description']}"
        embedding = get_embedding(text_to_embed)
        
        vectors.append({
            "id": product["id"],
            "values": embedding,
            "metadata": {
                "title": product["title"],
                "description": product["description"]
            }
        })
    
    if namespace:
        index.upsert(vectors, namespace=namespace)
    else:
        index.upsert(vectors)
    print(f"Successfully upserted {len(vectors)} product embeddings.")


def query_similar_products(index: Any, query: str, top_k: int = 3, namespace: str = None) -> Any:
    """Query Pinecone for most similar products."""
    query_embedding = get_embedding(query)
    
    if namespace:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
    else:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
    
    return results


def load_queries_from_file(file_path: str) -> List[str]:
    """Load queries from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            queries = [line.strip() for line in file if line.strip()]
            return queries
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default queries.")
        return [
            "clothing item for summer",
            "comfortable shoes for walking", 
            "warm winter clothing"
        ]


def main():
    """Main function to demonstrate Pinecone RAG implementation."""
    try:
        print("Loading configurations...")
        pinecone_config = load_pinecone_config()
        
        print("Loading product data...")
        products = load_products_from_json("products.json")
        print(f"Loaded {len(products)} products")
        
        print("Initializing Pinecone...")
        index = initialize_pinecone_index(pinecone_config)
        
        upsert_products_to_pinecone(index, products, pinecone_config["namespace"])
        
        print("Loading queries...")
        queries = load_queries_from_file("queries.txt")
        print(f"Loaded {len(queries)} queries")
        
        print("\nProduct Similarity Search Results:")
        print("=" * 80)
        
        for i, query in enumerate(queries, 1):
            results = query_similar_products(index, query, pinecone_config["top_k"], pinecone_config["namespace"])
            
            print(f"\n{i}. Query: '{query}'")
            print("-" * 60)
            
            if results.matches:
                for j, match in enumerate(results.matches, 1):
                    metadata = match.metadata
                    print(f"   {j}. {metadata['title']} (Score: {match.score:.4f})")
                    print(f"      {metadata['description']}")
            else:
                print(f"   No matches found for query: '{query}'")
        
        print(f"\nCompleted similarity search for {len(queries)} queries!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()