# Simple Pinecone Query - Product Similarity Search

This project demonstrates using Pinecone vector database for product similarity search by retrieving the top 3 most similar product records for a given query.

## Overview

The application:

- Loads product data from a JSON file
- Reads search queries from a text file
- Generates embeddings using Azure OpenAI
- Stores embeddings in Pinecone vector database
- Performs similarity search for all queries and displays results

## Features

- **Vector Storage**: Uses Pinecone for efficient vector storage and retrieval
- **Embeddings**: Leverages Azure OpenAI text-embedding models
- **Batch Query Processing**: Processes multiple queries from a text file
- **Product Search**: Finds top-k most similar products for each query
- **Configurable**: Environment-based configuration with validation
- **Namespace Support**: Supports Pinecone namespaces for data organization
- **Error Handling**: Robust configuration validation and error messages
- **Clean Output**: Streamlined results display without step-by-step comments

## Prerequisites

- Python 3.8+
- Pinecone account and API key
- Azure OpenAI account with embedding model deployment

## Installation

1. Clone or download this project
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (copy `.env.sample` to `.env` and update values):

   ```bash
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_DEPLOYMENT=text-embedding-3-small
   AZURE_OPENAI_API_VERSION=2024-07-01-preview
   
   # Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=products
   PINECONE_NAMESPACE=ai-dev
   
   # Search Configuration
   TOP_K=3
   ```

## Usage

Run the main script:

```bash
python simple_pinecone_query.py
```

## Project Structure

```text
simple_pinecone_query/
├── simple_pinecone_query.py  # Main application file
├── products.json             # Sample product data (30 products)
├── queries.txt               # Search queries to process
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in git)
├── .env.sample              # Sample environment variables
└── README.md                # This file
```

## How It Works

1. **Configuration Loading**: Validates Azure OpenAI and Pinecone environment variables
2. **Data Loading**: Loads product information from `products.json` (30 diverse products)
3. **Query Loading**: Reads search queries from `queries.txt` file
4. **Embedding Generation**: Creates vector embeddings for products and queries using Azure OpenAI
5. **Vector Storage**: Stores embeddings in Pinecone index with metadata and namespace
6. **Batch Processing**: Processes all queries and performs similarity search
7. **Results Display**: Shows top-3 similar products for each query with scores

## Sample Output

```text
Loading configurations...
Loading product data...
Loaded 30 products
Initializing Pinecone...
Using existing index: products
Generating embeddings and upserting to Pinecone...
Successfully upserted 30 product embeddings.
Loading queries...
Loaded 10 queries

Product Similarity Search Results:
================================================================================

1. Query: 'clothing item for summer'
------------------------------------------------------------
   1. Summer Dress (Score: 0.5762)
      Lightweight floral summer dress made from breathable fabric, perfect for hot weather and beach outings
   2. Tank Top (Score: 0.5311)
      Breathable cotton tank top in various colors, ideal for hot summer days and layering
   3. Red T-Shirt (Score: 0.4853)
      Comfortable cotton t-shirt in bright red color, perfect for casual wear and summer activities

2. Query: 'comfortable shoes for walking'
------------------------------------------------------------
   1. White Sneakers (Score: 0.6035)
      Comfortable sneakers perfect for daily wear, made with breathable materials and cushioned sole
   2. Beach Sandals (Score: 0.5043)
      Comfortable flip-flops with arch support and non-slip sole, ideal for beach and pool activities
   3. Casual Sneakers (Score: 0.4672)
      Versatile canvas sneakers in classic design, comfortable for everyday wear and casual occasions

3. Query: 'wireless headphones for music'
------------------------------------------------------------
   1. Wireless Headphones (Score: 0.5995)
      Bluetooth noise-cancelling headphones with 30-hour battery life and premium sound quality

Completed similarity search for 10 queries!
```

## Configuration

The application uses dataclass-based configuration with robust validation:

- **Azure OpenAI Config**: Validates endpoint, API key, and deployment name
- **Pinecone Config**: Validates API key, index name, and namespace settings
- **Environment Validation**: Clear error messages for missing required variables

### Configuration Functions

- `load_config()`: Loads and validates Azure OpenAI configuration
- `load_pinecone_config()`: Loads and validates Pinecone configuration
- Both functions check for missing environment variables and provide helpful error messages

## Error Handling

- **Configuration Validation**: Checks for missing environment variables at startup
- **Graceful Fallback**: Uses default products if JSON file is not found
- **API Error Handling**: Proper exception handling for Azure OpenAI and Pinecone API calls
- **Index Management**: Automatic index creation with validation
- **Clear Error Messages**: Detailed error messages indicating exactly what's missing or wrong

## Customization

- **Product Data**: Modify `products.json` to include your own product catalog
- **Search Queries**: Edit `queries.txt` to test different search scenarios
- **Search Parameters**: Adjust `TOP_K` in `.env` to retrieve more/fewer results per query
- **Index Configuration**: Change `PINECONE_INDEX_NAME` and `PINECONE_NAMESPACE` as needed
- **Metadata Fields**: Add custom metadata fields for enhanced search capabilities
- **Query Processing**: Modify the main loop to process queries differently

## Data File Formats

### Product Data (`products.json`)

The `products.json` file contains 30 diverse products across multiple categories:

- **Clothing**: T-shirts, jeans, jackets, dresses, hoodies, etc.
- **Footwear**: Sneakers, running shoes, hiking boots, sandals, formal shoes
- **Electronics**: Wireless headphones, smartwatch
- **Accessories**: Backpack, sunglasses, crossbody bag, baseball cap, scarf
- **Athletic Wear**: Sports bra, yoga pants, thermal underwear
- **Professional**: Business suits, polo shirts, formal shoes

Each product object has the following structure:

```json
[
    {
        "id": "prod1",
        "title": "Product Title",
        "description": "Product description for embedding generation"
    }
]
```

### Query Data (`queries.txt`)

The `queries.txt` file contains search queries, one per line:

```text
clothing item for summer
comfortable shoes for walking
warm winter clothing
stylish denim apparel
wireless headphones for music
athletic wear for gym
formal business attire
casual weekend outfit
waterproof outdoor gear
lightweight travel accessories
```

## Dependencies

- `pinecone-client`: Pinecone vector database client
- `openai`: Azure OpenAI client library
- `python-dotenv`: Environment variable management

## Concepts Covered

- **Configuration Management**: Environment variable validation and dataclass configuration
- **Pinecone Integration**: Initializing client, creating indexes, and managing namespaces
- **Vector Embeddings**: Generating embeddings with Azure OpenAI text-embedding models
- **Vector Operations**: Upserting vectors with metadata to Pinecone
- **Batch Processing**: Processing multiple queries from external files
- **Similarity Search**: Performing cosine similarity search and retrieving top-k results
- **File I/O Operations**: Reading JSON product data and text-based queries
- **Error Handling**: Robust validation and clear error messages
- **Data Management**: JSON and text file loading with fallback mechanisms
