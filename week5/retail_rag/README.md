# Retail RAG Chatbot for Product and Policy Support

A Retrieval-Augmented Generation (RAG) chatbot that helps retail employees and customers quickly get accurate answers about Walmart product details, store policies, and return/exchange guidelines.

## Features

- **RAG Architecture**: Uses ChromaDB for vector storage and Azure OpenAI for embeddings and text generation
- **Multi-turn Conversations**: Supports interactive conversations with conversation history
- **Demo Mode**: Predefined questions showcasing the chatbot capabilities
- **Interactive Mode**: Real-time Q&A with retail policy support
- **ChromaDB Persistence**: Stores vectors persistently and reuses them across sessions
- **Vector Store Refresh**: Option to rebuild the vector store from documents
- **Command Line Interface**: Easy-to-use CLI with multiple modes

## Architecture

The chatbot implements a complete RAG pipeline:

1. **Document Loading**: Loads Walmart policies from JSON format
2. **Vector Storage**: Uses ChromaDB to store document embeddings
3. **Retrieval**: Finds relevant policy documents based on user questions
4. **Generation**: Uses Azure OpenAI to generate contextual answers
5. **Response**: Provides clear, policy-backed answers to users

## Requirements

- Python 3.8+
- Azure OpenAI API access
- Required Python packages (see requirements.txt)

## Installation

1. Clone or download the project files
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env` file:

   ```bash
   AZURE_OPENAI_LLM_ENDPOINT=your_llm_endpoint
   AZURE_OPENAI_LLM_API_KEY=your_llm_api_key
   AZURE_OPENAI_LLM_DEPLOYMENT=your_llm_deployment

   AZURE_OPENAI_EMBED_ENDPOINT=your_embed_endpoint
   AZURE_OPENAI_EMBED_API_KEY=your_embed_api_key
   AZURE_OPENAI_EMBED_DEPLOYMENT=your_embed_deployment

   AZURE_OPENAI_API_VERSION=2024-07-01-preview
   ```

## Usage

### Demo Mode (Default)

Run predefined questions to see the chatbot in action:

```bash
python retail_chatbot.py --demo
```

### Interactive Mode

Start a conversation with the chatbot:

```bash
python retail_chatbot.py --interactive
```

### Both Modes

Run demo first, then interactive:

```bash
python retail_chatbot.py --demo --interactive
```

### Refresh Vector Store

Rebuild the vector store from documents:

```bash
python retail_chatbot.py --demo --refresh
```

## Interactive Commands

During interactive mode, you can use these commands:

- `quit`, `exit`, `bye`: End the conversation
- `history`: View conversation history
- `clear`: Clear conversation history

## Example Demo Output

```txt
Demo Question 1: Can I return a bicycle if I've ridden it outside?
--------------------------------------------------
Retrieved Context:
Bicycles purchased at Walmart can be returned within 90 days if not used outdoors and with all accessories present.

Generated Answer:
Based on Walmart's return policy, bicycles that have been ridden outdoors cannot be returned. The policy specifically states that bicycles can only be returned within 90 days if they have NOT been used outdoors and all accessories are present. Since you've ridden the bicycle outside, it would not be eligible for return under Walmart's current policy.
```

## Policy Documents

The chatbot includes 15 comprehensive Walmart policy documents covering:

- Electronics return policy (30 days with receipt)
- Grocery return policy (90 days, except perishables)  
- Warranty information (1-year on electronics/appliances)
- Walmart Plus benefits (free shipping)
- Prescription medication restrictions
- Open-box item returns
- No-receipt returns (store credit with ID)
- Price matching policy
- Vision Center returns (60 days)
- Cell phone return requirements
- Gift card redemption rules
- Seasonal merchandise policies
- Bicycle return conditions
- Online order returns
- Fraud prevention measures

## Technical Implementation

### Key Components

1. **WalmartRAGChatbot Class**: Main chatbot implementation
2. **ChromaDB Integration**: Persistent vector storage
3. **Azure OpenAI Integration**: Embeddings and text generation
4. **LangChain Pipeline**: Document processing and retrieval
5. **Command Line Interface**: argparse-based CLI

### Vector Storage

- Uses ChromaDB for persistent vector storage
- Automatically chunks large documents
- Similarity search with top-k retrieval
- Stores conversation history for multi-turn support

### Safety Features

- Environment variable validation
- Error handling for missing files/APIs
- Graceful handling of user interrupts
- Clear user feedback and status messages

## Benefits of RAG vs Traditional Approaches

### RAG Advantages

1. **Dynamic Knowledge**: Unlike static FAQs, RAG can combine multiple policy documents to answer complex questions
2. **Contextual Understanding**: Semantic search finds relevant policies even when exact keywords aren't used
3. **Always Current**: Easy to update knowledge base by adding new documents
4. **Nuanced Answers**: Can handle edge cases by combining multiple policy fragments
5. **Audit Trail**: Shows exactly which policies were used to generate each answer

### Compared to Keyword Search

- **Better Intent Understanding**: Handles synonyms and related concepts
- **Context Awareness**: Understands question context beyond just keywords
- **Comprehensive Responses**: Can synthesize information from multiple sources
- **Natural Language**: Users can ask questions naturally instead of using specific keywords

### Compared to Static FAQs

- **Unlimited Questions**: Not limited to pre-written Q&A pairs
- **Combination Logic**: Can answer questions requiring multiple policy considerations
- **Freshness**: Automatically incorporates new policies without manual FAQ updates
- **Specificity**: Provides detailed, policy-backed answers rather than generic responses

## Files Structure

```txt
retail_rag/
├── .env                           # Environment variables (not in repo)
├── .env.sample                    # Environment template
├── requirements.txt               # Python dependencies
├── retail_chatbot.py              # Main chatbot implementation
├── README.md                      # This documentation
├── products/
│   └── walmart_policies.json     # Policy documents
└── chroma_db/                     # Vector database (created at runtime)
```
