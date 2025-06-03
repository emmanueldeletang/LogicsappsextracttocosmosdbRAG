# LogicsappsextracttocosmosdbRAG
Get  a RAG on document with CosmosDB using the template implement in logc APPS 

a full web application where you can load file from a storage account or sharepoint and get a  llm and vector search i the document vector in the cosmosdb , the question and  answer will be put in  cache to save time and reduce latency 
 they are 3 type of search 
   Vector 
  full text 
  hybrid ..


## Features
- Vector search using Azure cosmsodb Nosql
- full text and hybrid search using Azure cosmsodb Nosql
- load file base on Azure Logic Apps as a document indexer for Azure Cosmos DB!
With this release, you can now use Logic Apps connectors and templates to ingest documents directly into Cosmos DB’s vector store—powering AI workloads like Retrieval-Augmented Generation (RAG) with ease.
This new capability orchestrates the full ingestion pipeline—from fetching documents to parsing, chunking, embedding, and indexing—allowing you to unlock insights from unstructured content across your enterprise systems.
Check out the announcement from Azure Cosmos team about this capability!
How It Works
Here’s how Logic Apps powers the ingestion flow:
  Connect to Source Systems
  While Logic Apps has more than 1400+ prebuilt connectors to pull documents from various systems, this experience streamlines the entire process via out of box templates to pull data from sources like Azure Blob Storage.
  Parse and Chunk Documents
  AI-powered parsing actions extract raw text. Then, the Chunk Document action:
  Tokenizes content into language model-friendly units
  Splits it into semantically meaningful chunks
  This ensures optimal size and quality for embedding and retrieval.
  Generate Embeddings with Azure OpenAI
  The chunks are passed to Azure OpenAI via connector to generate embeddings (e.g., using text-embedding-3-small). These vectors capture the meaning of your content for precise semantic search.
  Write to Azure Cosmos DB Vector Store
  Embeddings and metadata (like title, tags, and timestamps) are indexed in Cosmos DB’s, using a schema optimized for filtering, semantic ranking, and retrieval.
  Logic Apps Templates: Fast Start, Full Flexibility
  We’ve created ready-to-use templates to help you get started fast:
    📄 Blob Storage – Simple Text Parsing
    🧾 Blob Storage – OCR with Azure Document Intelligence
    📁 SharePoint – Simple Text Parsing
    🧠 SharePoint – OCR with Azure Document Intelligence
    Each template is customizable—so you can adapt it to your business needs or expand it with additional steps.

all details https://devblogs.microsoft.com/cosmosdb/new-generally-available-and-preview-search-capabilities-in-azure-cosmos-db-for-nosql/#public-preview:-azure-logic-apps-document-indexer


- Use cosmosdb Nosql as cache to save latency

## Requirements
- Tested only with Python 3.12
- Azure OpenAI account
- Azure Cosmos DB mongo VCORE

## Setup
- install and configure the logic apps , use text field for full text and vector for the vector ...https://devblogs.microsoft.com/cosmosdb/new-generally-available-and-preview-search-capabilities-in-azure-cosmos-db-for-nosql/#public-preview:-azure-logic-apps-document-indexer
- git clone the repository 
- Create virtual environment: python -m venv .venv
- Activate virtual ennvironment: .venv\scripts\activate
- Install required libraries: pip install -r requirements.txt
- Replace keys with your own values in example.env
- don't forget to have the model openAI one text-embbeding and one GPT ( can be 4.0 ,3.5 ) ...
- TO LAUNCH THE APPLICATION JUST do Streamlit run app.py
- have fun

