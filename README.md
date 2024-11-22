# Python Conference RAG Assistant

A smart assistant that helps you discover and learn about Python conferences worldwide using RAG (Retrieval-Augmented Generation) technology.

## Features

- Scrapes Python conference data from python.org/events
- Uses LlamaIndex for document indexing and retrieval
- Implements RAG with Gemini Pro LLM
- Vector embeddings using HuggingFace's BAAI/bge-small-en-v1.5
- Persistent storage of conference information

## Tech Stack

- Python 3.12
- LlamaIndex
- Google's Gemini Pro
- BeautifulSoup4
- HuggingFace Embeddings
- Vector Store Index

## Setup

1. Clone the repository
2. Install dependencies:
```sh
pip install -r requirements.txt

Create a .env file with your API keys:
GOOGLE_API_KEY=your_key_here

python pyconf_rag.py

 ```

## Project Structure

```

├── pyconf_rag.py          # Main conference scraper and RAG implementation
├── data/                  # Persistent storage for vector embeddings and documents
├── .env                   # Environment variables and API keys
├── requirements.txt       # Project dependencies
```


## Contribution

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear messages.
4. Push your changes to your fork.
5. Open a pull request with a detailed description of your changes.

MIT
