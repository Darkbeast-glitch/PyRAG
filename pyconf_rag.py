from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from decouple import config
import os
import requests


# Set environment variables (remove unnecessary ones)
os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")
# Configure settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm_model = Gemini(
    model="models/gemini-pro"
)  # Ensure this path/model is correct


# LOADING DATA afrom a web
def scrape_python_conference():
    try:
        # Debug print
        print("Starting conference scraping...")

        url = "https://www.python.org/events/"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        events = []

        # Find all event listings
        event_list = soup.find("ul", class_="list-recent-events")
        if not event_list:
            print("No event list found")
            return []

        for event in event_list.find_all("li"):
            event_data = {
                "name": (
                    event.find("h3", class_="event-title").text.strip()
                    if event.find("h3", class_="event-title")
                    else "No title"
                ),
                "date": (
                    event.find("time").text.strip() if event.find("time") else "No date"
                ),
                "location": (
                    event.find("span", class_="event-location").text.strip()
                    if event.find("span", class_="event-location")
                    else "No location"
                ),
                "url": (
                    event.find("h3", class_="event-title").find("a")["href"]
                    if event.find("h3", class_="event-title")
                    else ""
                ),
            }
            events.append(event_data)

        print(f"Found {len(events)} events")
        return events

    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        return []


# convert scraped date in document
def events_to_documents(events):
    if not events:
        print("Warning: No events provided")
        return []

    documents = []
    try:
        for event in events:
            # Create document with event details and metadata
            text = f"""
            Event: {event.get('event_title', 'Unnamed')}
            Date: {event.get('date', 'No date')}
            Location: {event.get('location', 'No location')}
            Description: {event.get('description', 'No description')}
            URL: {event.get('url', 'No URL')}
            """

            metadata = {
                "event_name": event.get("name"),
                "event_date": event.get("date"),
                "event_location": event.get("location"),
                "source": "python_conference_scraper",
            }

            doc = Document(text=text.strip(), metadata=metadata)
            documents.append(doc)

        print(f"Successfully processed {len(documents)} events into documents")
        return documents

    except Exception as e:
        print(f"Error processing events to documents: {str(e)}")
        return []


events = scrape_python_conference()
if not events:
    print("No events found during scraping")
    exit(1)

documents = events_to_documents(events)
if not documents:
    print("No documents to process")
    exit(1)
# Split the document into sentences
text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)

# Creae or update the index
try:
    # Create index with progress feedback
    print("Creating vector store index...")
    index = VectorStoreIndex.from_documents(
        documents, transformations=[text_splitter], show_progress=True
    )

    # Persist with error handling
    print("Persisting storage context...")
    index.storage_context.persist(persist_dir="./data")

    print("Successfully completed processing")
except Exception as e:
    print(f"Error during processing: {e}")  # Persist with error handling
print("Persisting storage context...")


# Load storage context
storage_context = StorageContext.from_defaults(persist_dir="./data")

# Load or rebuild index
index = load_index_from_storage(storage_context)


# Create a prompt template
template = """You are PythonConfExpert, an AI assistant specializing in Python conferences and events worldwide. You have access to up-to-date information about Python conferences, workshops, and meetups.

ROLE:
- Provide accurate information about Python conferences and events
- Help users find relevant conferences based on their needs
- Share details about venues, dates, and conference programs

INSTRUCTIONS:
1. Always structure your responses in this format:
   - Event Details (name, date, location)
   - Key Information
   - Additional Context (if available)

2. When answering questions:
   - Be concise and factual
   - Include specific dates and locations
   - Mention pricing if available
   - Add official website/links when present

3. Response Guidelines:
   - If information is uncertain, acknowledge the uncertainty
   - If data is outdated, mention the last update date
   - For missing information, say: "This information is not available in my current dataset"
   - Never make up or guess at conference details

CONTEXT: {context}

USER QUERY: {question}

RELEVANT EVENTS: {events}

Please provide your response following the above guidelines:
"""
prompt_tmpl = PromptTemplate(
    template=template,
    template_var_mappings={"query_str": "question", "context_str": "context"},
)


# Configure retriever
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

# Configure response synthesizer
response_synthesizer = get_response_synthesizer(llm=Settings.llm_model)

# Assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_tmpl})

# Query the index
response = query_engine.query("Can you name all the PyCon conferences?")
print(response)
