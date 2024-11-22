from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
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

# Set environment variables (remove unnecessary ones)
os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")
# Configure settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm_model = Gemini(
    model="models/gemini-pro"
)  # Ensure this path/model is correct


# Load the data
documents = SimpleDirectoryReader(input_files=["llama2.pdf"]).load_data()

# Split the document into sentences
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
Settings.text_splitter = text_splitter

# Index the document
index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])
index.storage_context.persist(persist_dir="/blogs")

# Rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="/blogs")

# Load index
index = load_index_from_storage(storage_context)

# Create a prompt template
template = """
You are a knowledgeable and precise assistant specialized in question-answering tasks, 
particularly from academic and research-based sources. 
Your goal is to provide accurate, concise, and contextually relevant answers based on the given information.

Instructions:

Comprehension and Accuracy: Carefully read and comprehend the provided context from the research paper to ensure accuracy in your response.
Conciseness: Deliver the answer in no more than three sentences, ensuring it is concise and directly addresses the question.
Truthfulness: If the context does not provide enough information to answer the question, clearly state, "I don't know."
Contextual Relevance: Ensure your answer is well-supported by the retrieved context and does not include any information beyond what is provided.

Remember if no context is provided please say you don't know the answer
Here is the question and context for you to work with:

\nQuestion: {question} \nContext: {context} \nAnswer:"""

prompt_tmpl = PromptTemplate(
    template=template,
    template_var_mappings={"query_str": "question", "context_str": "context"},
)

# Configure retriever
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

# Configure response synthesizer
response_synthesizer = get_response_synthesizer(llm=Settings.llm_model)

# Assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_tmpl})

# Query the index
response = query_engine.query("What are the different variants of Llama?")
print(response)
