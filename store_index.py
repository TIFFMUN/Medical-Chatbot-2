from src.helper import load_pdf, text_split
from langchain.vectorstores import Pinecone
import pinecone 
from dotenv import load_dotenv
import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings





# Load .env file 
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# Extract Data
extracted_data = load_pdf("C:\\Users\\TIFFANY MUN\\Medical-Chatbot-2\\data")


# Split data into chunks 
text_chunks = text_split(extracted_data)


#Initialise Pinecone 
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot3"  

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)


# Create Vector Embeddings 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Store Vector Embeddings in Pinecone 
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

uuids = [str(uuid4()) for _ in range(len(text_chunks))]

vector_store.add_documents(documents=text_chunks, ids=uuids)