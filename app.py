from flask import Flask, render_template, jsonify, request 
from pinecone import Pinecone, ServerlessSpec
import pinecone 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.prompt import *
from src.helper import format_docs 
import getpass
import os

app = Flask(__name__)

load_dotenv()

# Initialise Pinecone (do not need to create new index as loaded from store_index)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)


index_name = "medical-chatbot3"  

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Loading Index 
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# Retriever 
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# LLM Model 
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)



# Retrieval Augmented Generation 
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = rag_chain.invoke(input)
    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


