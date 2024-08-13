from flask import Flask, render_template, jsonify, request 
from pinecone import Pinecone, ServerlessSpec
import pinecone 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
from src.helper import format_docs
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

# Addition of Chat History 
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

from flask import Flask, request, jsonify, session
from flask_session import Session

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)  # Initialize session management


@app.route("/")
def index():
    return render_template('chat.html')

# Chatbot with Chat History 
@app.route("/get", methods=["GET", "POST"])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []
    msg = request.form["msg"]
    input = msg
    print("User Input:", input)
    result = conversational_rag_chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": session.get('session_id', 'default')}}
        )
    ai_response = result['answer']
    print("Response : ", ai_response)
    session['chat_history'].append({
            'user': input,
            'ai': ai_response
        })
    return str(ai_response)

# Chatbot without Chat History 
# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     result = rag_chain.invoke(input)
#     print("Response : ", result)
#     return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


