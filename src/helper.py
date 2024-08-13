from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Extract data from PDF 
def load_pdf(data):
    loader = DirectoryLoader(data, 
                        glob="*.pdf",
                        loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

