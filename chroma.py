# knowledge_base.py
from langchain_classic.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

vectorstore = Chroma(
    collection_name="coding_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def add_docs(docs) -> int:
    chunks = splitter.split_documents(docs)     
    vectorstore.add_documents(chunks)           
    return len(chunks)


def search(query, k=4):
    return vectorstore.similarity_search(query, k=k)