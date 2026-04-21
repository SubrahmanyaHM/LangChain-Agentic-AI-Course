import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
#from langchain_openai import OpenAIEmbeddings
#from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import hashlib

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(r'C:\Users\subra\OneDrive\Desktop\Analytics\AI\Building AI Agents with LangChain\Building AI Agents with Langchain\langchain-course\mediumblog1.txt', encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    texts_str= [doc.page_content for doc in texts]

    embeddings = OllamaEmbeddings(
    model="embeddinggemma",
)
    #Just for testing the embedding output, not necessary for ingestion
    query_result = embeddings.embed_documents(texts_str)
    print(query_result[0])
    print(len(query_result[0]))  # Should be 768 for embeddinggemma
    print("ingesting...")

    #Create unique IDs for each document chunk using a hash of the content. Even if run multiple times, the same content will have the same ID, preventing duplicates in Chroma.
    def make_id(doc):
        return hashlib.md5(doc.page_content.encode()).hexdigest()

    ids = [make_id(doc) for doc in texts]

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="my_docs",
        ids=ids
    )
    print("finish")

