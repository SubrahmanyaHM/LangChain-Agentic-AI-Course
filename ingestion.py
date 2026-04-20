import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
#from langchain_openai import OpenAIEmbeddings
#from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(r'C:\Users\subra\OneDrive\Desktop\Analytics\AI\Building AI Agents with LangChain\Building AI Agents with Langchain\langchain-course\mediumblog1.txt', encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    texts = [doc.page_content for doc in texts]

    embeddings = OllamaEmbeddings(
    model="embeddinggemma",
)

    query_result = embeddings.embed_documents(texts)
    print(query_result[:3])
    print("ingesting...")

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="my_docs",
    )
    print("finish")

