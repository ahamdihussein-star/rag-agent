from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def ingest_file(file_path: str):
    """Load and ingest a single file into Pinecone"""
    
    # Detect file type and load
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        print(f"❌ Unsupported file type: {file_path}")
        return
    
    # Load and split
    print(f"📄 Loading: {file_path}")
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    print(f"📦 Split into {len(chunks)} chunks")
    
    # Create embeddings and upsert
    print("🔄 Creating embeddings...")
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk.page_content)
        
        index.upsert(vectors=[{
            "id": f"{os.path.basename(file_path)}_{i}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "source": file_path
            }
        }])
    
    print(f"✅ Ingested: {file_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_path>")
    else:
        ingest_file(sys.argv[1])