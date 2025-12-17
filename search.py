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

def search(query: str, top_k: int = 3):
    """Search for similar documents"""
    
    print(f"🔍 Searching for: {query}")
    
    # Create query embedding
    query_vector = embeddings.embed_query(query)
    
    # Search in Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    print(f"\n📚 Found {len(results['matches'])} results:\n")
    
    for i, match in enumerate(results['matches']):
        score = match['score']
        text = match['metadata']['text']
        source = match['metadata']['source']
        
        print(f"--- Result {i+1} (Score: {score:.2f}) ---")
        print(f"Source: {source}")
        print(f"Text: {text[:200]}...")
        print()
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python search.py <query>")
    else:
        query = " ".join(sys.argv[1:])
        search(query)