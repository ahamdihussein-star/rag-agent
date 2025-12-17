from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index
index_name = os.getenv("PINECONE_INDEX_NAME")

# Check if index exists
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,  # OpenAI text-embedding-3-large dimension
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    print(f"✅ Index '{index_name}' created successfully!")
else:
    print(f"ℹ️ Index '{index_name}' already exists.")