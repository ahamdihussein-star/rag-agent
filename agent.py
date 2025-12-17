from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

def get_context(query: str, top_k: int = 3):
    """Get relevant context from Pinecone"""
    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n\n"
    
    return context

def ask(question: str):
    """Ask the agent a question"""
    
    print(f"🤔 Question: {question}\n")
    
    # Get relevant context
    context = get_context(question)
    
    # Create prompt
    prompt = f"""You are a helpful assistant. Answer the question based on the context provided.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    # Get response
    response = llm.invoke(prompt)
    
    print(f"🤖 Answer: {response.content}")
    return response.content

def chat():
    """Interactive chat mode"""
    print("=" * 50)
    print("🤖 RAG Agent Ready!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        question = input("\n📝 You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print()
        ask(question)

if __name__ == "__main__":
    chat()