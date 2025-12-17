from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from datetime import datetime
import os
import uuid

# Load environment variables
load_dotenv()

# Initialize
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# User ID (in production, this would come from authentication)
USER_ID = "default_user"

def get_document_context(query: str, top_k: int = 3):
    """Get relevant context from documents"""
    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"type": {"$ne": "memory"}}
    )
    
    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n\n"
    
    return context

def get_memory_context(query: str, top_k: int = 15):
    """Get relevant past conversations from memory"""
    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"type": "memory", "user_id": USER_ID}
    )
    
    memories = []
    for match in results['matches']:
        if match['score'] > 0.15:
            memories.append({
                "question": match['metadata'].get('question', ''),
                "answer": match['metadata'].get('answer', ''),
                "timestamp": match['metadata'].get('timestamp', '')
            })
    
    return memories

def save_to_memory(question: str, answer: str):
    """Save conversation to long-term memory"""
    
    memory_text = f"Question: {question}\nAnswer: {answer}"
    vector = embeddings.embed_query(memory_text)
    
    memory_id = f"memory_{USER_ID}_{uuid.uuid4().hex[:8]}"
    
    index.upsert(vectors=[{
        "id": memory_id,
        "values": vector,
        "metadata": {
            "type": "memory",
            "user_id": USER_ID,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "text": memory_text
        }
    }])
    
    print("💾 Saved to memory")

def ask(question: str):
    """Ask the agent a question with memory"""
    
    print(f"🤔 Question: {question}\n")
    
    # Get document context
    doc_context = get_document_context(question)
    
    # Get memory context
    memories = get_memory_context(question)
    
    # Format memories
    memory_text = ""
    if memories:
        print(f"🧠 Found {len(memories)} relevant memories:")
        for m in memories:
            print(f"   📌 {m['question'][:50]}...")
        memory_text = "\nRelevant past conversations:\n"
        for m in memories:
            memory_text += f"- User said: {m['question']}\n"
            memory_text += f"  Assistant responded: {m['answer'][:300]}\n\n"
    
    # Create prompt
    prompt = f"""You are a helpful assistant with memory of past conversations.

Document Context:
{doc_context}

{memory_text}

Current Question: {question}

Instructions:
- Answer based on the document context and your memory of past conversations
- If you remember something relevant from past conversations, USE IT to answer
- If the user asked about their name, job, or any personal info, check the memory first
- Be conversational and reference past interactions naturally

Answer:"""
    
    # Get response
    response = llm.invoke(prompt)
    answer = response.content
    
    print(f"🤖 Answer: {answer}\n")
    
    # Save to memory
    save_to_memory(question, answer)
    
    return answer

def chat():
    """Interactive chat mode with memory"""
    print("=" * 50)
    print("🤖 RAG Agent with Long-term Memory")
    print("Type 'quit' to exit")
    print("Type 'clear memory' to clear your memory")
    print("=" * 50)
    
    while True:
        question = input("\n📝 You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question.lower() == 'clear memory':
            print("🗑️ Memory clearing not implemented yet")
            continue
        
        if not question:
            continue
        
        print()
        ask(question)

if __name__ == "__main__":
    chat()