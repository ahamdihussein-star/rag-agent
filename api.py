from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from pinecone import Pinecone
from dotenv import load_dotenv
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import requests
import os
import uuid
import tempfile
import re
import json
import cohere
import numpy as np
from typing import List, Optional
import asyncio

# Load environment variables
load_dotenv()

# Initialize
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Initialize Cohere for Reranking
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Semantic Chunker - splits by meaning
semantic_chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)

# Fallback character-based splitter
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Folders
UPLOADS_FOLDER = "uploads"
CONVERSATIONS_FOLDER = "conversations"
DOCUMENTS_FOLDER = "documents"
BM25_INDEX_FILE = "bm25_index.json"

os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# FastAPI app
app = FastAPI(title="RAG Agent API", version="2.3.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    conversation_id: Optional[str] = None

class Source(BaseModel):
    type: str
    title: Optional[str] = None
    source: str
    score: float
    download_url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    memories_found: int
    sources: List[Source] = []
    conversation_id: str

class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks: int = 0

class UrlRequest(BaseModel):
    url: str

# ==================== Semantic Chunking ====================

def smart_chunk_text(text: str, min_chunk_size: int = 100) -> List[str]:
    if len(text) < min_chunk_size:
        return [text]
    
    try:
        chunks = semantic_chunker.split_text(text)
        
        merged_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < 1500:
                current_chunk += "\n\n" + chunk if current_chunk else chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk.strip())
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk.strip())
        
        if len(merged_chunks) < 2 and len(text) > 2000:
            return fallback_splitter.split_text(text)
        
        return merged_chunks
    
    except Exception as e:
        print(f"Semantic chunking failed: {e}, using fallback")
        return fallback_splitter.split_text(text)

# ==================== BM25 Index Management ====================

def load_bm25_index() -> dict:
    if os.path.exists(BM25_INDEX_FILE):
        with open(BM25_INDEX_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"documents": [], "metadata": []}

def save_bm25_index(bm25_data: dict):
    with open(BM25_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(bm25_data, f, ensure_ascii=False)

def add_to_bm25_index(chunks: List[str], metadata_list: List[dict]):
    bm25_data = load_bm25_index()
    
    for chunk, meta in zip(chunks, metadata_list):
        bm25_data["documents"].append(chunk)
        bm25_data["metadata"].append(meta)
    
    save_bm25_index(bm25_data)

def search_bm25(query: str, top_k: int = 10) -> List[dict]:
    bm25_data = load_bm25_index()
    
    if not bm25_data["documents"]:
        return []
    
    tokenized_docs = [doc.lower().split() for doc in bm25_data["documents"]]
    tokenized_query = query.lower().split()
    
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text": bm25_data["documents"][idx],
                "metadata": bm25_data["metadata"][idx],
                "score": float(scores[idx])
            })
    
    return results

# ==================== Document Storage ====================

def save_full_document(doc_id: str, content: str, metadata: dict):
    doc_path = os.path.join(DOCUMENTS_FOLDER, f"{doc_id}.json")
    doc_data = {
        "id": doc_id,
        "content": content,
        "metadata": metadata,
        "created_at": datetime.now().isoformat()
    }
    with open(doc_path, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)
    return doc_id

def load_full_document(doc_id: str) -> Optional[dict]:
    doc_path = os.path.join(DOCUMENTS_FOLDER, f"{doc_id}.json")
    if os.path.exists(doc_path):
        with open(doc_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_full_documents_by_ids(doc_ids: List[str]) -> List[dict]:
    documents = []
    seen_ids = set()
    for doc_id in doc_ids:
        if doc_id not in seen_ids:
            doc = load_full_document(doc_id)
            if doc:
                documents.append(doc)
                seen_ids.add(doc_id)
    return documents

# ==================== Conversation Functions ====================

def get_conversation_path(conversation_id: str) -> str:
    return os.path.join(CONVERSATIONS_FOLDER, f"{conversation_id}.json")

def load_conversation(conversation_id: str) -> Optional[dict]:
    path = get_conversation_path(conversation_id)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def save_conversation(conversation: dict):
    path = get_conversation_path(conversation['id'])
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)

def create_conversation(first_message: str) -> dict:
    conversation_id = uuid.uuid4().hex[:12]
    title = first_message[:50] + "..." if len(first_message) > 50 else first_message
    
    conversation = {
        "id": conversation_id,
        "title": title,
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    save_conversation(conversation)
    return conversation

def add_message_to_conversation(conversation_id: str, role: str, content: str, sources: list = []):
    conversation = load_conversation(conversation_id)
    if conversation:
        conversation['messages'].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "sources": sources
        })
        conversation['updated_at'] = datetime.now().isoformat()
        save_conversation(conversation)

def list_all_conversations() -> List[dict]:
    conversations = []
    for filename in os.listdir(CONVERSATIONS_FOLDER):
        if filename.endswith('.json'):
            path = os.path.join(CONVERSATIONS_FOLDER, filename)
            with open(path, 'r') as f:
                conv = json.load(f)
                conversations.append({
                    "id": conv['id'],
                    "title": conv['title'],
                    "created_at": conv['created_at'],
                    "updated_at": conv['updated_at'],
                    "message_count": len(conv['messages'])
                })
    conversations.sort(key=lambda x: x['updated_at'], reverse=True)
    return conversations

# ==================== Retrieval Functions ====================

def rerank_results(query: str, documents: List[dict], top_n: int = 5) -> List[dict]:
    if not documents:
        return []
    
    try:
        texts = [doc.get('text', doc.get('metadata', {}).get('text', '')) for doc in documents]
        texts = [t for t in texts if t]
        
        if not texts:
            return documents[:top_n]
        
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=texts,
            top_n=min(top_n, len(texts))
        )
        
        reranked = []
        for result in response.results:
            original_doc = documents[result.index]
            reranked.append({
                'text': texts[result.index],
                'metadata': original_doc.get('metadata', original_doc),
                'score': result.relevance_score
            })
        
        return reranked
    
    except Exception as e:
        print(f"Reranking error: {e}")
        return documents[:top_n]

def hybrid_search(query: str, top_k: int = 10, source_filter: dict = None) -> List[dict]:
    """Search in ALL sources (documents, websites, youtube, memory)"""
    query_vector = embeddings.embed_query(query)
    
    # دور في كل المصادر (بما فيهم memory)
    filter_dict = source_filter if source_filter else {}
    
    # Semantic search in Pinecone
    if filter_dict:
        semantic_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
    else:
        semantic_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
    
    # BM25 keyword search
    bm25_results = search_bm25(query, top_k=top_k)
    
    # Combine results
    combined = {}
    
    for match in semantic_results['matches']:
        doc_type = match['metadata'].get('type', '')
        
        # Handle memory type differently (no parent_id)
        if doc_type == 'memory':
            memory_id = match['id']
            combined[memory_id] = {
                'text': match['metadata'].get('text', ''),
                'metadata': match['metadata'],
                'score': match['score'],
                'source': 'semantic',
                'is_memory': True
            }
        else:
            parent_id = match['metadata'].get('parent_id')
            if parent_id:
                if parent_id not in combined or match['score'] > combined[parent_id]['score']:
                    combined[parent_id] = {
                        'text': match['metadata'].get('text', ''),
                        'metadata': match['metadata'],
                        'score': match['score'],
                        'source': 'semantic',
                        'is_memory': False
                    }
    
    # Add BM25 results
    for result in bm25_results:
        parent_id = result['metadata'].get('parent_id')
        if parent_id:
            bm25_score = min(result['score'] / 10, 1.0)
            
            if parent_id in combined:
                combined[parent_id]['score'] = min(combined[parent_id]['score'] + 0.1, 1.0)
                combined[parent_id]['source'] = 'hybrid'
            else:
                combined[parent_id] = {
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'score': bm25_score,
                    'source': 'bm25',
                    'is_memory': False
                }
    
    sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
    return sorted_results[:top_k]

def get_document_context_with_sources(query: str, top_k: int = 10, display_min_score: float = 0.30):
    """Get context from ALL sources including memory"""
    
    # دور في كل المصادر (بدون filter)
    search_results = hybrid_search(query, top_k=top_k, source_filter=None)
    reranked_docs = rerank_results(query, search_results, top_n=5)
    
    parent_ids = []
    sources_info = []
    memory_contexts = []
    
    for doc in reranked_docs:
        metadata = doc.get('metadata', {})
        score = doc.get('score', 0)
        doc_type = metadata.get('type', 'document')
        
        # Handle Memory
        if doc_type == 'memory':
            if score >= display_min_score:
                question = metadata.get('question', '')
                answer = metadata.get('answer', '')
                memory_contexts.append({
                    'question': question,
                    'answer': answer,
                    'score': score
                })
                sources_info.append({
                    "type": "memory",
                    "title": f"Previous Q&A: {question[:50]}...",
                    "source": "Memory",
                    "score": round(score, 2),
                    "download_url": None
                })
        else:
            # Handle Documents/Websites/YouTube
            parent_id = metadata.get('parent_id')
            
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)
                
                if score >= display_min_score:
                    source_url = metadata.get('source', 'Unknown')
                    source_title = metadata.get('title', metadata.get('filename', source_url))
                    file_path = metadata.get('file_path', '')
                    
                    download_url = None
                    if file_path and os.path.exists(file_path):
                        filename = os.path.basename(file_path)
                        download_url = f"/download/{filename}"
                    
                    sources_info.append({
                        "type": doc_type,
                        "title": source_title,
                        "source": source_url,
                        "score": round(score, 2),
                        "download_url": download_url
                    })
    
    # Build context from full documents
    full_documents = get_full_documents_by_ids(parent_ids)
    
    context = ""
    
    # Add Documents Context
    for doc in full_documents:
        context += f"=== {doc['metadata'].get('title', 'Document')} ===\n"
        context += doc['content'] + "\n\n"
    
    # Add Memory Context
    if memory_contexts:
        context += "=== Previous Conversations (Memory) ===\n"
        for mem in memory_contexts:
            context += f"Q: {mem['question']}\n"
            context += f"A: {mem['answer']}\n\n"
    
    # Remove duplicate sources
    seen_sources = set()
    unique_sources = []
    for source in sources_info:
        source_key = f"{source['type']}_{source['source']}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            unique_sources.append(source)
    
    return context, unique_sources[:5]

def save_to_memory(question: str, answer: str, user_id: str):
    memory_text = f"Question: {question}\nAnswer: {answer}"
    vector = embeddings.embed_query(memory_text)
    memory_id = f"memory_{user_id}_{uuid.uuid4().hex[:8]}"
    index.upsert(vectors=[{
        "id": memory_id,
        "values": vector,
        "metadata": {
            "type": "memory",
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "text": memory_text
        }
    }])

def extract_metadata(text: str, source_type: str):
    prompt = f"""Analyze the following {source_type} content and extract metadata.

Content:
{text[:3000]}

Respond in JSON format only:
{{
    "title": "A descriptive title (max 10 words)",
    "summary": "Brief summary (2-3 sentences)",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "language": "detected language"
}}

JSON only:"""

    try:
        response = llm.invoke(prompt)
        json_str = response.content.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        return json.loads(json_str.strip())
    except:
        return {"title": "Unknown", "summary": "", "keywords": [], "language": "unknown"}

# ==================== Ingestion with Semantic Chunking ====================

def ingest_document_with_semantic_chunks(full_text: str, source: str, doc_type: str, file_path: str, metadata: dict):
    parent_id = f"parent_{uuid.uuid4().hex[:12]}"
    
    save_full_document(parent_id, full_text, {
        "source": source,
        "type": doc_type,
        "file_path": file_path,
        "title": metadata.get("title", source),
        "summary": metadata.get("summary", "")
    })
    
    chunks = smart_chunk_text(full_text)
    
    print(f"📦 Created {len(chunks)} semantic chunks for {source}")
    
    metadata_list = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            "text": chunk,
            "source": source,
            "type": doc_type,
            "filename": source,
            "file_path": file_path,
            "title": metadata.get("title", source),
            "summary": metadata.get("summary", ""),
            "parent_id": parent_id,
            "chunk_index": i,
            "chunk_type": "semantic"
        }
        
        vector = embeddings.embed_query(chunk)
        index.upsert(vectors=[{
            "id": f"{parent_id}_chunk_{i}",
            "values": vector,
            "metadata": chunk_metadata
        }])
        
        metadata_list.append(chunk_metadata)
    
    add_to_bm25_index(chunks, metadata_list)
    
    return len(chunks)

# ==================== API Endpoints ====================

@app.get("/")
def root():
    return {"message": "🤖 RAG Agent API v2.3 - Search All Sources Including Memory"}

@app.get("/frontend")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/admin")
def serve_admin():
    return FileResponse("frontend/admin.html")

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(UPLOADS_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')
    return {"error": "File not found"}

# ==================== Conversation Endpoints ====================

@app.get("/conversations")
def get_conversations():
    return {"conversations": list_all_conversations()}

@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conversation = load_conversation(conversation_id)
    if conversation:
        return conversation
    return {"error": "Conversation not found"}

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    path = get_conversation_path(conversation_id)
    if os.path.exists(path):
        os.remove(path)
        return {"success": True, "message": "Conversation deleted"}
    return {"success": False, "message": "Conversation not found"}

# ==================== Chat Endpoint ====================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.message
    user_id = request.user_id
    conversation_id = request.conversation_id
    
    if not conversation_id:
        conversation = create_conversation(question)
        conversation_id = conversation['id']
    
    add_message_to_conversation(conversation_id, "user", question)
    
    conv = load_conversation(conversation_id)
    conversation_history = ""
    if conv and len(conv['messages']) > 1:
        recent_messages = conv['messages'][-7:-1]
        if recent_messages:
            conversation_history = "Recent Conversation:\n"
            for msg in recent_messages:
                role = "User" if msg['role'] == 'user' else "Assistant"
                content = msg['content'][:1500] + "..." if len(msg['content']) > 1500 else msg['content']
                conversation_history += f"{role}: {content}\n\n"
    
    search_query = question
    if conv and len(conv['messages']) > 2:
        user_messages = [m['content'] for m in conv['messages'] if m['role'] == 'user'][-3:]
        search_query = " ".join(user_messages)
    
    doc_context, sources = get_document_context_with_sources(search_query)
    
    prompt = f"""You are a friendly and helpful AI assistant.

{conversation_history}

Available Information (Documents + Memory):
{doc_context if doc_context.strip() else "No specific information found."}

User Message: {question}

Instructions:
- Be friendly and conversational
- If greeting, respond with a friendly greeting
- Use conversation history to understand context
- ONLY use information from above - do NOT make up information
- If no relevant info, say "I don't have specific information about that."
- Format using markdown (headers, bullet points, bold)
- NEVER mention "documents", "context", "memory" in your response
- NEVER list sources in text - shown separately
- Be thorough - include ALL relevant details
- When asked to list ALL items, include EVERY item found

Response:"""
    
    response = llm.invoke(prompt)
    answer = response.content
    
    add_message_to_conversation(conversation_id, "assistant", answer, sources)
    
    if doc_context.strip():
        save_to_memory(question, answer, user_id)
    
    source_objects = [Source(**s) for s in sources]
    
    return ChatResponse(
        answer=answer,
        memories_found=len([s for s in sources if s['type'] == 'memory']),
        sources=source_objects,
        conversation_id=conversation_id
    )

# ==================== Upload Endpoints ====================

@app.post("/upload", response_model=IngestResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        
        file_path = os.path.join(UPLOADS_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])
            metadata = extract_metadata(full_text, "document")
            chunks = ingest_document_with_semantic_chunks(full_text, file.filename, "document", file_path, metadata)
            os.unlink(tmp_path)
            return IngestResponse(success=True, message=f"✅ Ingested {file.filename} with {chunks} semantic chunks", chunks=chunks)
        
        elif suffix == ".txt":
            loader = TextLoader(tmp_path)
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])
            metadata = extract_metadata(full_text, "document")
            chunks = ingest_document_with_semantic_chunks(full_text, file.filename, "document", file_path, metadata)
            os.unlink(tmp_path)
            return IngestResponse(success=True, message=f"✅ Ingested {file.filename} with {chunks} semantic chunks", chunks=chunks)
        
        elif suffix == ".docx":
            from docx import Document as DocxDocument
            doc = DocxDocument(tmp_path)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
            metadata = extract_metadata(full_text, "document")
            chunks = ingest_document_with_semantic_chunks(full_text, file.filename, "document", file_path, metadata)
            os.unlink(tmp_path)
            return IngestResponse(success=True, message=f"✅ Ingested {file.filename} with {chunks} semantic chunks", chunks=chunks)
        
        elif suffix in [".xlsx", ".xls"]:
            from openpyxl import load_workbook
            wb = load_workbook(tmp_path)
            full_text = ""
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                full_text += f"\n--- Sheet: {sheet} ---\n"
                for row in ws.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) if cell else "" for cell in row])
                    if row_text.strip():
                        full_text += row_text + "\n"
            metadata = extract_metadata(full_text, "spreadsheet")
            chunks = ingest_document_with_semantic_chunks(full_text, file.filename, "spreadsheet", file_path, metadata)
            os.unlink(tmp_path)
            return IngestResponse(success=True, message=f"✅ Ingested {file.filename} with {chunks} semantic chunks", chunks=chunks)
        
        elif suffix == ".pptx":
            from pptx import Presentation
            prs = Presentation(tmp_path)
            full_text = ""
            for slide_num, slide in enumerate(prs.slides, 1):
                full_text += f"\n--- Slide {slide_num} ---\n"
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        full_text += shape.text + "\n"
            metadata = extract_metadata(full_text, "presentation")
            chunks = ingest_document_with_semantic_chunks(full_text, file.filename, "presentation", file_path, metadata)
            os.unlink(tmp_path)
            return IngestResponse(success=True, message=f"✅ Ingested {file.filename} with {chunks} semantic chunks", chunks=chunks)
        
        elif suffix in [".mp4", ".mp3", ".wav", ".m4a", ".webm", ".avi", ".mov"]:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            full_text = result["text"]
            metadata = extract_metadata(full_text, "video")
            chunks = ingest_document_with_semantic_chunks(full_text, file.filename, "video", file_path, metadata)
            os.unlink(tmp_path)
            return IngestResponse(success=True, message=f"✅ Transcribed {file.filename} with {chunks} semantic chunks", chunks=chunks)
        
        elif suffix in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            import base64
            from langchain_core.messages import HumanMessage
            
            with open(tmp_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
            media_type = media_types.get(suffix, "image/jpeg")
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Extract ALL text from this image. Return only the extracted text."},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                ],
            )
            
            response = llm.invoke([message])
            full_text = response.content
            metadata = extract_metadata(full_text, "image")
            chunks = ingest_document_with_semantic_chunks(full_text, file.filename, "image", file_path, metadata)
            os.unlink(tmp_path)
            return IngestResponse(success=True, message=f"✅ Processed {file.filename} with {chunks} semantic chunks", chunks=chunks)
        
        else:
            os.unlink(tmp_path)
            os.remove(file_path)
            return IngestResponse(success=False, message=f"Unsupported file type: {suffix}")
    
    except Exception as e:
        return IngestResponse(success=False, message=str(e))

@app.post("/scrape", response_model=IngestResponse)
def scrape_website(request: UrlRequest):
    try:
        url = request.url
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        full_text = '\n'.join(lines)
        
        if not full_text:
            return IngestResponse(success=False, message="No content found")
        
        metadata = extract_metadata(full_text, "website")
        
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        chunks = ingest_document_with_semantic_chunks(full_text, url, "website", "", metadata)
        
        return IngestResponse(success=True, message=f"✅ Scraped {domain} with {chunks} semantic chunks", chunks=chunks)
    
    except Exception as e:
        return IngestResponse(success=False, message=str(e))

@app.post("/youtube", response_model=IngestResponse)
def process_youtube(request: UrlRequest):
    try:
        url = request.url
        patterns = [r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})']
        video_id = None
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break
        
        if not video_id:
            return IngestResponse(success=False, message="Invalid YouTube URL")
        
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id)
        full_text = ' '.join([entry.text for entry in transcript_data])
        
        if not full_text:
            return IngestResponse(success=False, message="No transcript available")
        
        metadata = extract_metadata(full_text, "youtube video")
        chunks = ingest_document_with_semantic_chunks(full_text, url, "youtube", "", metadata)
        
        return IngestResponse(success=True, message=f"✅ Processed YouTube with {chunks} semantic chunks", chunks=chunks)
    
    except Exception as e:
        return IngestResponse(success=False, message=str(e))

@app.get("/stats")
def get_stats():
    stats = index.describe_index_stats()
    doc_count = len([f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith('.json')])
    bm25_data = load_bm25_index()
    bm25_count = len(bm25_data.get("documents", []))
    
    return {
        "total_vectors": stats.total_vector_count,
        "total_documents": doc_count,
        "bm25_index_size": bm25_count,
        "index_name": os.getenv("PINECONE_INDEX_NAME"),
        "chunking_type": "Semantic",
        "search_type": "Hybrid (All Sources + Memory)"
    }

@app.get("/files")
def list_files():
    files = []
    for filename in os.listdir(UPLOADS_FOLDER):
        file_path = os.path.join(UPLOADS_FOLDER, filename)
        files.append({
            "filename": filename,
            "size": os.path.getsize(file_path),
            "download_url": f"/download/{filename}"
        })
    return {"files": files}

# ==================== Admin Dashboard Endpoints ====================

@app.get("/admin/documents")
def get_all_documents():
    documents = []
    
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith('.json'):
            doc_path = os.path.join(DOCUMENTS_FOLDER, filename)
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append({
                    "id": doc['id'],
                    "title": doc['metadata'].get('title', 'Unknown'),
                    "source": doc['metadata'].get('source', 'Unknown'),
                    "type": doc['metadata'].get('type', 'document'),
                    "file_path": doc['metadata'].get('file_path', ''),
                    "summary": doc['metadata'].get('summary', ''),
                    "created_at": doc.get('created_at', ''),
                    "content_preview": doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                })
    
    documents.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return {"documents": documents, "total": len(documents)}

@app.get("/admin/documents/{doc_id}")
def get_document_detail(doc_id: str):
    doc = load_full_document(doc_id)
    
    if not doc:
        return {"error": "Document not found"}
    
    chunks = []
    try:
        dummy_vector = embeddings.embed_query("test")
        results = index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True,
            filter={"parent_id": {"$eq": doc_id}}
        )
        
        for match in results['matches']:
            chunks.append({
                "id": match['id'],
                "text": match['metadata'].get('text', ''),
                "chunk_index": match['metadata'].get('chunk_index', 0),
                "chunk_type": match['metadata'].get('chunk_type', 'unknown')
            })
        
        chunks.sort(key=lambda x: x.get('chunk_index', 0))
    except Exception as e:
        print(f"Error fetching chunks: {e}")
    
    return {
        "document": doc,
        "chunks": chunks,
        "chunk_count": len(chunks)
    }

@app.delete("/admin/documents/{doc_id}")
def delete_document(doc_id: str):
    doc_path = os.path.join(DOCUMENTS_FOLDER, f"{doc_id}.json")
    if os.path.exists(doc_path):
        with open(doc_path, 'r') as f:
            doc = json.load(f)
        
        file_path = doc['metadata'].get('file_path', '')
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        os.remove(doc_path)
    
    try:
        dummy_vector = embeddings.embed_query("test")
        results = index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True,
            filter={"parent_id": {"$eq": doc_id}}
        )
        
        chunk_ids = [match['id'] for match in results['matches']]
        if chunk_ids:
            index.delete(ids=chunk_ids)
    except Exception as e:
        print(f"Error deleting from Pinecone: {e}")
    
    try:
        bm25_data = load_bm25_index()
        new_docs = []
        new_meta = []
        
        for doc_text, meta in zip(bm25_data['documents'], bm25_data['metadata']):
            if meta.get('parent_id') != doc_id:
                new_docs.append(doc_text)
                new_meta.append(meta)
        
        bm25_data['documents'] = new_docs
        bm25_data['metadata'] = new_meta
        save_bm25_index(bm25_data)
    except Exception as e:
        print(f"Error updating BM25: {e}")
    
    return {"success": True, "message": f"Deleted document {doc_id}"}

@app.get("/admin/memories")
def get_all_memories():
    memories = []
    
    try:
        dummy_vector = embeddings.embed_query("memory")
        results = index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True,
            filter={"type": {"$eq": "memory"}}
        )
        
        for match in results['matches']:
            memories.append({
                "id": match['id'],
                "question": match['metadata'].get('question', ''),
                "answer": match['metadata'].get('answer', '')[:200] + "...",
                "user_id": match['metadata'].get('user_id', ''),
                "timestamp": match['metadata'].get('timestamp', '')
            })
        
        memories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    except Exception as e:
        print(f"Error fetching memories: {e}")
    
    return {"memories": memories, "total": len(memories)}

@app.delete("/admin/memories/{memory_id}")
def delete_memory(memory_id: str):
    try:
        index.delete(ids=[memory_id])
        return {"success": True, "message": f"Deleted memory {memory_id}"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.delete("/admin/memories")
def clear_all_memories():
    try:
        dummy_vector = embeddings.embed_query("memory")
        results = index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True,
            filter={"type": {"$eq": "memory"}}
        )
        
        memory_ids = [match['id'] for match in results['matches']]
        if memory_ids:
            index.delete(ids=memory_ids)
        
        return {"success": True, "message": f"Deleted {len(memory_ids)} memories"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/admin/stats")
def get_admin_stats():
    stats = index.describe_index_stats()
    doc_count = len([f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith('.json')])
    bm25_data = load_bm25_index()
    
    type_counts = {"document": 0, "website": 0, "youtube": 0, "image": 0, "video": 0}
    for meta in bm25_data.get('metadata', []):
        doc_type = meta.get('type', 'document')
        if doc_type in type_counts:
            type_counts[doc_type] += 1
    
    return {
        "total_vectors": stats.total_vector_count,
        "total_documents": doc_count,
        "bm25_chunks": len(bm25_data.get('documents', [])),
        "type_breakdown": type_counts,
        "index_name": os.getenv("PINECONE_INDEX_NAME")
    }

# ==================== Streaming Chat Endpoint ====================

async def generate_stream(question: str, user_id: str, conversation_id: str):
    """Generate streaming response with proper async"""
    
    # Create or load conversation
    if not conversation_id:
        conversation = create_conversation(question)
        conversation_id = conversation['id']
    
    # Add user message
    add_message_to_conversation(conversation_id, "user", question)
    
    # Get conversation history
    conv = load_conversation(conversation_id)
    conversation_history = ""
    if conv and len(conv['messages']) > 1:
        recent_messages = conv['messages'][-7:-1]
        if recent_messages:
            conversation_history = "Recent Conversation:\n"
            for msg in recent_messages:
                role = "User" if msg['role'] == 'user' else "Assistant"
                content = msg['content'][:1500] + "..." if len(msg['content']) > 1500 else msg['content']
                conversation_history += f"{role}: {content}\n\n"
    
    # Build search query
    search_query = question
    if conv and len(conv['messages']) > 2:
        user_messages = [m['content'] for m in conv['messages'] if m['role'] == 'user'][-3:]
        search_query = " ".join(user_messages)
    
    # Get document context (من كل المصادر بما فيهم memory)
    doc_context, sources = get_document_context_with_sources(search_query)
    
    # Send sources first
    sources_json = json.dumps({"type": "sources", "data": sources, "conversation_id": conversation_id})
    yield f"data: {sources_json}\n\n"
    
    # Build prompt
    prompt = f"""You are a friendly and helpful AI assistant.

{conversation_history}

Available Information (Documents + Memory):
{doc_context if doc_context.strip() else "No specific information found."}

User Message: {question}

Instructions:
- Be friendly and conversational
- If greeting, respond with a friendly greeting
- Use conversation history to understand context
- ONLY use information from above - do NOT make up information
- If no relevant info, say "I don't have specific information about that."
- Format using markdown (headers, bullet points, bold)
- NEVER mention "documents", "context", "memory" in your response
- NEVER list sources in text - shown separately
- Be thorough - include ALL relevant details
- When asked to list ALL items, include EVERY item found

Response:"""
    
    # Stream the response
    full_response = ""
    
    try:
        # Use streaming LLM with async - استخدم astream بدل stream
        streaming_llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)
        
        async for chunk in streaming_llm.astream(prompt):
            if chunk.content:
                full_response += chunk.content
                chunk_json = json.dumps({"type": "chunk", "data": chunk.content})
                yield f"data: {chunk_json}\n\n"
        
        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        # Save to conversation and memory
        add_message_to_conversation(conversation_id, "assistant", full_response, sources)
        
        if doc_context.strip():
            save_to_memory(question, full_response, user_id)
            
    except Exception as e:
        error_json = json.dumps({"type": "error", "data": str(e)})
        yield f"data: {error_json}\n\n"

@app.get("/chat/stream")
async def chat_stream(message: str, user_id: str = "default_user", conversation_id: str = None):
    """Streaming chat endpoint using Server-Sent Events"""
    return StreamingResponse(
        generate_stream(message, user_id, conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ==================== Debug Endpoint ====================

@app.get("/debug/search")
def debug_search(query: str):
    """Debug search endpoint to see what sources are found"""
    search_results = hybrid_search(query, top_k=10, source_filter=None)
    reranked = rerank_results(query, search_results, top_n=5)
    
    return {
        "query": query,
        "search_results_count": len(search_results),
        "search_results": [
            {
                "type": r.get('metadata', {}).get('type'),
                "title": r.get('metadata', {}).get('title'),
                "parent_id": r.get('metadata', {}).get('parent_id'),
                "score": r.get('score'),
                "is_memory": r.get('is_memory', False)
            }
            for r in search_results[:10]
        ],
        "reranked": [
            {
                "type": r.get('metadata', {}).get('type'),
                "title": r.get('metadata', {}).get('title'),
                "score": r.get('score')
            }
            for r in reranked
        ]
    }

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)