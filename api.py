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

# Folders - Use Railway Volume if available, otherwise local
DATA_DIR = "/app/data" if os.path.exists("/app/data") else "."
UPLOADS_FOLDER = os.path.join(DATA_DIR, "uploads")
CONVERSATIONS_FOLDER = os.path.join(DATA_DIR, "conversations")
DOCUMENTS_FOLDER = os.path.join(DATA_DIR, "documents")
BM25_INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.json")

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

class ImageInfo(BaseModel):
    url: str
    alt: Optional[str] = None
    title: Optional[str] = None
    context: Optional[str] = None
    doc_title: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    memories_found: int
    sources: List[Source] = []
    images: List[ImageInfo] = []
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

def save_full_document(doc_id: str, content: str, metadata: dict, images: List[dict] = None):
    doc_path = os.path.join(DOCUMENTS_FOLDER, f"{doc_id}.json")
    doc_data = {
        "id": doc_id,
        "content": content,
        "metadata": metadata,
        "images": images or [],
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
    """Get context from ALL sources including memory and images"""
    
    # دور في كل المصادر (بدون filter)
    search_results = hybrid_search(query, top_k=top_k, source_filter=None)
    reranked_docs = rerank_results(query, search_results, top_n=5)
    
    print(f"🔍 Search results count: {len(search_results)}")
    print(f"🔍 Reranked docs count: {len(reranked_docs)}")
    
    parent_ids = []
    sources_info = []
    memory_contexts = []
    chunk_texts = {}  # Store chunk texts as fallback
    all_images = []  # Store images from relevant documents
    
    for doc in reranked_docs:
        metadata = doc.get('metadata', {})
        score = doc.get('score', 0)
        doc_type = metadata.get('type', 'document')
        chunk_text = doc.get('text', '') or metadata.get('text', '')  # Try both places
        
        print(f"📄 Doc type: {doc_type}, Score: {score}, Has text: {bool(chunk_text)}, Text length: {len(chunk_text) if chunk_text else 0}")
        
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
                
                # Store chunk text as fallback
                if chunk_text:
                    if parent_id not in chunk_texts:
                        chunk_texts[parent_id] = {
                            'title': metadata.get('title', metadata.get('filename', 'Document')),
                            'texts': []
                        }
                    chunk_texts[parent_id]['texts'].append(chunk_text)
                
                if score >= display_min_score:
                    source_url = metadata.get('source', 'Unknown')
                    source_title = metadata.get('title', metadata.get('filename', source_url))
                    domain = metadata.get('domain', '')
                    file_path = metadata.get('file_path', '')
                    
                    # Add domain to title for clarity
                    if domain and domain.lower() not in source_title.lower():
                        source_title = f"{domain} - {source_title}"
                    
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
            elif parent_id and chunk_text:
                # Add more chunks for existing parent
                if parent_id in chunk_texts:
                    chunk_texts[parent_id]['texts'].append(chunk_text)
    
    print(f"📚 Parent IDs found: {len(parent_ids)}")
    print(f"📚 Chunk texts collected: {len(chunk_texts)}")
    
    # Build context from full documents OR chunk texts
    full_documents = get_full_documents_by_ids(parent_ids)
    
    print(f"📚 Full documents loaded: {len(full_documents)}")
    
    context = ""
    used_parent_ids = set()
    
    # Add Documents Context (from full documents)
    for doc in full_documents:
        parent_id = doc.get('id', '')
        used_parent_ids.add(parent_id)
        context += f"=== {doc['metadata'].get('title', 'Document')} ===\n"
        context += doc['content'] + "\n\n"
        
        # Collect images from document
        doc_images = doc.get('images', [])
        if doc_images:
            for img in doc_images:
                img['doc_title'] = doc['metadata'].get('title', 'Document')
                all_images.append(img)
    
    # Fallback: Add chunk texts for documents not found locally
    for parent_id, chunk_data in chunk_texts.items():
        if parent_id not in used_parent_ids:
            print(f"⚠️ Using fallback chunks for: {chunk_data['title']}")
            context += f"=== {chunk_data['title']} ===\n"
            # Join unique chunks
            unique_texts = list(set(chunk_data['texts']))
            context += "\n".join(unique_texts[:3]) + "\n\n"  # Limit to 3 chunks
    
    # Add Memory Context
    if memory_contexts:
        context += "=== Previous Conversations (Memory) ===\n"
        for mem in memory_contexts:
            context += f"Q: {mem['question']}\n"
            context += f"A: {mem['answer']}\n\n"
    
    print(f"📝 Final context length: {len(context)}")
    print(f"🖼️ Images found: {len(all_images)}")
    
    # Remove duplicate sources
    seen_sources = set()
    unique_sources = []
    for source in sources_info:
        source_key = f"{source['type']}_{source['source']}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            unique_sources.append(source)
    
    return context, unique_sources[:5], all_images[:10]  # Limit to 10 images

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

def extract_domain_name(url: str) -> str:
    """Extract readable domain name from URL (e.g., docs.oracle.com -> Oracle)"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Extract main company name
        domain_parts = domain.replace('www.', '').split('.')
        
        # Common patterns
        company_mappings = {
            'oracle': 'Oracle',
            'boomi': 'Boomi',
            'opentext': 'OpenText',
            'microsoft': 'Microsoft',
            'google': 'Google',
            'aws': 'AWS',
            'amazon': 'Amazon',
            'salesforce': 'Salesforce',
            'ibm': 'IBM',
            'sap': 'SAP',
        }
        
        for part in domain_parts:
            for key, value in company_mappings.items():
                if key in part:
                    return value
        
        # Default: capitalize first meaningful part
        for part in domain_parts:
            if part not in ['docs', 'www', 'help', 'support', 'com', 'org', 'net', 'io']:
                return part.capitalize()
        
        return ""
    except:
        return ""

def ingest_document_with_semantic_chunks(full_text: str, source: str, doc_type: str, file_path: str, metadata: dict, images: List[dict] = None):
    parent_id = f"parent_{uuid.uuid4().hex[:12]}"
    
    # Extract domain/company name for better search
    domain_name = extract_domain_name(source) if doc_type == "website" else ""
    
    save_full_document(parent_id, full_text, {
        "source": source,
        "type": doc_type,
        "file_path": file_path,
        "title": metadata.get("title", source),
        "summary": metadata.get("summary", ""),
        "domain": domain_name
    }, images=images)
    
    chunks = smart_chunk_text(full_text)
    
    print(f"📦 Created {len(chunks)} semantic chunks for {source} (domain: {domain_name}, images: {len(images) if images else 0})")
    
    metadata_list = []
    for i, chunk in enumerate(chunks):
        # Add domain name to chunk text for better semantic search
        enhanced_text = f"{domain_name} - {chunk}" if domain_name else chunk
        
        chunk_metadata = {
            "text": enhanced_text,
            "source": source,
            "type": doc_type,
            "filename": source,
            "file_path": file_path,
            "title": metadata.get("title", source),
            "summary": metadata.get("summary", ""),
            "parent_id": parent_id,
            "chunk_index": i,
            "chunk_type": "semantic",
            "domain": domain_name
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
    return {"message": "🤖 RAG Agent API v2.4 - With Progress Tracking"}

# ==================== Progress Streaming Helpers ====================

def progress_event(step: str, percent: int, message: str, **extra):
    """Create a progress event for SSE"""
    data = {"type": "progress", "step": step, "percent": percent, "message": message, **extra}
    return f"data: {json.dumps(data)}\n\n"

def done_event(message: str, chunks: int = 0, **extra):
    """Create a done event for SSE"""
    data = {"type": "done", "percent": 100, "message": message, "chunks": chunks, **extra}
    return f"data: {json.dumps(data)}\n\n"

def error_event(message: str):
    """Create an error event for SSE"""
    data = {"type": "error", "message": message}
    return f"data: {json.dumps(data)}\n\n"

def ingest_with_progress(full_text: str, source: str, doc_type: str, file_path: str, metadata: dict, images: List[dict] = None):
    """Generator that yields progress updates during ingestion"""
    
    parent_id = f"parent_{uuid.uuid4().hex[:12]}"
    domain_name = extract_domain_name(source) if doc_type == "website" else ""
    
    yield progress_event("saving_doc", 40, "Saving document...")
    
    save_full_document(parent_id, full_text, {
        "source": source,
        "type": doc_type,
        "file_path": file_path,
        "title": metadata.get("title", source),
        "summary": metadata.get("summary", ""),
        "domain": domain_name
    }, images=images)
    
    yield progress_event("chunking", 45, "Creating semantic chunks...")
    
    chunks = smart_chunk_text(full_text)
    total_chunks = len(chunks)
    
    img_count = len(images) if images else 0
    yield progress_event("chunking", 50, f"Created {total_chunks} chunks, {img_count} images")
    
    metadata_list = []
    for i, chunk in enumerate(chunks):
        # Calculate progress (50% to 95% for embeddings)
        embed_progress = 50 + int((i / total_chunks) * 45)
        yield progress_event("embedding", embed_progress, f"Embedding chunk {i+1}/{total_chunks}...")
        
        enhanced_text = f"{domain_name} - {chunk}" if domain_name else chunk
        
        chunk_metadata = {
            "text": enhanced_text,
            "source": source,
            "type": doc_type,
            "filename": source,
            "file_path": file_path,
            "title": metadata.get("title", source),
            "summary": metadata.get("summary", ""),
            "parent_id": parent_id,
            "chunk_index": i,
            "chunk_type": "semantic",
            "domain": domain_name
        }
        
        vector = embeddings.embed_query(chunk)
        index.upsert(vectors=[{
            "id": f"{parent_id}_chunk_{i}",
            "values": vector,
            "metadata": chunk_metadata
        }])
        
        metadata_list.append(chunk_metadata)
    
    yield progress_event("bm25", 97, "Adding to BM25 index...")
    add_to_bm25_index(chunks, metadata_list)
    
    yield done_event(f"✅ Ingested with {total_chunks} chunks, {img_count} images", total_chunks)

# ==================== Streaming Scrape Endpoint ====================

@app.get("/scrape/stream")
async def scrape_website_stream(url: str):
    """Scrape a website with progress streaming"""
    
    async def generate():
        try:
            yield progress_event("fetching", 10, f"Fetching {url}...")
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            html_content = response.text
            
            yield progress_event("parsing", 20, "Parsing HTML content...")
            
            # Extract images BEFORE removing elements
            images = extract_images_from_page(url, html_content)
            
            soup = BeautifulSoup(html_content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            full_text = '\n'.join(lines)
            
            if not full_text:
                yield error_event("No content found on page")
                return
            
            yield progress_event("images", 25, f"Found {len(images)} images")
            yield progress_event("metadata", 30, "Extracting metadata...")
            metadata = extract_metadata(full_text, "website")
            
            # Use the progress generator for ingestion with images
            for event in ingest_with_progress(full_text, url, "website", "", metadata, images=images):
                yield event
                
        except Exception as e:
            yield error_event(str(e))
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ==================== Streaming Recursive Scrape ====================

@app.get("/scrape-recursive/stream")
async def scrape_recursive_stream(url: str, max_depth: int = 2, max_pages: int = 20, same_path_only: bool = True):
    """Recursive scrape with progress streaming"""
    
    async def generate():
        try:
            visited = set()
            to_visit = [(url, 0)]  # (url, depth)
            pages_scraped = []
            total_chunks = 0
            total_images = 0
            
            yield progress_event("starting", 5, f"Starting recursive scrape from {url}")
            
            while to_visit and len(visited) < max_pages:
                current_url, depth = to_visit.pop(0)
                
                if current_url in visited:
                    continue
                    
                visited.add(current_url)
                page_num = len(visited)
                
                # Calculate overall progress based on pages
                overall_progress = min(5 + int((page_num / max_pages) * 90), 95)
                
                yield progress_event("scraping", overall_progress, 
                    f"Scraping page {page_num}/{max_pages}: {current_url[:50]}...",
                    current_page=page_num, total_pages=max_pages, current_url=current_url)
                
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(current_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    html_content = response.text
                    
                    # Extract links if we haven't reached max depth
                    if depth < max_depth:
                        new_links = extract_links_from_page(current_url, html_content, same_path_only)
                        for link in new_links:
                            if link not in visited and len(to_visit) + len(visited) < max_pages * 2:
                                to_visit.append((link, depth + 1))
                    
                    # Extract images BEFORE removing elements
                    images = extract_images_from_page(current_url, html_content)
                    total_images += len(images)
                    
                    # Parse content
                    soup = BeautifulSoup(html_content, 'html.parser')
                    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                        element.decompose()
                    
                    text = soup.get_text(separator='\n', strip=True)
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    full_text = '\n'.join(lines)
                    
                    if full_text and len(full_text) > 100:
                        metadata = extract_metadata(full_text[:3000], "website")
                        chunks = ingest_document_with_semantic_chunks(full_text, current_url, "website", "", metadata, images=images)
                        total_chunks += chunks
                        pages_scraped.append(current_url)
                        
                        yield progress_event("page_done", overall_progress,
                            f"✓ Page {page_num}: {chunks} chunks, {len(images)} images",
                            page_url=current_url, page_chunks=chunks, page_images=len(images))
                    
                except Exception as e:
                    yield progress_event("page_error", overall_progress,
                        f"✗ Failed: {current_url[:30]}... - {str(e)[:30]}")
                    continue
            
            yield done_event(
                f"✅ Scraped {len(pages_scraped)} pages with {total_chunks} chunks, {total_images} images",
                total_chunks,
                pages_scraped=pages_scraped,
                total_pages=len(pages_scraped),
                total_images=total_images
            )
            
        except Exception as e:
            yield error_event(str(e))
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ==================== Streaming YouTube Endpoint ====================

@app.get("/youtube/stream")
async def youtube_stream(url: str):
    """Add YouTube video with progress streaming"""
    
    async def generate():
        try:
            yield progress_event("fetching", 10, "Fetching YouTube transcript...")
            
            from youtube_transcript_api import YouTubeTranscriptApi
            import re
            
            video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
            if not video_id_match:
                yield error_event("Invalid YouTube URL")
                return
            
            video_id = video_id_match.group(1)
            
            yield progress_event("transcript", 30, "Downloading transcript...")
            
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            except Exception as e:
                yield error_event(f"Could not get transcript: {str(e)}")
                return
            
            yield progress_event("processing", 40, "Processing transcript...")
            
            full_text = " ".join([entry['text'] for entry in transcript_list])
            
            if not full_text:
                yield error_event("No transcript content found")
                return
            
            yield progress_event("metadata", 35, "Extracting metadata...")
            metadata = extract_metadata(full_text, "youtube")
            
            # Use the progress generator for ingestion
            for event in ingest_with_progress(full_text, url, "youtube", "", metadata):
                yield event
                
        except Exception as e:
            yield error_event(str(e))
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ==================== Debug Search Endpoint ====================

@app.get("/debug-search")
def debug_search(q: str):
    """Debug endpoint to see raw search results"""
    
    # 1. Raw Semantic Search from Pinecone
    query_vector = embeddings.embed_query(q)
    semantic_results = index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True
    )
    
    semantic_docs = []
    for match in semantic_results['matches']:
        semantic_docs.append({
            "id": match['id'],
            "score": round(match['score'], 4),
            "title": match['metadata'].get('title', 'N/A'),
            "type": match['metadata'].get('type', 'N/A'),
            "source": match['metadata'].get('source', 'N/A')[:50],
            "text_preview": match['metadata'].get('text', '')[:100] + "..."
        })
    
    # 2. BM25 Results
    bm25_results = search_bm25(q, top_k=10)
    bm25_docs = []
    for result in bm25_results:
        bm25_docs.append({
            "score": round(result['score'], 4),
            "title": result['metadata'].get('title', 'N/A'),
            "type": result['metadata'].get('type', 'N/A'),
        })
    
    # 3. Hybrid Results
    hybrid_results = hybrid_search(q, top_k=10)
    hybrid_docs = []
    for doc in hybrid_results:
        hybrid_docs.append({
            "score": round(doc['score'], 4),
            "title": doc['metadata'].get('title', 'N/A'),
            "type": doc['metadata'].get('type', 'N/A'),
            "source": doc['metadata'].get('source', 'N/A')[:50],
        })
    
    # 4. After Reranking
    reranked = rerank_results(q, hybrid_results, top_n=5)
    reranked_docs = []
    for doc in reranked:
        reranked_docs.append({
            "score": round(doc['score'], 4),
            "title": doc['metadata'].get('title', 'N/A'),
            "type": doc['metadata'].get('type', 'N/A'),
        })
    
    return {
        "query": q,
        "1_semantic_search": semantic_docs,
        "2_bm25_search": bm25_docs,
        "3_hybrid_combined": hybrid_docs,
        "4_after_reranking": reranked_docs,
        "display_threshold": 0.30
    }

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
    
    doc_context, sources, images = get_document_context_with_sources(search_query)
    
    # Add image info to prompt if images found
    image_info = ""
    if images:
        image_info = "\n\nRelevant Images (can be referenced in response):\n"
        for i, img in enumerate(images[:5], 1):
            image_info += f"- Image {i}: {img.get('alt', 'No description')} (from {img.get('doc_title', 'Unknown')})\n"
    
    prompt = f"""You are a friendly and helpful AI assistant.

{conversation_history}

Available Information (Documents + Memory):
{doc_context if doc_context.strip() else "No specific information found."}
{image_info}

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
- If relevant images are available and helpful for the answer, mention them

Response:"""
    
    response = llm.invoke(prompt)
    answer = response.content
    
    add_message_to_conversation(conversation_id, "assistant", answer, sources)
    
    if doc_context.strip():
        save_to_memory(question, answer, user_id)
    
    source_objects = [Source(**s) for s in sources]
    image_objects = [ImageInfo(**img) for img in images]
    
    return ChatResponse(
        answer=answer,
        memories_found=len([s for s in sources if s['type'] == 'memory']),
        sources=source_objects,
        images=image_objects,
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

# ==================== Recursive Web Scraper ====================

class RecursiveScrapeRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 20
    same_path_only: bool = True

class RecursiveScrapeResponse(BaseModel):
    success: bool
    message: str
    total_pages: int = 0
    total_chunks: int = 0
    pages_scraped: List[str] = []

def extract_links_from_page(url: str, html_content: str, same_path_only: bool = True) -> List[str]:
    """Extract valid links from a page that belong to the same context"""
    from urllib.parse import urlparse, urljoin, unquote
    
    soup = BeautifulSoup(html_content, 'html.parser')
    parsed_base = urlparse(url)
    base_domain = parsed_base.netloc
    
    # Get path segments (decode URL encoding)
    base_path_decoded = unquote(parsed_base.path)
    base_segments = [s for s in base_path_decoded.split('/') if s]
    
    # For same_path_only, match first N segments (more flexible)
    # e.g., /docs/Atomsphere/Integration/ -> match first 3 segments
    if len(base_segments) >= 3:
        match_segments = base_segments[:3]  # Match first 3 path segments
    elif len(base_segments) >= 2:
        match_segments = base_segments[:2]
    else:
        match_segments = base_segments[:1] if base_segments else []
    
    match_path = '/' + '/'.join(match_segments) if match_segments else ''
    
    links = set()
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        
        # Skip anchors, javascript, mailto, etc.
        if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:') or href.startswith('tel:'):
            continue
        
        # Convert relative URLs to absolute
        full_url = urljoin(url, href)
        parsed_url = urlparse(full_url)
        
        # Must be same domain
        if parsed_url.netloc != base_domain:
            continue
        
        # Skip non-http links
        if parsed_url.scheme not in ['http', 'https']:
            continue
        
        # Skip file downloads
        if any(parsed_url.path.lower().endswith(ext) for ext in ['.pdf', '.zip', '.doc', '.docx', '.xls', '.xlsx', '.png', '.jpg', '.jpeg', '.gif']):
            continue
        
        # If same_path_only, check if the link shares the same base path segments
        if same_path_only and match_path:
            link_path_decoded = unquote(parsed_url.path)
            if not link_path_decoded.startswith(match_path):
                continue
        
        # Clean URL (remove fragments and trailing slashes)
        clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        clean_url = clean_url.rstrip('/')
        
        links.add(clean_url)
    
    return list(links)

def extract_images_from_page(url: str, html_content: str, min_size: int = 100) -> List[dict]:
    """Extract meaningful images from a page (skip icons, logos, etc.)"""
    from urllib.parse import urlparse, urljoin
    
    soup = BeautifulSoup(html_content, 'html.parser')
    images = []
    
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if not src:
            continue
        
        # Convert relative URLs to absolute
        img_url = urljoin(url, src)
        
        # Skip data URIs and tiny images (likely icons)
        if img_url.startswith('data:'):
            continue
        
        # Skip common icon/logo patterns
        skip_patterns = ['icon', 'logo', 'avatar', 'emoji', 'badge', 'button', 'sprite', 'tracking', 'pixel', 'spacer']
        if any(pattern in img_url.lower() for pattern in skip_patterns):
            continue
        
        # Get alt text and surrounding context
        alt_text = img.get('alt', '')
        title = img.get('title', '')
        
        # Try to get context from parent elements
        context = ""
        parent = img.parent
        for _ in range(3):  # Go up 3 levels max
            if parent:
                # Look for nearby text
                text = parent.get_text(strip=True)[:200] if parent.get_text(strip=True) else ""
                if text and len(text) > len(context):
                    context = text
                parent = parent.parent
        
        # Check for figure caption
        figure = img.find_parent('figure')
        if figure:
            figcaption = figure.find('figcaption')
            if figcaption:
                context = figcaption.get_text(strip=True)
        
        images.append({
            'url': img_url,
            'alt': alt_text,
            'title': title,
            'context': context[:300] if context else alt_text,
            'source_page': url
        })
    
    return images

def scrape_single_page(url: str) -> tuple:
    """Scrape a single page and return (text_content, html_content, images, success)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract images BEFORE removing elements
        images = extract_images_from_page(url, html_content)
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        full_text = '\n'.join(lines)
        
        return full_text, html_content, images, True
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "", "", [], False

@app.post("/scrape-recursive", response_model=RecursiveScrapeResponse)
def scrape_website_recursive(request: RecursiveScrapeRequest):
    """Recursively scrape a website and all its subpages within the same context"""
    try:
        from urllib.parse import urlparse
        
        start_url = request.url.rstrip('/')
        max_depth = min(request.max_depth, 3)  # Cap at 3 to prevent infinite loops
        max_pages = min(request.max_pages, 50)  # Cap at 50 pages
        same_path_only = request.same_path_only
        
        parsed_start = urlparse(start_url)
        domain = parsed_start.netloc
        
        # Track visited URLs and pages to scrape
        visited = set()
        to_visit = [(start_url, 0)]  # (url, depth)
        pages_scraped = []
        total_chunks = 0
        
        while to_visit and len(pages_scraped) < max_pages:
            current_url, depth = to_visit.pop(0)
            
            # Skip if already visited
            if current_url in visited:
                continue
            
            visited.add(current_url)
            
            # Scrape the page
            text_content, html_content, success = scrape_single_page(current_url)
            
            if not success or not text_content or len(text_content) < 100:
                continue
            
            # Ingest the content
            try:
                metadata = extract_metadata(text_content, "website")
                chunks = ingest_document_with_semantic_chunks(text_content, current_url, "website", "", metadata)
                total_chunks += chunks
                pages_scraped.append(current_url)
                print(f"✅ Scraped: {current_url} ({chunks} chunks)")
            except Exception as e:
                print(f"Error ingesting {current_url}: {e}")
                continue
            
            # Find more links if we haven't reached max depth
            if depth < max_depth and len(pages_scraped) < max_pages:
                new_links = extract_links_from_page(current_url, html_content, same_path_only)
                
                for link in new_links:
                    if link not in visited and (link, depth + 1) not in to_visit:
                        to_visit.append((link, depth + 1))
        
        if not pages_scraped:
            return RecursiveScrapeResponse(
                success=False,
                message="No pages could be scraped",
                total_pages=0,
                total_chunks=0,
                pages_scraped=[]
            )
        
        return RecursiveScrapeResponse(
            success=True,
            message=f"✅ Scraped {len(pages_scraped)} pages from {domain} with {total_chunks} total chunks",
            total_pages=len(pages_scraped),
            total_chunks=total_chunks,
            pages_scraped=pages_scraped
        )
    
    except Exception as e:
        return RecursiveScrapeResponse(
            success=False,
            message=str(e),
            total_pages=0,
            total_chunks=0,
            pages_scraped=[]
        )

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
    try:
        # Get Pinecone stats
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
        except Exception as e:
            print(f"Pinecone stats error: {e}")
            total_vectors = 0
        
        # Count local documents
        try:
            os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
            doc_count = len([f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith('.json')])
        except Exception as e:
            print(f"Document count error: {e}")
            doc_count = 0
        
        # Count BM25 index
        try:
            bm25_data = load_bm25_index()
            bm25_count = len(bm25_data.get("documents", []))
        except Exception as e:
            print(f"BM25 index error: {e}")
            bm25_count = 0
        
        return {
            "total_vectors": total_vectors,
            "total_documents": doc_count,
            "bm25_index_size": bm25_count,
            "index_name": os.getenv("PINECONE_INDEX_NAME"),
            "chunking_type": "Semantic",
            "search_type": "Hybrid (All Sources + Memory)"
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {
            "total_vectors": 0,
            "total_documents": 0,
            "bm25_index_size": 0,
            "index_name": os.getenv("PINECONE_INDEX_NAME"),
            "chunking_type": "Semantic",
            "search_type": "Hybrid (All Sources + Memory)",
            "error": str(e)
        }

@app.get("/files")
def list_files():
    try:
        os.makedirs(UPLOADS_FOLDER, exist_ok=True)
        files = []
        for filename in os.listdir(UPLOADS_FOLDER):
            file_path = os.path.join(UPLOADS_FOLDER, filename)
            if os.path.isfile(file_path):
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "download_url": f"/download/{filename}"
                })
        return {"files": files}
    except Exception as e:
        print(f"List files error: {e}")
        return {"files": [], "error": str(e)}

# ==================== Admin Dashboard Endpoints ====================

@app.get("/admin/documents")
def get_all_documents():
    try:
        os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
        documents = []
        
        for filename in os.listdir(DOCUMENTS_FOLDER):
            if filename.endswith('.json'):
                try:
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
                except Exception as e:
                    print(f"Error reading document {filename}: {e}")
                    continue
        
        documents.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        print(f"Admin documents error: {e}")
        return {"documents": [], "total": 0, "error": str(e)}

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

@app.delete("/admin/clear-all")
def clear_all_knowledge_base(include_memories: bool = False, include_conversations: bool = False):
    """
    Clear entire knowledge base (documents, websites, YouTube).
    Optionally include memories and conversations.
    """
    results = {
        "vectors_deleted": 0,
        "documents_deleted": 0,
        "bm25_cleared": False,
        "memories_deleted": 0,
        "conversations_deleted": 0
    }
    
    try:
        # 1. Delete all vectors from Pinecone (except memories unless specified)
        try:
            dummy_vector = embeddings.embed_query("document")
            
            if include_memories:
                # Delete ALL vectors
                all_results = index.query(
                    vector=dummy_vector,
                    top_k=10000,
                    include_metadata=True
                )
                all_ids = [match['id'] for match in all_results['matches']]
            else:
                # Delete only non-memory vectors
                all_results = index.query(
                    vector=dummy_vector,
                    top_k=10000,
                    include_metadata=True
                )
                all_ids = [
                    match['id'] for match in all_results['matches'] 
                    if match['metadata'].get('type') != 'memory'
                ]
            
            if all_ids:
                # Delete in batches of 1000
                for i in range(0, len(all_ids), 1000):
                    batch = all_ids[i:i+1000]
                    index.delete(ids=batch)
                results["vectors_deleted"] = len(all_ids)
                
        except Exception as e:
            print(f"Error deleting vectors: {e}")
        
        # 2. Delete local document files
        try:
            os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
            doc_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith('.json')]
            for f in doc_files:
                os.remove(os.path.join(DOCUMENTS_FOLDER, f))
            results["documents_deleted"] = len(doc_files)
        except Exception as e:
            print(f"Error deleting documents: {e}")
        
        # 3. Clear BM25 index
        try:
            bm25_path = BM25_INDEX_FILE
            if os.path.exists(bm25_path):
                os.remove(bm25_path)
            results["bm25_cleared"] = True
        except Exception as e:
            print(f"Error clearing BM25: {e}")
        
        # 4. Delete uploaded files
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
            upload_files = os.listdir(UPLOADS_FOLDER)
            for f in upload_files:
                file_path = os.path.join(UPLOADS_FOLDER, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"Error deleting uploads: {e}")
        
        # 5. Optionally delete conversations
        if include_conversations:
            try:
                os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)
                conv_files = [f for f in os.listdir(CONVERSATIONS_FOLDER) if f.endswith('.json')]
                for f in conv_files:
                    os.remove(os.path.join(CONVERSATIONS_FOLDER, f))
                results["conversations_deleted"] = len(conv_files)
            except Exception as e:
                print(f"Error deleting conversations: {e}")
        
        return {
            "success": True, 
            "message": "Knowledge base cleared successfully!",
            "details": results
        }
        
    except Exception as e:
        return {"success": False, "message": str(e), "details": results}

@app.get("/admin/stats")
def get_admin_stats():
    try:
        # Get Pinecone stats
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
        except Exception as e:
            print(f"Pinecone stats error: {e}")
            total_vectors = 0
        
        # Count local documents
        try:
            os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
            doc_count = len([f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith('.json')])
        except Exception as e:
            print(f"Document count error: {e}")
            doc_count = 0
        
        # Load BM25 data
        try:
            bm25_data = load_bm25_index()
        except Exception as e:
            print(f"BM25 load error: {e}")
            bm25_data = {"documents": [], "metadata": []}
        
        type_counts = {"document": 0, "website": 0, "youtube": 0, "image": 0, "video": 0}
        for meta in bm25_data.get('metadata', []):
            doc_type = meta.get('type', 'document')
            if doc_type in type_counts:
                type_counts[doc_type] += 1
        
        return {
            "total_vectors": total_vectors,
            "total_documents": doc_count,
            "bm25_chunks": len(bm25_data.get('documents', [])),
            "type_breakdown": type_counts,
            "index_name": os.getenv("PINECONE_INDEX_NAME")
        }
    except Exception as e:
        print(f"Admin stats error: {e}")
        return {
            "total_vectors": 0,
            "total_documents": 0,
            "bm25_chunks": 0,
            "type_breakdown": {"document": 0, "website": 0, "youtube": 0, "image": 0, "video": 0},
            "index_name": os.getenv("PINECONE_INDEX_NAME"),
            "error": str(e)
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
    doc_context, sources, images = get_document_context_with_sources(search_query)
    
    # Send sources and images first
    sources_json = json.dumps({"type": "sources", "data": sources, "images": images, "conversation_id": conversation_id})
    yield f"data: {sources_json}\n\n"
    
    # Add image info to prompt if images found
    image_info = ""
    if images:
        image_info = "\n\nRelevant Images (can be referenced in response):\n"
        for i, img in enumerate(images[:5], 1):
            image_info += f"- Image {i}: {img.get('alt', 'No description')} (from {img.get('doc_title', 'Unknown')})\n"
    
    # Build prompt
    prompt = f"""You are a friendly and helpful AI assistant.

{conversation_history}

Available Information (Documents + Memory):
{doc_context if doc_context.strip() else "No specific information found."}
{image_info}

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
- If relevant images are available and helpful for the answer, mention them

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

# ==================== Export Endpoints ====================

class ExportRequest(BaseModel):
    content: str
    title: str = "Export"
    images: List[dict] = []

@app.post("/export/word")
async def export_to_word(request: ExportRequest):
    """Export content with images to Word document"""
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import io
    import re
    
    doc = Document()
    
    # Add title
    title = doc.add_heading(request.title, 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Parse markdown-like content and add to document
    content = request.content
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Handle headers
        if line.startswith('### '):
            doc.add_heading(line[4:], level=3)
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
        elif line.startswith('# '):
            doc.add_heading(line[2:], level=1)
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('* '):
            p = doc.add_paragraph(style='List Bullet')
            # Handle bold text
            text = line[2:]
            parts = re.split(r'\*\*(.*?)\*\*', text)
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Bold part
                    p.add_run(part).bold = True
                else:
                    p.add_run(part)
        # Handle numbered lists
        elif re.match(r'^\d+\.', line):
            p = doc.add_paragraph(style='List Number')
            text = re.sub(r'^\d+\.\s*', '', line)
            parts = re.split(r'\*\*(.*?)\*\*', text)
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    p.add_run(part).bold = True
                else:
                    p.add_run(part)
        # Regular paragraph
        else:
            p = doc.add_paragraph()
            # Handle bold text
            parts = re.split(r'\*\*(.*?)\*\*', line)
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    p.add_run(part).bold = True
                else:
                    p.add_run(part)
    
    # Add images section if images exist
    if request.images:
        doc.add_page_break()
        doc.add_heading('Related Images', level=1)
        
        for i, img in enumerate(request.images, 1):
            try:
                # Download image
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(img['url'], headers=headers, timeout=10)
                if response.status_code == 200:
                    image_stream = io.BytesIO(response.content)
                    doc.add_picture(image_stream, width=Inches(5))
                    
                    # Add caption
                    caption = img.get('alt', '') or img.get('context', f'Image {i}')
                    cap_para = doc.add_paragraph(caption[:200])
                    cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cap_para.runs[0].font.size = Pt(10)
                    cap_para.runs[0].font.italic = True
                    doc.add_paragraph()  # Spacing
            except Exception as e:
                # If image fails, add placeholder text
                doc.add_paragraph(f"[Image {i}: {img.get('alt', 'Could not load image')}]")
    
    # Save to bytes
    doc_bytes = io.BytesIO()
    doc.save(doc_bytes)
    doc_bytes.seek(0)
    
    # Return as downloadable file
    filename = f"{request.title.replace(' ', '_')}.docx"
    return StreamingResponse(
        doc_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/export/pdf")
async def export_to_pdf(request: ExportRequest):
    """Export content with images to PDF"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, ListFlowable, ListItem
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io
    import re
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20
    ))
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        leading=14
    ))
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_CENTER,
        italic=True,
        spaceAfter=15
    ))
    
    story = []
    
    # Add title
    story.append(Paragraph(request.title, styles['CustomTitle']))
    story.append(Spacer(1, 20))
    
    # Parse content
    content = request.content
    lines = content.split('\n')
    current_list = []
    
    for line in lines:
        line = line.strip()
        if not line:
            # Flush any pending list
            if current_list:
                story.append(ListFlowable(current_list, bulletType='bullet'))
                current_list = []
            continue
        
        # Convert markdown bold to HTML
        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        
        # Handle headers
        if line.startswith('### '):
            if current_list:
                story.append(ListFlowable(current_list, bulletType='bullet'))
                current_list = []
            story.append(Paragraph(line[4:], styles['Heading3']))
        elif line.startswith('## '):
            if current_list:
                story.append(ListFlowable(current_list, bulletType='bullet'))
                current_list = []
            story.append(Paragraph(line[3:], styles['Heading2']))
        elif line.startswith('# '):
            if current_list:
                story.append(ListFlowable(current_list, bulletType='bullet'))
                current_list = []
            story.append(Paragraph(line[2:], styles['CustomHeading']))
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('* '):
            current_list.append(ListItem(Paragraph(line[2:], styles['CustomBody'])))
        # Handle numbered lists
        elif re.match(r'^\d+\.', line):
            if current_list:
                story.append(ListFlowable(current_list, bulletType='bullet'))
                current_list = []
            text = re.sub(r'^\d+\.\s*', '', line)
            story.append(Paragraph(text, styles['CustomBody']))
        # Regular paragraph
        else:
            if current_list:
                story.append(ListFlowable(current_list, bulletType='bullet'))
                current_list = []
            story.append(Paragraph(line, styles['CustomBody']))
    
    # Flush any remaining list
    if current_list:
        story.append(ListFlowable(current_list, bulletType='bullet'))
    
    # Add images section
    if request.images:
        story.append(Spacer(1, 30))
        story.append(Paragraph("Related Images", styles['CustomHeading']))
        story.append(Spacer(1, 15))
        
        for i, img in enumerate(request.images, 1):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(img['url'], headers=headers, timeout=10)
                if response.status_code == 200:
                    image_stream = io.BytesIO(response.content)
                    img_obj = RLImage(image_stream, width=4*inch, height=3*inch, kind='proportional')
                    story.append(img_obj)
                    
                    caption = img.get('alt', '') or img.get('context', f'Image {i}')
                    story.append(Paragraph(caption[:200], styles['Caption']))
            except Exception as e:
                story.append(Paragraph(f"[Image {i}: Could not load]", styles['Caption']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    filename = f"{request.title.replace(' ', '_')}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
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
