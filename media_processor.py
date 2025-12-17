import whisper
import base64
import os
import tempfile
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Load Whisper model (will download on first use)
whisper_model = None

def get_whisper_model():
    """Load Whisper model lazily"""
    global whisper_model
    if whisper_model is None:
        print("📥 Loading Whisper model (first time may take a while)...")
        whisper_model = whisper.load_model("base")
    return whisper_model

def extract_metadata(text: str, source_type: str, filename: str):
    """Use LLM to extract metadata from content"""
    
    print("🧠 Extracting metadata...")
    
    prompt = f"""Analyze the following {source_type} content and extract metadata.

Content:
{text[:3000]}

Respond in JSON format only:
{{
    "title": "A descriptive title for this content (max 10 words)",
    "summary": "A brief summary of what this content is about (2-3 sentences)",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "topics": ["main topic 1", "main topic 2"],
    "language": "detected language (e.g., English, Arabic, etc.)"
}}

JSON only, no other text:"""

    try:
        response = llm.invoke(prompt)
        
        # Parse JSON from response
        json_str = response.content.strip()
        # Remove markdown code blocks if present
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()
        
        metadata = json.loads(json_str)
        print(f"📋 Title: {metadata.get('title', 'N/A')}")
        return metadata
    except Exception as e:
        print(f"⚠️ Could not extract metadata: {e}")
        return {
            "title": filename,
            "summary": "No summary available",
            "keywords": [],
            "topics": [],
            "language": "unknown"
        }

def transcribe_video(file_path: str):
    """Transcribe audio/video file using Whisper"""
    
    print(f"🎥 Transcribing: {file_path}")
    
    model = get_whisper_model()
    result = model.transcribe(file_path)
    
    text = result["text"]
    print(f"📝 Transcribed {len(text)} characters")
    
    return text

def extract_text_from_image(file_path: str):
    """Extract text from image using OpenAI Vision"""
    
    print(f"🖼️ Processing image: {file_path}")
    
    # Read and encode image
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Determine media type
    ext = os.path.splitext(file_path)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    media_type = media_types.get(ext, "image/jpeg")
    
    # Use GPT-4o Vision to extract text
    from langchain_core.messages import HumanMessage
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract ALL text from this image. Return only the extracted text, nothing else. If there's no text, describe the image content in detail."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_data}"},
            },
        ],
    )
    
    response = llm.invoke([message])
    text = response.content
    
    print(f"📝 Extracted {len(text)} characters")
    return text

def ingest_media(file_path: str):
    """Process and ingest media file into Pinecone"""
    
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    
    # Determine file type and process
    video_extensions = [".mp4", ".mp3", ".wav", ".m4a", ".webm", ".avi", ".mov"]
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    
    if ext in video_extensions:
        text = transcribe_video(file_path)
        media_type = "video"
    elif ext in image_extensions:
        text = extract_text_from_image(file_path)
        media_type = "image"
    else:
        print(f"❌ Unsupported file type: {ext}")
        return False
    
    if not text:
        print("❌ No text extracted")
        return False
    
    # Extract metadata using LLM
    metadata = extract_metadata(text, media_type, filename)
    
    # Split into chunks
    chunks = text_splitter.split_text(text)
    print(f"📦 Split into {len(chunks)} chunks")
    
    # Create embeddings and upsert
    print("🔄 Creating embeddings...")
    
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        
        index.upsert(vectors=[{
            "id": f"{media_type}_{filename}_{i}",
            "values": vector,
            "metadata": {
                "text": chunk,
                "source": file_path,
                "type": media_type,
                "filename": filename,
                "title": metadata.get("title", filename),
                "summary": metadata.get("summary", ""),
                "keywords": metadata.get("keywords", []),
                "topics": metadata.get("topics", []),
                "language": metadata.get("language", "unknown")
            }
        }])
    
    print(f"✅ Ingested: {file_path}")
    print(f"   📋 Title: {metadata.get('title', 'N/A')}")
    print(f"   📝 Summary: {metadata.get('summary', 'N/A')[:100]}...")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python media_processor.py <image_path>   - Process image (OCR)")
        print("  python media_processor.py <video_path>   - Process video/audio (transcription)")
        print("")
        print("Supported formats:")
        print("  Images: jpg, jpeg, png, gif, webp")
        print("  Video/Audio: mp4, mp3, wav, m4a, webm, avi, mov")
    else:
        file_path = sys.argv[1]
        ingest_media(file_path)