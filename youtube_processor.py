from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import re
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

def extract_video_id(url: str):
    """Extract video ID from YouTube URL"""
    
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def get_transcript(video_id: str):
    """Get transcript from YouTube video"""
    
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id)
        
        full_text = ' '.join([entry.text for entry in transcript_data])
        return full_text
        
    except Exception as e:
        print(f"❌ Error getting transcript: {e}")
        return None

def extract_metadata(text: str, video_id: str, url: str):
    """Use LLM to extract metadata from content"""
    
    print("🧠 Extracting metadata...")
    
    prompt = f"""Analyze the following YouTube video transcript and extract metadata.

Transcript:
{text[:3000]}

Respond in JSON format only:
{{
    "title": "A descriptive title for this video (max 10 words)",
    "summary": "A brief summary of what this video is about (2-3 sentences)",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "topics": ["main topic 1", "main topic 2"],
    "language": "detected language (e.g., English, Arabic, etc.)",
    "content_type": "tutorial/entertainment/news/educational/review/other"
}}

JSON only, no other text:"""

    try:
        response = llm.invoke(prompt)
        
        json_str = response.content.strip()
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
            "title": f"YouTube Video {video_id}",
            "summary": "No summary available",
            "keywords": [],
            "topics": [],
            "language": "unknown",
            "content_type": "unknown"
        }

def ingest_youtube(url: str):
    """Get YouTube transcript and ingest into Pinecone"""
    
    print(f"🎬 Processing YouTube: {url}")
    
    video_id = extract_video_id(url)
    
    if not video_id:
        print("❌ Could not extract video ID from URL")
        return False
    
    print(f"📺 Video ID: {video_id}")
    
    print("📝 Fetching transcript...")
    text = get_transcript(video_id)
    
    if not text:
        print("❌ No transcript available for this video")
        return False
    
    print(f"📄 Got {len(text)} characters")
    
    # Extract metadata using LLM
    metadata = extract_metadata(text, video_id, url)
    
    # Split into chunks
    chunks = text_splitter.split_text(text)
    print(f"📦 Split into {len(chunks)} chunks")
    
    # Create embeddings and upsert
    print("🔄 Creating embeddings...")
    
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        
        index.upsert(vectors=[{
            "id": f"youtube_{video_id}_{i}",
            "values": vector,
            "metadata": {
                "text": chunk,
                "source": url,
                "type": "youtube",
                "video_id": video_id,
                "title": metadata.get("title", f"YouTube Video {video_id}"),
                "summary": metadata.get("summary", ""),
                "keywords": metadata.get("keywords", []),
                "topics": metadata.get("topics", []),
                "language": metadata.get("language", "unknown"),
                "content_type": metadata.get("content_type", "unknown")
            }
        }])
    
    print(f"✅ Ingested YouTube video: {video_id}")
    print(f"   📋 Title: {metadata.get('title', 'N/A')}")
    print(f"   📝 Summary: {metadata.get('summary', 'N/A')[:100]}...")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python youtube_processor.py <youtube_url>")
    else:
        url = sys.argv[1]
        ingest_youtube(url)