import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
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

def scrape_url(url: str):
    """Scrape a single URL and return its text content"""
    
    try:
        print(f"🌐 Fetching: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        print(f"📄 Extracted {len(text)} characters")
        return text
    
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None

def ingest_url(url: str):
    """Scrape URL and ingest into Pinecone"""
    
    # Scrape content
    text = scrape_url(url)
    
    if not text:
        return False
    
    # Split into chunks
    chunks = text_splitter.split_text(text)
    print(f"📦 Split into {len(chunks)} chunks")
    
    # Create embeddings and upsert
    print("🔄 Creating embeddings...")
    
    domain = urlparse(url).netloc
    
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        
        index.upsert(vectors=[{
            "id": f"web_{domain}_{i}",
            "values": vector,
            "metadata": {
                "text": chunk,
                "source": url,
                "type": "website",
                "domain": domain
            }
        }])
    
    print(f"✅ Ingested: {url}")
    return True

def scrape_multiple_pages(base_url: str, max_pages: int = 5):
    """Scrape multiple pages from a website"""
    
    visited = set()
    to_visit = [base_url]
    domain = urlparse(base_url).netloc
    
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        
        if url in visited:
            continue
        
        visited.add(url)
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Ingest this page
            ingest_url(url)
            
            # Find more links on the same domain
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if urlparse(full_url).netloc == domain and full_url not in visited:
                    to_visit.append(full_url)
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Scraped {len(visited)} pages from {domain}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scraper.py <url>              - Scrape single page")
        print("  python scraper.py <url> --crawl 5    - Scrape up to 5 pages")
    else:
        url = sys.argv[1]
        
        if "--crawl" in sys.argv:
            idx = sys.argv.index("--crawl")
            max_pages = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 5
            scrape_multiple_pages(url, max_pages)
        else:
            ingest_url(url)