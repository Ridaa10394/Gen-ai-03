"""One-time keys smoke test. Delete after verifying."""
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

print("=== Env vars ===")
print(f"GROQ_API_KEY:   {'SET' if GROQ_API_KEY and 'your_' not in GROQ_API_KEY else 'MISSING/PLACEHOLDER'}")
print(f"QDRANT_URL:     {QDRANT_URL if QDRANT_URL and 'your_' not in QDRANT_URL else 'MISSING/PLACEHOLDER'}")
print(f"QDRANT_API_KEY: {'SET' if QDRANT_API_KEY and 'your_' not in QDRANT_API_KEY else 'MISSING/PLACEHOLDER'}")

if not (GROQ_API_KEY and QDRANT_URL and QDRANT_API_KEY):
    print("\nERROR: missing env vars.")
    sys.exit(1)

if QDRANT_API_KEY == QDRANT_URL or QDRANT_API_KEY.startswith("https://"):
    print("\nERROR: QDRANT_API_KEY looks like a URL, not the JWT key.")
    sys.exit(1)

print("\n=== Local embedding test (fastembed) ===")
from rag.embeddings import embed_query
try:
    v = embed_query("hello world")
    print(f"OK: {len(v)}-dim embedding")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("\n=== Groq chat test ===")
from groq import Groq
client = Groq(api_key=GROQ_API_KEY)
try:
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Reply with just: ok"}],
    )
    print(f"OK: {r.choices[0].message.content.strip()[:60]}")
except Exception as e:
    print(f"FAIL: {str(e)[:200]}")
    sys.exit(1)

print("\n=== Qdrant test ===")
from qdrant_client import QdrantClient
try:
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    cols = qc.get_collections()
    print(f"OK: connected. Existing collections: {[c.name for c in cols.collections]}")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("\nAll systems working. Run: streamlit run app.py")
