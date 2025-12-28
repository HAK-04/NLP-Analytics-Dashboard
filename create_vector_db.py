import chromadb
from chromadb.utils import embedding_functions
import json
import os
import re
import statistics
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from dotenv import load_dotenv

nltk.download('punkt_tab')

# Load Env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


DB_DIRECTORY = "university_rag_db"
COLLECTION_NAME = "pak_university_posts"
JSON_FILES = ["giki_data.json", "lums_data.json", "nust_data.json"]

# NLP Setup
def setup_nltk():
    resources = ["vader_lexicon", "stopwords", "punkt"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
            
setup_nltk()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):

    # lowercase, remove special char, tokenize, stopword removal
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def analyze_text_metrics(text):

    if not text or not text.strip():
        return {
            "sentiment": 0.0,
            "avg_sent_len": 0.0,
            "word_count": 0,
            "cleaned_text": ""
        }

    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # 1. Sentiment
    sentiment = sia.polarity_scores(text)["compound"]
    
    # 2. Complexity
    avg_sentence_len = len(words) / len(sentences) if sentences else 0
    
    # 3. Cleaned Text
    cleaned_text = preprocess_text(text)
    
    return {
        "sentiment": sentiment,
        "avg_sent_len": avg_sentence_len,
        "word_count": len(words),
        "cleaned_text": cleaned_text
    }

def get_chunks(post):

    chunks = []
    
    # Chunk 1
    post_content = f"{post.get('title', '')}\n{post.get('body', '')}".strip()
    
    if post_content:
        metrics = analyze_text_metrics(post_content)
        chunks.append({
            "text": post_content,
            "type": "submission",
            **metrics
        })

    # Chunk 2+
    current_chunk_text = ""
    for comment in post.get('comments', []):
        if not comment or not comment.strip():
            continue
            
        if len(current_chunk_text) + len(comment) > 1000:
            metrics = analyze_text_metrics(current_chunk_text)
            # content check
            if current_chunk_text.strip():
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "type": "comments",
                    **metrics
                })
            current_chunk_text = ""
        current_chunk_text += f"{comment}\n"
    
    # check for remaining comments
    if current_chunk_text.strip():
        metrics = analyze_text_metrics(current_chunk_text)
        chunks.append({
            "text": current_chunk_text.strip(),
            "type": "comments",
            **metrics
        })

    return chunks

def create_database():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    client = chromadb.PersistentClient(path=DB_DIRECTORY)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Resetting database...")
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
    
    total_chunks = 0
    print("Processing files and performing NLP analysis...")

    for file_path in JSON_FILES:
        if not os.path.exists(file_path): continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            posts = json.load(f)

        ids, docs, metadatas = [], [], []

        for post in posts:
            post_chunks = get_chunks(post)
            
            for idx, chunk in enumerate(post_chunks):
                chunk_id = f"{post['id']}_{idx}"
                
                ids.append(chunk_id)
                docs.append(chunk['text'])
                
                # NLP metrics in meta
                metadatas.append({
                    "post_id": post['id'],
                    "subreddit": post.get('subreddit', 'unknown'),
                    "chunk_type": chunk['type'],
                    "upvotes": post['upvotes'],
                    "timestamp": post['timestamp'],
                    "sentiment": chunk['sentiment'],          
                    "complexity": chunk['avg_sent_len'],      
                    "word_count": chunk['word_count'],        
                    "clean_text": chunk['cleaned_text'][:1000]
                })

        # batch
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                documents=docs[i:end],
                metadatas=metadatas[i:end]
            )
        
        total_chunks += len(ids)
        print(f"Processed {file_path}: {len(ids)} chunks indexed.")

    print(f"\nDatabase Ready. Total chunks: {total_chunks}")
    # print(f"Location: {DB_DIRECTORY}")

if __name__ == "__main__":
    create_database()