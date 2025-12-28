import praw
import json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load credentials
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

SUBREDDITS = ["giki", "lums", "nust"]
TARGET_PER_SUB = 1000

def scrape_subreddit(sub_name):
    print(f"\n--- Starting Deep Scrape for r/{sub_name} ---")
    subreddit = reddit.subreddit(sub_name)
    scraped_data = []
    seen_ids = set()

    # We iterate through different categories to gather more than 1000 posts
    # PRAW limits individual listings to 1000, but they often overlap.
    generators = [
        ("Top (All Time)", subreddit.top(time_filter="all", limit=1000)),
        ("Top (Year)", subreddit.top(time_filter="year", limit=1000)),
        ("Hot", subreddit.hot(limit=1000)),
        ("New", subreddit.new(limit=1000))
    ]

    for label, post_generator in generators:
        if len(scraped_data) >= TARGET_PER_SUB:
            break
            
        print(f"  Attempting Category: {label}...")
        
        try:
            for post in post_generator:
                if len(scraped_data) >= TARGET_PER_SUB:
                    break
                
                if post.id in seen_ids:
                    continue

                try:
                    # Respect rate limits for comment expansion
                    post.comments.replace_more(limit=0)
                    comments = [c.body for c in post.comments.list()[:15]]
                    
                    post_entry = {
                        "id": post.id,
                        "subreddit": sub_name,
                        "title": post.title,
                        "body": post.selftext,
                        "upvotes": post.score,
                        "timestamp": datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat(),
                        "num_comments": post.num_comments,
                        "comments": comments,
                        "url": f"https://reddit.com{post.permalink}"
                    }

                    scraped_data.append(post_entry)
                    seen_ids.add(post.id)

                    # Periodic reporting
                    if len(scraped_data) % 50 == 0:
                        print(f"    Progress: {len(scraped_data)}/{TARGET_PER_SUB}")
                        # Small break to respect API
                        time.sleep(1)

                except Exception as e:
                    print(f"    Error on post {post.id}: {e}")
                    time.sleep(5) # Wait if it's a rate limit issue
                    continue

        except Exception as e:
            print(f"  Category {label} failed or limit reached: {e}")
            continue

    # Save data
    filename = f"{sub_name}_data.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=4)
    print(f"Done! Saved {len(scraped_data)} unique posts for r/{sub_name}")

if __name__ == "__main__":
    for sub in SUBREDDITS:
        scrape_subreddit(sub)
        print("Waiting 10 seconds before next subreddit...")
        time.sleep(10)