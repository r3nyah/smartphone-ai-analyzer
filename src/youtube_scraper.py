import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from youtubesearchpython import VideosSearch
from itertools import islice
import os
import time

# Configuration
PROCESSED_DATA_DIR = "data/processed"

def search_video_id(device_name):
    """
    Searches YouTube for 'Review [Device Name] Indonesia' and returns the Video ID.
    """
    query = f"Review {device_name} Indonesia"
    print(f"    [...] Searching YouTube for: '{query}'")
    
    try:
        videos_search = VideosSearch(query, limit=1)
        results = videos_search.result()
        
        if results['result']:
            video_id = results['result'][0]['id']
            video_title = results['result'][0]['title']
            print(f"    [+] Found Video: {video_title} (ID: {video_id})")
            return video_id
        else:
            print(f"    [-] No video found for {device_name}")
            return None
            
    except Exception as e:
        print(f"    [!] Error searching video: {e}")
        return None

def get_comments(video_id, limit=50):
    """
    Downloads comments from a specific video ID using the correct method.
    """
    if not video_id: return []
    
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        generator = downloader.get_comments(video_id, sort_by=SORT_BY_POPULAR) 
        
        for comment in islice(generator, limit):
            comments.append(comment['text'])
            
    except Exception as e:
        print(f"    [!] Error scraping comments: {e}")
        
    return comments

def analyze_sentiment(comments):
    """
    Analyzes sentiment based on Indonesian tech keywords.
    """
    if not comments: return 0, 0, 0

    positive_keywords = [
        'bagus', 'keren', 'mantap', 'kenceng', 'worth', 'mulus', 
        'dingin', 'smooth', 'gacor', 'awet', 'best', 'suka', 'enak', 'juara',
        'rekomendasi', 'terbaik', 'sakti', 'ngebut'
    ]
    negative_keywords = [
        'panas', 'lag', 'lemot', 'boros', 'jelek', 'nyesel', 
        'frame drop', 'bug', 'mahal', 'kecewa', 'hang', 'rusak', 
        'sampah', 'bapuk', 'kentang', 'tros'
    ]
    
    score = 50 # Start neutral
    total_text = " ".join(comments).lower()
    
    pos_count = 0
    neg_count = 0

    for word in positive_keywords:
        count = total_text.count(word)
        pos_count += count
        score += (count * 1.5)

    for word in negative_keywords:
        count = total_text.count(word)
        neg_count += count
        score -= (count * 2.5) 
    
    final_score = max(0, min(100, score))
    return int(final_score), pos_count, neg_count

def run_production_demo():
    print("[+] Loading Data...")
    csv_path = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
    
    if not os.path.exists(csv_path):
        print("Error: training_data.csv not found. Run data_pipeline.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # Pick 3 random devices
    sample_devices = df['full_name'].sample(3).tolist()
    
    print(f"[-] Starting Analysis for: {sample_devices}\n")
    
    for device in sample_devices:
        print(f"--- ANALYZING: {device} ---")
        
        video_id = search_video_id(device)
        
        if video_id:
            comments = get_comments(video_id, limit=30) 
            
            if comments:
                print(f"    [+] Scraped {len(comments)} comments.")
                
                score, pos, neg = analyze_sentiment(comments)
                
                print(f"    [=] SENTIMENT SCORE: {score}/100")
                print(f"        (+{pos} Positive words | -{neg} Negative words)")
                
                if score >= 75: verdict = "Highly Recommended"
                elif score >= 50: verdict = "Good / Average"
                else: verdict = "Not Recommended (Issues Detected)"
                
                print(f"    [=] AI VERDICT: {verdict}")
            else:
                print("    [-] Comments extraction returned empty.")
        
        print("\n")
        time.sleep(1)

if __name__ == "__main__":
    run_production_demo()