import re

# Simple list to hold history
memory = []

def add_memory(text):
    # Keep memory clean of duplicates
    if text not in memory:
        memory.append(text)

def retrieve_memory(query, k=3):
    if not memory:
        return []

    # 1. Clean the query into keywords
    query_words = set(re.findall(r'\w+', query.lower()))
    
    # 2. Score each past memory based on word overlap
    scored_memory = []
    for text in memory:
        text_words = set(re.findall(r'\w+', text.lower()))
        # Count how many words match
        score = len(query_words.intersection(text_words))
        scored_memory.append((score, text))

    # 3. Sort by score (highest first)
    scored_memory.sort(key=lambda x: x[0], reverse=True)

    # 4. Return top K results
    # If no keywords match, it naturally falls back to the most recent items
    results = [item[1] for item in scored_memory[:k]]
    
    return results