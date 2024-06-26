import pickle
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from common import get_embedding

def load_vector_database(input_file):
    """Load vector database from local file"""
    with open(input_file, 'rb') as f:
        return pickle.load(f)

def vector_search(query_embedding, vector_db, weight=0.5):
    """Perform vector search and return weighted scores"""
    video_scores = []
    for video in vector_db:
        similarities = [
            cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            for chunk_embedding in video['embeddings']
        ]
        distances = [
            euclidean_distances([query_embedding], [chunk_embedding])[0][0]
            for chunk_embedding in video['embeddings']
        ]
        
        max_similarity = max(similarities)
        min_distance = min(distances)
        
        # Combine similarity and distance using weight
        weighted_score = max_similarity - weight * min_distance
        video_scores.append((video, weighted_score))
    
    return video_scores


def keyword_search(query, vector_db):
    """Perform keyword search and return matching scores"""
    query_keywords = set(jieba.cut(query.lower()))
    video_scores = []
    
    for video in vector_db:
        video_text = (video['video_name'] + ' ' + video['video_sub']).lower()
        video_keywords = set(jieba.cut(video_text))
        common_keywords = query_keywords.intersection(video_keywords)
        
        keyword_score = len(common_keywords) / len(query_keywords)  # Simple ratio of matching keywords
        video_scores.append((video, keyword_score))
    
    return video_scores

def combined_search(query, vector_db, vector_weight=0.5, keyword_weight=0.5, top_k=3):
    """Combine vector search and keyword search results"""
    query_embedding = get_embedding(query)
    
    # Perform vector search
    vector_scores = vector_search(query_embedding, vector_db, weight=vector_weight)
    
    # Perform keyword search
    keyword_scores = keyword_search(query, vector_db)
    
    # Combine scores
    combined_scores = []
    for (video_v, v_score), (video_k, k_score) in zip(vector_scores, keyword_scores):
        if video_v == video_k:
            combined_score = vector_weight * v_score + keyword_weight * k_score
            combined_scores.append((video_v, combined_score, v_score, k_score))
    
    # Sort by combined score
    combined_scores.sort(key=lambda x: -x[1])
    
    # Select top_k results
    top_videos = combined_scores[:top_k]
    
    return top_videos


if __name__ == "__main__":
    vector_db_file = "hjb-sx-1.1_vector_database.pkl"
    
    # Load vector database
    vector_db = load_vector_database(vector_db_file)
    
    # Example query
    query = input("Enter your query: ")
    
    # Find matching videos
    matches = combined_search(query, vector_db, vector_weight=0.3, keyword_weight=0.8, top_k=5)
    

    for video, combined_score, vector_score, keyword_score in matches:
        print(f"ID: {video['video_id']}")
        print(f"Name: {video['video_name']}")
        print(f"Sub: {video['video_sub']}")
        print(f"Combined Score: {combined_score}")
        print(f"Vector Score: {vector_score}")
        print(f"Keyword Score: {keyword_score}")
        print("---")