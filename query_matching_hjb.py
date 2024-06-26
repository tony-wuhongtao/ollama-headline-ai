import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from common import get_embedding

def load_vector_database(input_file):
    """Load vector database from local file"""
    with open(input_file, 'rb') as f:
        return pickle.load(f)

def find_matching_videos(query, vector_db, top_k=3):
    """Find matching videos for a given query"""
    query_embedding = get_embedding(query)
    
    # # Compute max similarities for each video
    # video_similarities = []
    # for video in vector_db:
    #     max_similarity = max([
    #         cosine_similarity([query_embedding], [chunk_embedding])[0][0]
    #         for chunk_embedding in video['embeddings']
    #     ])
    #     min_distance = min([
    #         euclidean_distances([query_embedding], [chunk_embedding])[0][0]
    #         for chunk_embedding in video['embeddings']
    #     ])
    #     video_similarities.append((video, max_similarity, min_distance))

    # video_similarities.sort(key=lambda x: (x[2], -x[1]))  # sort by min distance (asc) and max similarity (desc)
    # candidates = video_similarities[:top_k]

    
    # ranked_results = []
    # for video, similarity, distance in candidates:
    #     ranked_results.append((video, similarity, distance))
    #     ranked_results.sort(key=lambda x: (x[2], -x[1]))

    
    # return ranked_results

    video_similarities = []
    for video in vector_db:
        similarities = [
            cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            for chunk_embedding in video['embeddings']
        ]
        distances = [
            euclidean_distances([query_embedding], [chunk_embedding])[0][0]
            for chunk_embedding in video['embeddings']
        ]
        
        # Weighted combination of similarity and distance
        weighted_scores = [sim - 0.5 * dist for sim, dist in zip(similarities, distances)]
        
        # Use the max weighted score for each video
        max_score = max(weighted_scores)
        
        video_similarities.append((video, max_score))
    
    # Sort videos by weighted score in descending order
    video_similarities.sort(key=lambda x: -x[1])
    
    # Select top_k videos
    top_videos = video_similarities[:top_k]
    
    ranked_results = [(video, score) for video, score in top_videos]
    
    return ranked_results


if __name__ == "__main__":
    vector_db_file = "hjb-sx-1.1_vector_database.pkl"
    
    # Load vector database
    vector_db = load_vector_database(vector_db_file)
    
    # Example query
    query = input("Enter your query: ")
    
    # Find matching videos
    matches = find_matching_videos(query, vector_db)
    

    for video, score in matches:
        print(f"ID: {video['video_id']}")
        print(f"Name: {video['video_name']}")
        print(f"Sub: {video['video_sub']}")
        print(f"Score: {score}")
        print("---")