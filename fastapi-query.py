import pickle
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from common import get_embedding,get_llm_response
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import asyncio
from contextlib import asynccontextmanager


class Query(BaseModel):
    query: str = Field(default="", description="Query to search for")
    top_k: int = Field(default=3, ge=1, description="Number of top results to return")


class Question(BaseModel):
    prompt: str = Field(default="", description="Prompt to generate a response")
    model: str = Field(default="llama3:8b", description="model to use in ollama")
    temperature: float = Field(default=0, description="Temperature for the model")
    

async def load_vector_database(input_file):
    """Load vector database from local file"""
    with open(input_file, 'rb') as f:
        return pickle.load(f)

async def find_matching_videos(query, vector_db, top_k):
    """Find matching videos for a given query"""
    query_embedding = await asyncio.to_thread(get_embedding, query)

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

    candidates = video_similarities[:top_k]

    ranked_results = []
    for video, score in candidates:
        ranked_results.append({
            "id": video.get('video_id', ''),
            "name": video.get('video_name', ''),
            "link": video.get('video_link', ''),
            "sub": video.get('video_sub', ''),
            "teacher": video.get('video_teacher', ''),
            'cover_link': video.get('video_cover_link', ''),
            'vid': video.get('video_vid', ''),
            'summary': video.get('video_summary', ''),    
            'score': float(score),
            # "distance": float(distance),
            # "similarity": float(similarity),
        })
    return ranked_results


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
def keyword_search_hjb(query, vector_db):
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

def keyword_search_znd(query, vector_db):
    """Perform keyword search and return matching scores"""
    query_keywords = set(jieba.cut(query.lower()))
    video_scores = []
    
    for video in vector_db:
        video_text = (video['video_name'] + ' ' + video['video_sub'] + ' ' + video['video_summary']).lower()
        video_keywords = set(jieba.cut(video_text))
        common_keywords = query_keywords.intersection(video_keywords)
        
        keyword_score = len(common_keywords) / len(query_keywords)  # Simple ratio of matching keywords
        video_scores.append((video, keyword_score))
    
    return video_scores

async def combined_search_hjb(query, vector_db, vector_weight=0.5, keyword_weight=0.5, top_k=3):
    """Combine vector search and keyword search results"""
    query_embedding = get_embedding(query)
    
    # Perform vector search
    vector_scores = vector_search(query_embedding, vector_db, weight=vector_weight)
    
    # Perform keyword search
    keyword_scores = keyword_search_hjb(query, vector_db)
    
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

    ranked_results = []
    for video, combined_score, vector_score, keyword_score in top_videos:
        ranked_results.append({
            "id": video.get('video_id', ''),
            "name": video.get('video_name', ''),
            "link": video.get('video_link', ''),
            "sub": video.get('video_sub', ''),
            "teacher": video.get('video_teacher', ''),
            'cover_link': video.get('video_cover_link', ''),
            'vid': video.get('video_vid', ''),
            'summary': video.get('video_summary', ''),    
            'score': {
                'combined': float(combined_score),
                'keyword': float(keyword_score),
                'vector': float(vector_score),
            },
            # "distance": float(distance),
            # "similarity": float(similarity),
        })
    return ranked_results

async def combined_search_znd(query, vector_db, vector_weight=0.5, keyword_weight=0.5, top_k=3):
    """Combine vector search and keyword search results"""
    query_embedding = get_embedding(query)
    
    # Perform vector search
    vector_scores = vector_search(query_embedding, vector_db, weight=vector_weight)
    
    # Perform keyword search
    keyword_scores = keyword_search_znd(query, vector_db)
    
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
    ranked_results = []
    for video, combined_score, vector_score, keyword_score in top_videos:
        ranked_results.append({
            "id": video.get('video_id', ''),
            "name": video.get('video_name', ''),
            "link": video.get('video_link', ''),
            "sub": video.get('video_sub', ''),
            "teacher": video.get('video_teacher', ''),
            'cover_link': video.get('video_cover_link', ''),
            'vid': video.get('video_vid', ''),
            'summary': video.get('video_summary', ''),    
            'score': {
                'combined': float(combined_score),
                'keyword': float(keyword_score),
                'vector': float(vector_score),
            },
            # "distance": float(distance),
            # "similarity": float(similarity),
        })
    return ranked_results

# 加载vector database
vector_db_file = "znd456_vector_database.pkl"
vector_db_file_hjb = "hjb-sx-1.1_vector_database.pkl"
vector_db = None
vector_db_hjb = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, vector_db_hjb
    vector_db = await load_vector_database(vector_db_file)
    vector_db_hjb = await load_vector_database(vector_db_file_hjb)
    print("==============================================")
    print("| Server is running and vector dbs has loaded|")
    print("==============================================")
    yield
    # Clean up the ML models and release the resources
    vector_db = None
    vector_db_hjb = None

app = FastAPI(lifespan=lifespan)

@app.post("/aisearchznd",tags=["AI-Local-LLM-RAG"],description="基于向量库的视频推荐搜索，实施范围“456年级语数英重难点”，纯使用向量数据库进行向量距离和余弦角度最优化搜索，速度较快", summary="视频AI推荐搜索-重难点接口", response_description="返回视频列表")
async def search(query: Query):
    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector database not loaded")
    
    matches = await find_matching_videos(query.query, vector_db,top_k=query.top_k)
    return {"results": matches}

@app.post("/mix/aisearchznd",tags=["AI-Local-LLM-RAG"],description="基于向量库和关键字的视频推荐混合搜索，实施范围“456年级语数英重难点”，速度较快", summary="视频AI推荐混合搜索-重难点接口", response_description="返回视频列表")
async def search(query: Query):
    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector database not loaded")
    
    matches = await combined_search_znd(query.query, vector_db,vector_weight=0.2, keyword_weight=0.8,top_k=query.top_k)
    return {"results": matches}

@app.post("/aisearchhjb",tags=["AI-Local-LLM-RAG"],description="基于向量库的视频推荐搜索，实施范围“沪教版数学切片视频”，纯使用向量数据库进行向量距离和余弦角度最优化搜索，速度较快", summary="视频AI推荐搜索-沪教版数学接口", response_description="返回视频列表")
async def search(query: Query):
    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector database not loaded")
    
    matches = await find_matching_videos(query.query, vector_db_hjb,top_k=query.top_k)
    return {"results": matches}

@app.post("/mix/aisearchhjb",tags=["AI-Local-LLM-RAG"],description="基于向量库和关键字的视频推荐混合搜索，实施范围“沪教版数学切片视频”，速度较快", summary="视频AI推荐混合搜索-沪教版数学接口", response_description="返回视频列表")
async def search(query: Query):
    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector database not loaded")
    
    matches = await combined_search_hjb(query.query, vector_db_hjb,vector_weight=0.2, keyword_weight=0.8,top_k=query.top_k)
    return {"results": matches}

@app.post("/aiquestion",tags=["AI-Local-LLM-RAG"],description="调用LLM回答问题", summary="AI问答", response_description="返回答案")
async def question(query: Question):
    res = await get_llm_response(query.prompt,query.model,temperature=query.temperature)
    return {"result": res}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
## run
## uvicorn fastapi-query:app --host 0.0.0.0 --port 8000 --work 2
