import pickle
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from common import get_embedding,get_llm_response
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from summary import get_summary
from summary_coze import get_summary_coze
from fastapi.middleware.cors import CORSMiddleware



class Query(BaseModel):
    query: str = Field(default="", description="Query to search for")
    db_key: str = Field(default='znd_v2', description="vecter db key,znd | znd_v2 | hjb")
    top_k: int = Field(default=3, ge=1, description="Number of top results to return")
    vector_weight: float = Field(default=0.2,description="Vector weight default 0.2")
    keyword_weight: float = Field(default=0.8,description="Keyword weight default 0.8")
    keyword_fields: List[str] = Field(default=["video_name", "video_sub","video_keywords","video_knowledgePoints"], description="Fields to use for keyword search")




class Question(BaseModel):
    prompt: str = Field(default="", description="Prompt to generate a response")
    model: str = Field(default="llama3:8b", description="model to use in ollama")
    temperature: float = Field(default=0, description="Temperature for the model")


class Summary(BaseModel):
    title: str = Field(default="", description="Title of the summary")
    script: str = Field(default="", description="Summary of the article")
    

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
            'simpleSummary': video.get('video_simpleSummary', ''),
            'keywords': video.get('video_keywords',''),
            'knowledgePoints': video.get('video_knowledgePoints', ''),
            'summary': video.get('video_summary', ''),    
            'score': float(score),
            # "distance": float(distance),
            # "similarity": float(similarity),
        })
    return ranked_results


def keyword_search(query, vector_db, fields):
    """Perform keyword search and return matching scores"""
    query_keywords = set(jieba.cut(query.lower()))
    video_scores = []
    for video in vector_db:
        video_text = ' '.join([video[field] for field in fields]).lower()
        video_keywords = set(jieba.cut(video_text))
        common_keywords = query_keywords.intersection(video_keywords)

        keyword_score = len(common_keywords) / len(query_keywords)  # Simple ratio of matching keywords
        video_scores.append((video, keyword_score))
    
    return video_scores

async def combined_search(query, vector_db, vector_weight=0.5, keyword_weight=0.5, top_k=3, keyword_fields=[]):
    """Combine vector search and keyword search results"""
    query_embedding = get_embedding(query)
    # Perform vector search
    vector_scores = vector_search(query_embedding, vector_db, weight=vector_weight)
    
    # Perform keyword search
    keyword_scores = keyword_search(query, vector_db, fields=keyword_fields)

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
            'simpleSummary': video.get('video_simpleSummary', ''),
            'keywords': video.get('video_keywords',''),
            'knowledgePoints': video.get('video_knowledgePoints', ''),
            'summary': video.get('video_summary', ''),    
            'score': {
                'combined': float(combined_score),
                'keyword': float(keyword_score),
                'vector': float(vector_score),
            },
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

# Load vector databases
vector_db_files = {
    "znd": "znd456_vector_database.pkl",
    "hjb": "hjb_sx_1.2_vector_database.pkl",
    "znd_v2": "znd456_v2_vector_database.pkl"
}
vector_dbs = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_dbs
    for key, file in vector_db_files.items():
        vector_dbs[key] = await load_vector_database(file)
    print("==============================================")
    print("| Server is running and vector dbs has loaded|")
    print("==============================================")
    yield
    # Clean up the ML models and release the resources
    vector_dbs.clear()

app = FastAPI(lifespan=lifespan)
# 允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

@app.post(
        "/aisearch", 
        tags=["AI-Local-LLM-RAG"], 
        description="纯向量库匹配的视频推荐搜索，db_key 可以是 znd hjb znd_v2，其中znd_v2是带有关键知识点和关键字的版本，推荐使用", 
        summary="视频推荐，向量搜索", 
        response_description="Returns video list")
async def search(query: Query):
    if query.db_key not in vector_dbs:
        raise HTTPException(status_code=500, detail="Vector database not loaded")

    matches = await find_matching_videos(query.query, vector_dbs[query.db_key], top_k=query.top_k)
    return {"results": matches}

@app.post(
        "/mix/aisearch", 
        tags=["AI-Local-LLM-RAG"], 
        description='''基于向量库和关键字的视频推荐混合搜索，db_key 可以是 znd hjb znd_v2，其中znd_v2是带有关键知识点和关键字的版本，推荐使用，keyword_fields为关键字搜索字段名，例如keyword_fields=["video_name", "video_sub", "video_keywords","video_knowledgePoints"]''', 
        summary="视频推荐，混合搜索", response_description="Returns video list")
async def search(query: Query):
    if query.db_key not in vector_dbs:
        raise HTTPException(status_code=500, detail="Vector database not loaded")

    matches = await combined_search(query.query, vector_dbs[query.db_key], vector_weight=query.vector_weight, keyword_weight=query.keyword_weight, top_k=query.top_k, keyword_fields=query.keyword_fields)
    return {"results": matches}

@app.post("/aiquestion",tags=["AI-Local-LLM-RAG"],description="调用LLM回答问题", summary="AI问答", response_description="返回答案")
async def question(query: Question):
    res = await get_llm_response(query.prompt,query.model,temperature=query.temperature)
    return {"result": res}

@app.post("/aisummary",tags=["AI-Local-LLM-RAG"],description="调用LLM(llama3:8b)总结课程内容", summary="课程内容总结", response_description="返回json格式的summary,keywords[],knowledgePoints[]")
async def summary(query: Summary):
    res = await get_summary(query.title, query.script)
    return {"result": res}

@app.post("/coze/aisummary",tags=["AI-COZE-LLM-RAG"],description="调用coze总结课程内容", summary="课程内容总结", response_description="返回json格式的summary,keywords[],knowledgePoints[]")
async def summary_coze(query: Summary):
    res = await get_summary_coze(query.title, query.script)
    return {"result": res}






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
## run
## uvicorn fastapi-query:app --host 0.0.0.0 --port 8000 --work 2
