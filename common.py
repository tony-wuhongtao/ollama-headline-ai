import json
import requests
import re

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api"

def get_embedding(text):
    """Get embedding for a given text using Ollama API"""
    response = requests.post(f"{OLLAMA_API}/embeddings", json={
        "model": "milkey/m3e:latest",
        # "model": "chatfire/bge-m3:q8_0",
        "prompt": text
    })
    return response.json()['embedding']

async def get_llm_response(prompt,model="llama3:8b",temperature=0):
    """Get LLM response using Ollama API"""
    try:
        response = requests.post(f"{OLLAMA_API}/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        })
        # response.raise_for_status()

        # full_response = ""
        # for line in response.iter_lines():
        #     if line:
        #         data = json.loads(line)
        #         if 'response' in data:
        #             full_response += data['response']

        # # 尝试从响应中提取 JSON
        # json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
        # if json_match:
        #     try:
        #         result = json.loads(json_match.group())
        #         return result
        #     except json.JSONDecodeError:
        #         print(f"Failed to parse extracted JSON: {json_match.group()}")
        # else:
        #     print(f"No JSON found in response: {full_response}")

        # 如果无法解析 JSON，尝试直接从文本中提取评分和解释
        # rating_match = re.search(r'"rating":\s*(\d+)', full_response)
        # explanation_match = re.search(r'"explanation":\s*"([^"]*)"', full_response)
        
        # rating = int(rating_match.group(1)) if rating_match else 5
        # explanation = explanation_match.group(1) if explanation_match else "Based on similarity score (parsing failed)"

        # return {"rating": rating, "explanation": explanation}
        res_json = response.json()
        response_text = res_json.get("response")
    
        return response_text

    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return {"error": f"Error: API request failed - {e}"}

def chunk_text(text, chunk_size=600):
    """Split text into chunks"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]