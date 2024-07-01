import requests
import json


BOT_ID="7385003212933283894"
USER = "ollamadev"
CONVERSATION_ID = "summary"
COZE_TOKEN = "pat_hJtFveZva0nIRLUY70HQfWLXNE8yCKLMpCLjOqACusT2K4E0SBQs73V41mL1IkP1"
COZE_URL = "https://api.coze.cn/open_api/v2/chat"

async def get_summary_coze(title:str, script:str):
    query = {
        "title": title,
        "script": script
    }

    data = {
        "conversation_id": CONVERSATION_ID,
        "bot_id": BOT_ID,
        "user": USER,
        "query": json.dumps(query),
        "stream": False
    }

    try:
        res = requests.post(COZE_URL, json=data, headers={
            'Accept': '*/*',
            'Host': 'api.coze.cn',
            'Connection': 'keep-alive',
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f'Bearer {COZE_TOKEN}'
        })

        res_json = res.json()
        content = res_json["messages"][0]["content"]
        content_json = json.loads(content)

        return content_json


    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return {"error": f"Error: API request failed - {e}"}

