from common import get_llm_response


async def get_summary(title: str, script: str):
    Title = title
    Script = script

    # prompt = f"""
    # <s>[INST] <<SYS>>
    # You are an expert in course content summary tasks. 
    # Generate Chinese Simplified Chinese course content text within 250 words and within 5 knowledge points and content keywords based on the input course title and script. 
    # All output must be in valid JSON. Don't add explanation beyond the JSON.  
    # Please ensure that your reply is concise and fluent, contains all the knowledge points and points, does not need to be formatted, but needs to be logical and clear.
    # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    # If you don't know the answer, please don't share false information.
    # <</SYS>>
    # Based on the audio script of this lesson, what is the content of this lesson? 
    # Output must be in valid JSON like the following example {{"keyWords": [数学,因数], "summary": "in_less_than_250_words"}}. Output must include only JSON.
    # Title: {Title}
    # Script: {Script}
    # [/INST]
    # """
    # prompt = f"""
    # <s>[INST] <<SYS>>
    # 你是一个课程内容总结专家. 
    # 基于输入的课程标题和脚本, 生成50字以内的课程内容总结。
    # 确保你的回答极度简洁，包含所有内容要点，包含所有的课程难点和重点，以关键字或简短的句子输出。
    # 如果一个问题没有意义，或者不是事实性的连贯，解释为什么而不是回答不正确的内容。
    # 如果你不知道答案，请不要分享错误信息。
    # 用中文回答我。
    # <</SYS>>
    # 这个课程的内容是什么？输出必须为有效的JSON，如以下示例 ,{{"summary": ""}}
    # 输出必须仅包含JSON。中文回答。
    # Title: {Title}Script: {Script}[/INST]
    # """
    prompt = f"""
    你是一个课程内容总结专家. 基于输入的课程标题{Title}和脚本{Script}, 生成200字以内的课程内容总结。
    确保你的回答极度简洁，包含所有内容要点，包含所有的课程难点和重点，以关键字或简短的句子输出。
    如果一个问题没有意义，或者不是事实性的连贯，解释为什么而不是回答不正确的内容。
    如果你不知道答案，请不要分享错误信息。
    不要以"总结:"开头,直接回复内容。输出json格式。
    {{
    "summary": "总结内容",
    "keywords": ["关键词1", "关键词2"],
    "knowledgePoints": ["知识点1", "知识点2"]    
    }}
    用中文回答,用中文回答,用中文回答
    """
    res = await get_llm_response(prompt, model="llama3:8b", temperature=0.1)
    print(res)
    return res
