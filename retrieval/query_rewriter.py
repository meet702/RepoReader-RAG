from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

def rewrite_query(user_question: str, chat_history: list) -> str:
    if not chat_history:
        return user_question
        
    llm = ChatOllama(model="qwen2.5-coder:7b")
    
    messages = [
        SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
    ] + chat_history + [
        HumanMessage(content=f"New question: {user_question}")
    ]
    
    result = llm.invoke(messages)
    search_question = result.content.strip()
    return search_question
