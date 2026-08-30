from langchain_core.messages import HumanMessage, SystemMessage
from llm.provider import get_llm, message_content_to_text

def rewrite_query(user_question: str, chat_history: list) -> str:
    if not chat_history:
        return user_question
        
    llm = get_llm()
    
    messages = [
        SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
    ] + chat_history + [
        HumanMessage(content=f"New question: {user_question}")
    ]
    
    result = llm.invoke(messages)
    search_question = message_content_to_text(result.content).strip()
    return search_question
