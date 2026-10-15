# explain.py
from langchain_classic.schema import SystemMessage, HumanMessage

SYSTEM_PROMPT = "You are a programming tutor. Explain the user's code clearly."


def explain(llm, query, memory_text=""):
    system = SYSTEM_PROMPT
    if memory_text:
        system += f"\n\nConversation so far:\n{memory_text}"

    messages = [SystemMessage(content=system), HumanMessage(content=query)]
    for chunk in llm.stream(messages):
        yield chunk.content