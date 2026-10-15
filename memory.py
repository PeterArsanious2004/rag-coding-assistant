from langchain_classic.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5, memory_key="history", input_key="input")


def get_history() -> str:
    return memory.load_memory_variables({}).get("history", "")


def save_turn(query: str, answer: str):
    memory.save_context({"input": query}, {"output": answer})


def clear_history():
    memory.clear()