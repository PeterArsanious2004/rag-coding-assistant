# learn.py
from langchain_core.documents import Document
from chroma import add_docs
from generate import state


def looks_like_solution(text: str) -> bool:
    return "```" in text          


def learn(solution: str) -> str:
    pending = state["pending_query"]
    doc = Document(
        page_content=f"Request: {pending}\n\nSolution:\n{solution}",
        metadata={"source": "user_feedback", "request": pending[:300]},
    )
    n = add_docs([doc])           
    state["pending_query"] = None
    return f"Thanks! I stored your solution ({n} chunk(s)). Ask me again and I'll use it."