# relevance.py
from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Decide whether ANY part of the retrieved context helps answer the request. "
           "If at least one chunk directly addresses the request, reply RELEVANT. "
           "Otherwise reply NOT_RELEVANT. Reply with the label only."),
    ("human", "Request:\n{query}\n\nContext:\n{context}"),
])


def build_relevance_chain(llm):
    return LLMChain(llm=llm, prompt=prompt, output_key="verdict")

def is_relevant(chain, query, context) -> bool:
    if not context.strip():
        return False
    for _ in range(3):
        out = chain.invoke({"query": query, "context": context})["verdict"].upper()
        print(f"[relevance verdict] {out!r}")        # shows in the terminal running streamlit
        if "NOT_RELEVANT" in out or "NOT RELEVANT" in out:
            return False
        if "RELEVANT" in out:
            return True
    return False