# classifier.py
from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the request. Reply with exactly one word:\n"
               "EXPLAIN = explain/review/debug existing code\n"
               "GENERATE = write or modify code"),
    ("human", "{query}"),
])


def build_classifier_chain(llm) -> LLMChain:
    return LLMChain(llm=llm, prompt=prompt, output_key="label")


def classify(chain: LLMChain, query: str) -> str:
    out = chain.invoke({"query": query})["label"]
    return "EXPLAIN" if "EXPLAIN" in out.upper() else "GENERATE"