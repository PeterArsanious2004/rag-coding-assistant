# generate.py
from langchain_classic.schema import SystemMessage, HumanMessage
from chroma import search
from relevance import is_relevant

NOT_FOUND = ("I couldn't find relevant knowledge for your request. Could you provide "
             "the correct solution or reference so I can learn it for future interactions?")

SYSTEM = ("You are an expert software engineer. Write COMPLETE, runnable code in fenced "
          "Markdown blocks with a language tag, including comments, best practices and "
          "error handling. Add a short explanation after the code. "
          "Use this retrieved context where helpful:\n{context}")

# Remembers the request we failed to answer (used by the learning step)
state = {"pending_query": None}


def generate(llm, relevance_chain, query, memory_text=""):
    # Include a bit of history so follow-ups like "add pagination to it" still retrieve well
    docs = search(query)
    context = "\n\n".join(d.page_content for d in docs)

    if not is_relevant(relevance_chain, query, context):
        state["pending_query"] = query
        yield NOT_FOUND
        return

    state["pending_query"] = None
    system = SYSTEM.format(context=context)
    if memory_text:
        system += f"\n\nConversation so far:\n{memory_text}"

    messages = [SystemMessage(content=system), HumanMessage(content=query)]
    for chunk in llm.stream(messages):
        yield chunk.content