# app.py
import re

import streamlit as st

from chroma import add_docs
from classifier import build_classifier_chain, classify
from code_runner import run_python
from explain import explain
from generate import generate, state
from learn import learn, looks_like_solution
from llm import get_llm
from loader import load
from memory import clear_history, get_history, save_turn
from relevance import build_relevance_chain

st.set_page_config(page_title="AI Coding Assistant", layout="wide")

FENCE = "`" * 3
CODE_RE = re.compile(FENCE + r"(\w+)?\n(.*?)" + FENCE, re.S)


# ---------- build the models once ----------
@st.cache_resource
def get_components():
    return (
        get_llm(),                          # answers
        build_classifier_chain(get_llm()),    # intent classifier
        build_relevance_chain(get_llm()),     # relevance evaluator
    )


llm, classifier_chain, relevance_chain = get_components()

# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []      # {"role", "content", "route"}
if "run_results" not in st.session_state:
    st.session_state.run_results = {}   # button key -> result dict


def one_chunk(text):
    yield text


# ---------- sidebar: file upload + memory ----------
with st.sidebar:
    st.header("📁 Files")
    files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["txt", "md", "py", "js", "java", "pdf"],
    )
    if files:
        c1, c2 = st.columns(2)
        if c1.button("Add to KB"):
            with st.spinner("Loading, chunking, embedding..."):
                n = add_docs(load.load_all_docs(files))
            st.success(f"Stored {n} chunks in Chroma")
        if c2.button("Attach to chat"):
            parts = []
            for f in files:
                if not f.name.lower().endswith(".pdf"):
                    parts.append(f"\n\nFile `{f.name}`:\n{FENCE}\n"
                                 f"{f.getvalue().decode('utf-8', errors='ignore')}\n{FENCE}")
            st.session_state.attached = "".join(parts)
            st.success("Attached to your next message")

    st.header("🧠 Memory")
    with st.expander("Conversation memory", expanded=False):
        st.text(get_history() or "(empty)")
    if state["pending_query"]:
        st.info(f"Waiting for a solution to:\n\n{state['pending_query']}")

    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.session_state.run_results = {}
        state["pending_query"] = None
        clear_history()
        st.rerun()

# ---------- main: chat history ----------
st.title("💻 AI Coding Assistant")

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if m.get("route"):
            st.caption(f"Route: {m['route']}")
        st.markdown(m["content"])     # fenced code blocks are syntax-highlighted

        if m["role"] == "assistant":
            for j, (lang, code) in enumerate(CODE_RE.findall(m["content"])):
                if (lang or "python").lower() not in ("python", "py"):
                    continue
                key = f"run_{i}_{j}"
                if st.button("▶ Run code", key=key):
                    with st.spinner("Running..."):
                        st.session_state.run_results[key] = run_python(code)
                res = st.session_state.run_results.get(key)
                if res:
                    status = "✅ Success" if res["returncode"] == 0 else "❌ Error"
                    st.caption(f"{status} (exit code {res['returncode']})")
                    if res["stdout"]:
                        st.code(res["stdout"], language="text")
                    if res["stderr"]:
                        st.code(res["stderr"], language="text")

# ---------- main: new message ----------
if prompt := st.chat_input("Ask me to write or explain code..."):
    attached = st.session_state.pop("attached", "")
    full_prompt = prompt + attached

    st.session_state.messages.append({"role": "user", "content": full_prompt})
    with st.chat_message("user"):
        st.markdown(full_prompt)

    with st.chat_message("assistant"):
        history = get_history()

        if state["pending_query"] and looks_like_solution(full_prompt):
            route = "LEARN"
            stream = one_chunk(learn(full_prompt))
        else:
            state["pending_query"] = None
            route = classify(classifier_chain,
                             f"{history}\n{full_prompt}" if history else full_prompt)
            if route == "EXPLAIN":
                stream = explain(llm, full_prompt, history)
            else:
                stream = generate(llm, relevance_chain, full_prompt, history)

        st.caption(f"Route: {route}")
        answer = st.write_stream(stream)      # streaming display

    save_turn(full_prompt, answer)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "route": route}
    )
    st.rerun()   # redraw so the Run buttons and memory panel appear