import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # must match ingest.py
LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")    # Groq model names change; check console.groq.com/docs/models
DB_DIR = "chroma_db"
COLLECTION = "legal_cases"
TOP_K = 5

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a legal research assistant. Answer the question using ONLY the numbered "
     "context passages below, which come from Indian Supreme Court judgments.\n"
     "Rules:\n"
     "- After each claim, cite the passage it came from, like [1] or [2].\n"
     "- If the passages do not contain the answer, say exactly: "
     "\"I couldn't find this in the provided judgments.\" Do not guess.\n"
     "- Do not use outside knowledge. Keep the answer clear and concise.\n"
     "- This is research support, not legal advice."),
    ("human", "Context passages:\n{context}\n\nQuestion: {question}"),
])


@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )


@st.cache_resource
def load_llm():
    return ChatGroq(model=LLM_MODEL, temperature=0)


def format_context(docs):
    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[{i}] (File: {d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}")
    return "\n\n".join(parts)


st.set_page_config(page_title="Legal Case Research Assistant", page_icon="⚖️")
st.title("⚖️ Legal Case Research Assistant")
st.caption("Ask questions about Indian Supreme Court judgments. Answers are grounded in the retrieved passages and cite their sources.")

if not Path(DB_DIR).exists():
    st.error("chroma_db folder not found. Run `python ingest.py` first.")
    st.stop()
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file and save it.")
    st.stop()

question = st.text_input("Your question", placeholder="e.g. What did the court hold about the time limit for filing a written statement?")

if question:
    db = load_db()
    llm = load_llm()

    with st.spinner("Searching judgments..."):
        docs = db.max_marginal_relevance_search(question, k=TOP_K, fetch_k=20)

    try:
        with st.spinner("Writing answer..."):
            messages = PROMPT.format_messages(context=format_context(docs), question=question)
            answer = llm.invoke(messages).content
    except Exception as e:
        st.error(f"The LLM call failed: {e}")
        st.stop()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, d in enumerate(docs, start=1):
        with st.expander(f"[{i}] {d.metadata['source']} - page {d.metadata['page']}"):
            st.write(d.page_content)