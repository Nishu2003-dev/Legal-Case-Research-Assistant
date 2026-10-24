"""search_test.py - quick check that retrieval works before you add the LLM."""
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)
db = Chroma(
    collection_name="legal_cases",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

question = " ".join(sys.argv[1:]) or "What did the court say about the limitation period?"
for doc, score in db.similarity_search_with_score(question, k=4):
    print(f"\n[{doc.metadata['source']} | page {doc.metadata['page']} | distance {score:.3f}]")
    print(doc.page_content[:400], "...")