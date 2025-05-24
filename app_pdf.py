import os
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv(".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_embedding_and_qdrant():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_size = len(embedding_model.embed_query("test"))
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name="documents",
        vectors_config={"size": vector_size, "distance": "Cosine"},
    )
    return qdrant, embedding_model
@st.cache_data
def load_and_add_documents(_qdrant, _embedding_model):
    
    # เปลี่ยนเส้นทางไปยังไฟล์ PDF ที่ต้องการ ตัวอย่าง ข้อมูลยา 50 ชนิด
    source = "./pdf/summer_nan.pdf"
    converter = DocumentConverter()
    result = converter.convert(source)
    markdown_text = result.document.export_to_markdown()
    doc = Document(page_content=markdown_text)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = splitter.split_documents([doc])

    texts = [d.page_content for d in document_chunks]
    vectors = _embedding_model.embed_documents(texts)
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"text": texts[i]})
        for i in range(len(texts))
    ]
    _qdrant.upsert(collection_name="documents", points=points)
    
def search_documents(query, qdrant, embedding_model):
    query_vector = embedding_model.embed_query(query)
    search_results = qdrant.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=4,
    )
    return [hit.payload.get("text", "เอกสารไม่มีข้อความ") for hit in search_results] if search_results else []
def generate_answer(query, qdrant, embedding_model):
    retrieved_docs = search_documents(query, qdrant, embedding_model)
    if not retrieved_docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    context = "\n".join([doc for doc in retrieved_docs if isinstance(doc, str)])
    if not context.strip():
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"

    # กำหนด prompt สำหรับการสร้างคำตอบ
    system_prompt = "คุณคือ AI ผู้ช่วยตอบคำถามเกี่ยวกับเอกสาร"
    
    # สร้าง prompt สำหรับการถามคำถาม
    prompt = f"{system_prompt}\n\nข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}\n\nคำตอบ:"
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 AI Innovator LLM & RAG")
    st.subheader("Chatbot ช่วยตอบคำถามเกี่ยวกันแผนการท่องเที่ยวตามฤดูกาลในจังหวัดน่าน")

    qdrant, embedding_model = load_embedding_and_qdrant()
    load_and_add_documents(qdrant, embedding_model)
    st.success("✅ ข้อมูลเอกสารพร้อมใช้งานแล้ว!")

    query = st.text_input("คุณ:", placeholder="พิมพ์คำถามของคุณที่นี่...")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "สวัสดี มีอะไรให้ช่วยไหม :)"}
        ]

    if st.button("ส่ง"):
        if query:
            answer = generate_answer(query, qdrant, embedding_model)
            st.session_state["messages"].append({"role": "user", "content": query})
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        else:
            st.warning("กรุณาพิมพ์คำถามก่อนส่ง")

    for msg in st.session_state["messages"]:
        role = "Bot" if msg["role"] == "assistant" else "คุณ"
        st.write(f"**{role}:** {msg['content']}")

if __name__ == "__main__":
    main()
