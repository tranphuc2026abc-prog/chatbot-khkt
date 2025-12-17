import os
import glob
import base64
import streamlit as st
import shutil
import re
import uuid
import time
from typing import List, Generator

# --- Imports & Error Handling ---
try:
    import nest_asyncio
    nest_asyncio.apply() # Fix lỗi loop của LlamaParse
    from llama_parse import LlamaParse 
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG) 
# ==============================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    LLM_MODEL = 'llama-3.1-8b-instant' # Tốc độ cao, phù hợp chatbot realtime
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"

    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt"
    PROCESSED_MD_DIR = "PROCESSED_MD" 

    # Assets (Nếu không có file ảnh, hệ thống sẽ dùng icon mặc định)
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    # RAG Parameters
    RETRIEVAL_K = 20       # Lấy rộng để lọc
    FINAL_K = 4            # Chỉ đưa 4 context tốt nhất vào LLM
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      
    FAISS_WEIGHT = 0.6     

    LLM_TEMPERATURE = 0.3 # Thấp để ổn định, tránh "chém gió"

# ===============================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER) 
# ===============================

class UIManager:
    @staticmethod
    def get_img_as_base64(file_path):
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    @staticmethod
    def inject_custom_css():
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
            html, body, [class*="css"], .stMarkdown {
                font-family: 'Inter', sans-serif !important;
            }
            /* Giao diện Header */
            .main-header {
                background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%);
                padding: 1.5rem 2rem; border-radius: 15px; color: white;
                margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0, 119, 182, 0.3);
                display: flex; align-items: center; justify-content: space-between;
            }
            .header-left h1 {
                color: #caf0f8 !important; font-weight: 900; margin: 0; font-size: 2rem;
            }
            .header-left p { color: #e0fbfc; margin: 5px 0 0 0; }
            
            /* Giao diện Project Card bên Sidebar */
            .project-card {
                background: white; padding: 15px; border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }
            .project-title {
                color: #0077b6; font-weight: 800; font-size: 1.1rem;
                text-align: center; text-transform: uppercase; margin-bottom: 10px;
            }

            /* Badge hiển thị nguồn */
            .source-badge {
                display: inline-flex; align-items: center;
                padding: 4px 10px; border-radius: 20px;
                font-size: 0.75rem; font-weight: 600; color: white;
                margin-right: 8px; margin-top: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .source-badge:hover { transform: translateY(-2px); }
            
            /* Chat Message */
            [data-testid="stChatMessageContent"] {
                border-radius: 15px !important; padding: 1rem !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            if os.path.exists(AppConfig.LOGO_SCHOOL):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(AppConfig.LOGO_SCHOOL, use_container_width=True)
                st.markdown("<div style='text-align:center; font-weight:700; color:#023e8a; margin-bottom:20px;'>THCS & THPT PHẠM KIỆT</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="project-card">
                <div class="project-title">KTC CHATBOT</div>
                <div style="font-size: 0.85rem; color: #555; text-align: center;">
                    <i>Trợ lý ảo hỗ trợ học tập môn Tin học<br>theo định hướng CT GDPT 2018</i>
                </div>
                <hr style="margin: 10px 0; border-top: 1px dashed #dee2e6;">
                <div style="font-size: 0.9rem; line-height: 1.6;">
                    <b>👨‍💻 Tác giả:</b> Bùi Tá Tùng - Cao Sỹ Bảo Chung<br>
                    <b>👨‍🏫 GVHD:</b> Thầy Nguyễn Thế Khanh
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### ⚙️ Tiện ích")
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            if st.button("🔄 Cập nhật dữ liệu SGK", use_container_width=True):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop('retriever_engine', None)
                st.toast("Đã xóa cache dữ liệu. Vui lòng reload lại trang!", icon="✅")
                time.sleep(1)
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" style="width:100px; height:100px; border-radius:50%; border:3px solid rgba(255,255,255,0.3); box-shadow:0 4px 10px rgba(0,0,0,0.2); object-fit:cover;">' if logo_nhom_b64 else ""

        st.markdown(f"""
        <div class="main-header">
            <div class="header-left">
                <h1>KTC CHATBOT</h1>
                <p>Hỏi đáp Tin học - Chuẩn kiến thức SGK Kết nối tri thức</p>
            </div>
            <div class="header-right">
                {img_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================================
# 3. LOGIC BACKEND (RAG ENGINE)
# ==================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key: return None
            return Groq(api_key=api_key)
        except Exception: return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        return HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except: return None

    # --- Xử lý tách đoạn văn bản (Chunking) theo cấu trúc SGK ---
    @staticmethod
    def _structural_chunking(text: str, source_meta: dict) -> List[Document]:
        lines = text.split('\n')
        chunks = []
        
        current_context = {"chapter": "Mở đầu", "lesson": "Tổng quan"}
        buffer = []

        # Regex đơn giản hóa để bắt tiêu đề
        p_chapter = re.compile(r'^(CHƯƠNG|Chương)\s+[0-9IVX]+', re.IGNORECASE)
        p_lesson = re.compile(r'^(BÀI|Bài)\s+[0-9]+', re.IGNORECASE)

        def commit_chunk(buf, meta, ctx):
            content = "\n".join(buf).strip()
            if len(content) > 50:
                new_meta = meta.copy()
                new_meta.update({
                    "chunk_id": str(uuid.uuid4())[:8],
                    "chapter": ctx["chapter"],
                    "lesson": ctx["lesson"]
                })
                chunks.append(Document(page_content=content, metadata=new_meta))

        for line in lines:
            line = line.strip()
            if not line: continue
            
            if p_chapter.match(line):
                commit_chunk(buffer, source_meta, current_context)
                buffer = [line]
                current_context["chapter"] = line
            elif p_lesson.match(line):
                commit_chunk(buffer, source_meta, current_context)
                buffer = [line]
                current_context["lesson"] = line
            else:
                buffer.append(line)
        
        commit_chunk(buffer, source_meta, current_context)
        return chunks

    @staticmethod
    def _parse_pdf_with_llama(file_path: str) -> str:
        # Check cache Markdown đã xử lý chưa để đỡ tốn API
        os.makedirs(AppConfig.PROCESSED_MD_DIR, exist_ok=True)
        file_name = os.path.basename(file_path)
        md_file_path = os.path.join(AppConfig.PROCESSED_MD_DIR, f"{file_name}.md")
        
        if os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # Nếu chưa có cache thì gọi LlamaParse
        key = st.secrets.get("LLAMA_CLOUD_API_KEY")
        if not key: return ""

        try:
            parser = LlamaParse(api_key=key, result_type="markdown", language="vi")
            docs = parser.load_data(file_path)
            if docs:
                with open(md_file_path, "w", encoding="utf-8") as f:
                    f.write(docs[0].text)
                return docs[0].text
        except: pass
        return ""

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None

        # Load VectorDB từ đĩa nếu có
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
                return db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
            except: pass

        # Nếu chưa có, quét thư mục PDF để tạo mới
        if not os.path.exists(AppConfig.PDF_DIR):
            os.makedirs(AppConfig.PDF_DIR)
            return None
        
        files = glob.glob(os.path.join(AppConfig.PDF_DIR, "*.pdf"))
        if not files: return None
        
        all_chunks = []
        progress_text = st.empty()
        
        for f in files:
            progress_text.text(f"Đang số hóa tri thức: {os.path.basename(f)}...")
            txt = RAGEngine._parse_pdf_with_llama(f)
            if txt:
                chunks = RAGEngine._structural_chunking(txt, {"source": os.path.basename(f)})
                all_chunks.extend(chunks)
        
        progress_text.empty()
        
        if all_chunks:
            db = FAISS.from_documents(all_chunks, embeddings)
            db.save_local(AppConfig.VECTOR_DB_PATH)
            return db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})
        return None

    # --- CORE: HÀM SINH CÂU TRẢ LỜI & HIỂN THỊ NGUỒN ---
    @staticmethod
    def generate_response(client, retriever, query) -> Generator[str, None, None]:
        if not retriever:
            yield "Dữ liệu đang được tải, bạn chờ một chút nhé..."
            return

        # 1. Truy xuất dữ liệu thô
        docs = retriever.invoke(query)
        
        # 2. Chấm điểm ưu tiên (SGK > SGV > Code)
        scored_docs = []
        for doc in docs:
            src = doc.metadata.get('source', '')
            score = 0.0
            if "KNTT" in src or "SGK" in src: score = 1.0 # Ưu tiên cao nhất
            elif "GV" in src: score = 0.5
            scored_docs.append({"doc": doc, "bonus": score})
        
        # 3. Rerank (Sắp xếp lại bằng AI)
        final_docs = []
        try:
            ranker = RAGEngine.load_reranker()
            if ranker and scored_docs:
                passages = [{"id": str(i), "text": x["doc"].page_content, "meta": x["doc"].metadata} for i, x in enumerate(scored_docs)]
                req = RerankRequest(query=query, passages=passages)
                results = ranker.rank(req)
                
                # Tính điểm cuối = Điểm AI + Điểm Bonus
                reranked = []
                for res in results:
                    idx = int(res['id'])
                    final_score = res['score'] + (scored_docs[idx]['bonus'] * 0.3)
                    reranked.append({"res": res, "score": final_score})
                
                reranked.sort(key=lambda x: x['score'], reverse=True)
                final_docs = [Document(page_content=r['res']['text'], metadata=r['res']['meta']) for r in reranked[:AppConfig.FINAL_K]]
            else:
                # Fallback nếu không có Ranker
                scored_docs.sort(key=lambda x: x['bonus'], reverse=True)
                final_docs = [x["doc"] for x in scored_docs[:AppConfig.FINAL_K]]
        except:
            final_docs = [x["doc"] for x in scored_docs[:AppConfig.FINAL_K]]

        if not final_docs:
            yield "Xin lỗi, hiện tại trong CSDL SGK chưa có thông tin về vấn đề này."
            return

        # 4. Chuẩn bị Context và Nguồn hiển thị (Badge)
        context_text = ""
        source_badges_html = ""
        seen_sources = set()

        for doc in final_docs:
            context_text += f"---\nNội dung: {doc.page_content}\n"
            
            # Xử lý hiển thị Badge
            src_raw = doc.metadata.get('source', 'Tài liệu')
            lesson = doc.metadata.get('lesson', '').replace('Bài', 'B.').strip()
            
            # Logic màu sắc
            if "KNTT" in src_raw or "SGK" in src_raw:
                color, icon, lbl = "#0077b6", "📘", "SGK Tin học" # Blue
            elif "GV" in src_raw:
                color, icon, lbl = "#d35400", "📙", "SGV Tin học" # Orange
            elif "Python" in src_raw:
                color, icon, lbl = "#27ae60", "🐍", "Code Python" # Green
            else:
                color, icon, lbl = "#7f8c8d", "📄", "Tài liệu khác" # Grey
            
            uid = f"{lbl}-{lesson}"
            if uid not in seen_sources:
                source_badges_html += f"""
                <span class="source-badge" style="background-color: {color};">
                    {icon} {lbl} > {lesson}
                </span>
                """
                seen_sources.add(uid)

        # 5. Gọi LLM
        sys_prompt = f"""Bạn là Trợ lý ảo KTC, chuyên gia về Tin học THPT (SGK Kết nối tri thức).
Dựa vào ngữ cảnh sau:
{context_text}

Hãy trả lời câu hỏi của học sinh: "{query}"
Yêu cầu:
- Trả lời ngắn gọn, dễ hiểu, sư phạm.
- Nếu là code Python, hãy giải thích từng dòng.
- TUYỆT ĐỐI KHÔNG tự bịa ra nguồn tài liệu.
"""
        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[{"role": "system", "content": sys_prompt}],
                stream=True,
                temperature=AppConfig.LLM_TEMPERATURE
            )
            
            # Stream nội dung text trước
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            # Cuối cùng yield HTML badge nguồn
            if source_badges_html:
                yield f"\n\n<div style='margin-top:10px; padding-top:10px; border-top:1px dashed #ccc;'>{source_badges_html}</div>"
                
        except Exception as e:
            yield f"Đang gặp sự cố kết nối AI: {str(e)}"

# ===================
# 4. MAIN APPLICATION
# ===================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Lỗi thư viện: {IMPORT_ERROR}")
        st.info("Gợi ý: Kiểm tra file requirements.txt (cần: langchain, groq, flashrank, llama-parse, ...)")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    # Khởi tạo session
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Chatbot. Bạn cần hỗ trợ kiến thức bài nào trong SGK?"}]

    groq_client = RAGEngine.load_groq_client()

    # Khởi tạo Retriever (Chạy 1 lần)
    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động động cơ tri thức số..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
    
    # Render lịch sử chat
    bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑‍🎓" if role == "user" else bot_avatar
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Xử lý input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=bot_avatar):
            response_placeholder = st.empty()
            full_response = ""
            
            # Gọi Generator
            response_gen = RAGEngine.generate_response(
                groq_client,
                st.session_state.retriever_engine,
                prompt
            )
            
            # Streaming Loop
            for chunk in response_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()