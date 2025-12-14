import os
import glob
import base64
import streamlit as st
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator

# --- Imports với xử lý lỗi ---
try:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from groq import Groq
    # Rerank optimization
    from flashrank import Ranker, RerankRequest
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG) - TỐI ƯU CHO KHKT
# ==============================================================================

st.set_page_config(
    page_title="KTC Chatbot - THCS & THPT Phạm Kiệt",
    page_icon="LOGO.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AppConfig:
    # Model Config
    LLM_MODEL = 'llama-3.1-8b-instant'
    LLM_VISION_MODEL = 'llama-3.2-11b-vision-preview'
    LLM_AUDIO_MODEL = 'whisper-large-v3'

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"

    # Paths
    PDF_DIR = "PDF_KNOWLEDGE"
    VECTOR_DB_PATH = "faiss_db_index"
    RERANK_CACHE = "./opt"

    # Assets
    LOGO_PROJECT = "LOGO.jpg"
    LOGO_SCHOOL = "LOGO PKS.png"

    # RAG Parameters (Tối ưu cho sách giáo khoa/tài liệu kỹ thuật)
    CHUNK_SIZE = 800       # Tăng nhẹ để giữ trọn vẹn ngữ cảnh kỹ thuật
    CHUNK_OVERLAP = 150    # Overlap để không mất nối từ
    RETRIEVAL_K = 30       # Lấy rộng để BM25 bắt từ khóa
    FINAL_K = 5            # Chỉ lấy 5 đoạn tốt nhất cho LLM đọc để tránh nhiễu
    
    # Hybrid Search Weights
    BM25_WEIGHT = 0.4      # Ưu tiên từ khóa chính xác
    FAISS_WEIGHT = 0.6     # Ưu tiên ngữ nghĩa

    # LLM Generation
    LLM_TEMPERATURE = 0.1  # Giảm nhiệt độ tối đa để tăng tính chính xác (Academic)
    LLM_MAX_TOKENS = 1500

# ==============================================================================
# 2. XỬ LÝ GIAO DIỆN (UI MANAGER) - [GIỮ NGUYÊN 100% THEO YÊU CẦU]
# ==============================================================================

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
            html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stChatInput {
                font-family: 'Inter', sans-serif !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #f8f9fa; border-right: 1px solid #e9ecef;
            }
            .project-card {
                background: white; padding: 15px; border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }
            .project-title {
                color: #0077b6; font-weight: 800; font-size: 1.1rem;
                margin-bottom: 5px; text-align: center; text-transform: uppercase;
            }
            .project-sub {
                font-size: 0.8rem; color: #6c757d; text-align: center;
                margin-bottom: 15px; font-style: italic;
            }
            .main-header {
                background: linear-gradient(135deg, #023e8a 0%, #0077b6 100%);
                padding: 1.5rem 2rem; border-radius: 15px; color: white;
                margin-bottom: 2rem; box-shadow: 0 8px 20px rgba(0, 119, 182, 0.3);
                display: flex; align-items: center; justify-content: space-between;
            }
            .header-left h1 {
                color: #caf0f8 !important; font-weight: 900; margin: 0;
                font-size: 2.2rem; letter-spacing: -0.5px;
            }
            .header-left p {
                color: #e0fbfc; margin: 5px 0 0 0; font-size: 1rem; opacity: 0.9;
            }
            .header-right img {
                border-radius: 50%; border: 3px solid rgba(255,255,255,0.3);
                box-shadow: 0 4px 10px rgba(0,0,0,0.2); width: 100px; height: 100px;
                object-fit: cover;
            }
            [data-testid="stChatMessageContent"] {
                border-radius: 15px !important; padding: 1rem !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
                background: #e3f2fd; color: #0d47a1;
            }
            [data-testid="stChatMessageContent"]:not(:has(+ [data-testid="stChatMessageAvatar"])) {
                background: white; border: 1px solid #e9ecef;
                border-left: 5px solid #00b4d8;
            }
            div.stButton > button {
                border-radius: 8px; background-color: white; color: #0077b6;
                border: 1px solid #90e0ef; transition: all 0.2s;
            }
            div.stButton > button:hover {
                background-color: #0077b6; color: white;
                border-color: #0077b6; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
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
                <div class="project-sub">Sản phẩm dự thi KHKT cấp trường</div>
                <hr style="margin: 10px 0; border-top: 1px dashed #dee2e6;">
                <div style="font-size: 0.9rem; line-height: 1.6;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600; color: #555;">Tác giả:</span>
                        <span style="text-align: right; color: #222;"><b>Bùi Tá Tùng</b><br><b>Cao Sỹ Bảo Chung</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-weight: 600; color: #555;">GVHD:</span>
                        <span style="text-align: right; color: #222;">Thầy <b>Nguyễn Thế Khanh</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-weight: 600; color: #555;">Năm học:</span>
                        <span style="text-align: right; color: #222;"><b>2025 - 2026</b></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📂 Tính năng nâng cao (AI Vision)", expanded=False):
                st.markdown("<small>Tải ảnh lỗi code hoặc file ghi âm câu hỏi</small>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'mp3', 'wav', 'py'], key="multimodal_upload")
                if uploaded_file:
                    st.session_state.uploaded_file_obj = uploaded_file
                    st.success("Đã nhận file!")

            st.markdown("### ⚙️ Tiện ích")
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.uploaded_file_obj = None
                st.rerun()

            if st.button("🔄 Cập nhật dữ liệu mới", use_container_width=True):
                if os.path.exists(AppConfig.VECTOR_DB_PATH):
                    shutil.rmtree(AppConfig.VECTOR_DB_PATH)
                st.session_state.pop('retriever_engine', None)
                st.rerun()

    @staticmethod
    def render_header():
        logo_nhom_b64 = UIManager.get_img_as_base64(AppConfig.LOGO_PROJECT)
        img_html = f'<img src="data:image/jpeg;base64,{logo_nhom_b64}" alt="Logo">' if logo_nhom_b64 else ""

        st.markdown(f"""
        <div class="main-header">
            <div class="header-left">
                <h1>KTC CHATBOT</h1>
                <p style="font-size: 1.1rem; margin-top: 5px;">Học Tin dễ dàng - Thao tác vững vàng</p>
            </div>
            <div class="header-right">
                {img_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC BACKEND - TÁI CẤU TRÚC CHO KHKT
# ==============================================================================

class RAGEngine:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_groq_client():
        try:
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key:
                return None
            return Groq(api_key=api_key)
        except Exception:
            return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        # Dùng model MiniLM chuẩn cho tiếng Việt/Đa ngữ
        try:
            return HuggingFaceEmbeddings(
                model_name=AppConfig.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Lỗi tải Embedding: {e}")
            return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_reranker():
        # ⚠ Fix: Đảm bảo flashrank khởi tạo đúng và có cache
        try:
            return Ranker(model_name=AppConfig.RERANK_MODEL_NAME, cache_dir=AppConfig.RERANK_CACHE)
        except Exception as e:
            st.warning(f"Lỗi tải Reranker (sẽ chạy chế độ thường): {e}")
            return None

    @staticmethod
    def _detect_topic(text: str) -> str:
        tx = (text or "").lower()
        if any(t in tx for t in ["<html", "css", "javascript", "thẻ"]): return "html_web"
        if any(t in tx for t in ["def ", "import ", "python", "biến", "hàm"]): return "python"
        if any(t in tx for t in ["sql", "primary key", "csdl", "bảng", "truy vấn"]): return "database"
        if any(t in tx for t in ["virus", "mật khẩu", "bản quyền", "luật an ninh"]): return "security"
        return "general"

    @staticmethod
    def _read_source_files(pdf_dir: str) -> List[Document]:
        if not os.path.exists(pdf_dir):
            return []
        
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        txt_files = glob.glob(os.path.join(pdf_dir, "*.txt"))
        all_files = pdf_files + txt_files
        docs: List[Document] = []

        for file_path in all_files:
            try:
                source_file = os.path.basename(file_path)
                # Xử lý tên file đẹp hơn cho citation
                source_name = source_file.rsplit('.', 1)[0].replace('_', ' ')
                
                if file_path.endswith('.pdf'):
                    reader = PdfReader(file_path)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            docs.append(Document(
                                page_content=text.replace('\x00', '').strip(),
                                metadata={"source": source_file, "page": page_num + 1, "title": source_name}
                            ))
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                        if text:
                            docs.append(Document(
                                page_content=text,
                                metadata={"source": source_file, "page": 1, "title": source_name}
                            ))
            except Exception:
                continue
        return docs

    @staticmethod
    def _chunk_documents(docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        # ⚠ Tối ưu splitter: ưu tiên ngắt câu, rồi đến dòng
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            add_start_index=True
        )
        chunks: List[Document] = []
        for d in docs:
            split_docs = splitter.split_documents([d])
            for i, sd in enumerate(split_docs):
                meta = dict(d.metadata)
                # ID duy nhất cho citation
                meta["chunk_id"] = f"{meta['source']}#p{meta['page']}#c{i}"
                meta["topic"] = RAGEngine._detect_topic(sd.page_content)
                chunks.append(Document(page_content=sd.page_content, metadata=meta))
        return chunks

    @staticmethod
    def build_hybrid_retriever(embeddings):
        if not embeddings: return None

        # Load/Create Vector DB
        vector_db = None
        if os.path.exists(AppConfig.VECTOR_DB_PATH):
            try:
                vector_db = FAISS.load_local(AppConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception:
                pass

        if not vector_db:
            raw_docs = RAGEngine._read_source_files(AppConfig.PDF_DIR)
            if not raw_docs:
                st.error(f"Không tìm thấy tài liệu trong {AppConfig.PDF_DIR}")
                return None
            
            chunk_docs = RAGEngine._chunk_documents(raw_docs)
            if not chunk_docs: return None

            vector_db = FAISS.from_documents(chunk_docs, embeddings)
            vector_db.save_local(AppConfig.VECTOR_DB_PATH)

        # Tạo Hybrid Retriever (BM25 + FAISS)
        try:
            docstore_docs = list(vector_db.docstore._dict.values())
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = AppConfig.RETRIEVAL_K

            faiss_retriever = vector_db.as_retriever(
                search_type="mmr", # Maximum Marginal Relevance để đa dạng hóa
                search_kwargs={"k": AppConfig.RETRIEVAL_K, "lambda_mult": 0.5}
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[AppConfig.BM25_WEIGHT, AppConfig.FAISS_WEIGHT]
            )
            return ensemble_retriever
        except Exception:
            return vector_db.as_retriever(search_kwargs={"k": AppConfig.RETRIEVAL_K})

    @staticmethod
    def process_multimodal(client, uploaded_file):
        # Hàm xử lý ảnh/âm thanh giữ nguyên logic
        vision_desc = ""
        audio_text = ""
        if uploaded_file.type.startswith('image'):
            base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            try:
                resp = client.chat.completions.create(
                    model=AppConfig.LLM_VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hãy trích xuất nội dung code hoặc văn bản trong ảnh này chi tiết."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }]
                )
                vision_desc = resp.choices[0].message.content or ""
            except Exception: pass
        elif uploaded_file.type.startswith('audio'):
            try:
                tmp_path = "temp_audio_input.mp3"
                with open(tmp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                with open(tmp_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=(tmp_path, f.read()),
                        model=AppConfig.LLM_AUDIO_MODEL
                    )
                audio_text = transcription.text or ""
                os.remove(tmp_path)
            except Exception: pass
        return vision_desc, audio_text

    @staticmethod
    def _format_context_with_citations(final_docs: List[Document]) -> Tuple[str, List[str]]:
        """
        Tạo context với ID [S1], [S2] rõ ràng để LLM trích dẫn.
        """
        context_parts = []
        sources_list = []
        
        for i, doc in enumerate(final_docs):
            idx = i + 1
            source_name = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            content = doc.page_content.replace('\n', ' ').strip()
            
            # Context block cho LLM
            context_parts.append(f"SOURCE_ID: [S{idx}]\nCONTENT: {content}\n")
            
            # List hiển thị cho UI
            sources_list.append(f"[{idx}] {source_name} - Trang {page}")
            
        return "\n----------------\n".join(context_parts), sources_list

    @staticmethod
    def generate_response(client, retriever, query, vision_context=None):
        # 1. Retrieval
        if not retriever:
            return ["Hệ thống chưa sẵn sàng."], []
        
        initial_docs = retriever.invoke(query)
        
        # 2. Soft Filter (Giữ lại logic lọc theo lớp/chủ đề nhưng an toàn hơn)
        filtered_docs = []
        grade_keywords = ["10", "11", "12"]
        detected_grade = next((g for g in grade_keywords if g in query), None)
        
        if detected_grade:
            for d in initial_docs:
                if detected_grade in d.metadata.get("source", "") or detected_grade in d.metadata.get("title", ""):
                    filtered_docs.append(d)
            # Nếu lọc xong mà mất hết doc thì lấy lại initial (fallback)
            if not filtered_docs: filtered_docs = initial_docs
        else:
            filtered_docs = initial_docs

        # 3. Rerank (FlashRank)
        final_docs = []
        try:
            ranker = RAGEngine.load_reranker()
            if ranker and filtered_docs:
                passages = [
                    {"id": str(i), "text": d.page_content, "meta": d.metadata} 
                    for i, d in enumerate(filtered_docs)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = ranker.rank(rerank_req)
                
                # Chỉ lấy top K sau rerank
                for res in results[:AppConfig.FINAL_K]:
                    final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
            else:
                final_docs = filtered_docs[:AppConfig.FINAL_K]
        except Exception:
            final_docs = filtered_docs[:AppConfig.FINAL_K]

        # 4. Kiểm tra rỗng (Strict Refusal)
        if not final_docs:
            return ["Không tìm thấy thông tin phù hợp trong tài liệu đã cung cấp. Vui lòng kiểm tra lại câu hỏi hoặc tài liệu nguồn."], []

        # 5. Format Context
        context_text, source_display = RAGEngine._format_context_with_citations(final_docs)
        
        # 6. System Prompt (Academic Standard)
        vision_instruction = f"Thông tin từ ảnh/code user gửi: {vision_context}\n" if vision_context else ""
        
        system_prompt = f"""Bạn là KTC Chatbot, trợ lý AI hỗ trợ môn Tin học tại trường THCS & THPT Phạm Kiệt.
NHIỆM VỤ: Trả lời câu hỏi dựa CHÍNH XÁC và DUY NHẤT vào [CONTEXT] bên dưới.

{vision_instruction}

QUY TẮC BẮT BUỘC (VI PHẠM LÀ SAI):
1. **Trích dẫn (Citation):** Mọi thông tin đưa ra phải gắn kèm thẻ nguồn [Sx] tương ứng.
   - Ví dụ: "Python là ngôn ngữ lập trình bậc cao [S1]."
   - Không được tự bịa ra [Sx] nếu không có trong Context.
2. **Trung thực:** Nếu [CONTEXT] không chứa câu trả lời, hãy nói: "Không tìm thấy thông tin trong tài liệu."
3. **Phong cách:** Sư phạm, ngắn gọn, dễ hiểu cho học sinh.
4. **Code:** Nếu hỏi code, hãy giải thích logic trước, sau đó đưa code minh họa ngắn gọn.

[CONTEXT STARTS]
{context_text}
[CONTEXT ENDS]
"""

        # 7. Streaming Response
        try:
            stream = client.chat.completions.create(
                model=AppConfig.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=AppConfig.LLM_TEMPERATURE,
                max_tokens=AppConfig.LLM_MAX_TOKENS
            )
            return stream, source_display
        except Exception as e:
            return [f"Lỗi kết nối AI: {str(e)}"], []

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    if not DEPENDENCIES_OK:
        st.error(f"⚠️ Lỗi thư viện: {IMPORT_ERROR}")
        st.info("Vui lòng chạy lệnh: pip install flashrank rank_bm25 langchain-huggingface")
        st.stop()

    UIManager.inject_custom_css()
    UIManager.render_sidebar()
    UIManager.render_header()

    # Khởi tạo Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Chào bạn! Mình là KTC Chatbot. Bạn cần hỗ trợ bài tập Tin học phần nào?"}]

    groq_client = RAGEngine.load_groq_client()

    # Init Retriever (Chỉ chạy 1 lần hoặc khi refresh DB)
    if "retriever_engine" not in st.session_state:
        with st.spinner("🚀 Đang khởi động hệ thống tri thức số (Hybrid)..."):
            embeddings = RAGEngine.load_embedding_model()
            st.session_state.retriever_engine = RAGEngine.build_hybrid_retriever(embeddings)
            if st.session_state.retriever_engine:
                st.toast("✅ Dữ liệu sẵn sàng!", icon="📚")

    # Render Chat History
    for msg in st.session_state.messages:
        bot_avatar = AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"
        avatar = "🧑‍🎓" if msg["role"] == "user" else bot_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Gợi ý câu hỏi (UI cũ)
    if len(st.session_state.messages) < 2:
        st.markdown("##### 💡 Gợi ý ôn tập:")
        cols = st.columns(3)
        prompt_btn = None
        if cols[0].button("🐍 Python: Số nguyên tố"):
            prompt_btn = "Viết chương trình Python nhập vào một số nguyên n và kiểm tra xem n có phải là số nguyên tố hay không. Giải thích code."
        if cols[1].button("🗃️ CSDL: Khóa chính"):
            prompt_btn = "Giải thích khái niệm Khóa chính (Primary Key) trong CSDL quan hệ và cho ví dụ minh họa."
        if cols[2].button("⚖️ Luật An ninh mạng"):
            prompt_btn = "Nêu các hành vi bị nghiêm cấm theo Luật An ninh mạng Việt Nam. Trích dẫn điều khoản nếu có."
        if prompt_btn:
            st.session_state.temp_input = prompt_btn
            st.rerun()

    # Xử lý input từ gợi ý
    if "temp_input" in st.session_state and st.session_state.temp_input:
        user_input = st.session_state.temp_input
        del st.session_state.temp_input
    else:
        user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AppConfig.LOGO_PROJECT if os.path.exists(AppConfig.LOGO_PROJECT) else "🤖"):
            response_placeholder = st.empty()

            if not groq_client:
                st.error("❌ Chưa cấu hình API Key.")
            else:
                # Multimodal Logic
                vision_context = None
                if "uploaded_file_obj" in st.session_state and st.session_state.uploaded_file_obj:
                    with st.status("🖼️ Đang phân tích file...", expanded=False):
                        vision_desc, audio_text = RAGEngine.process_multimodal(groq_client, st.session_state.uploaded_file_obj)
                        if audio_text:
                            user_input = f"{user_input} (Nội dung ghi âm: {audio_text})"
                            st.info(f"🎙️ Đã nghe: {audio_text}")
                        if vision_desc:
                            vision_context = vision_desc

                # Generate Response
                stream, sources = RAGEngine.generate_response(
                    groq_client,
                    st.session_state.retriever_engine,
                    user_input,
                    vision_context
                )

                full_response = ""
                # Xử lý Stream an toàn
                if isinstance(stream, list): # Trường hợp lỗi hoặc từ chối
                    full_response = stream[0]
                    response_placeholder.markdown(full_response)
                else:
                    try:
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                full_response += delta
                                response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                    except Exception as e:
                        response_placeholder.error(f"Lỗi hiển thị: {e}")

                # Hiển thị Citation
                if sources:
                    with st.expander("📚 Tài liệu tham khảo (Đã kiểm chứng)"):
                        for src in sources:
                            st.markdown(f"- 📖 *{src}*")

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Reset upload state
                if "uploaded_file_obj" in st.session_state and st.session_state.uploaded_file_obj:
                    st.session_state.uploaded_file_obj = None

if __name__ == "__main__":
    main()