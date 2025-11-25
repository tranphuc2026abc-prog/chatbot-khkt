import streamlit as st
from groq import Groq
import os
import glob
import time
from pypdf import PdfReader

# --- CÁC THƯ VIỆN RAG (LANGCHAIN) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Chatbot KTC - Trợ lý Tin học",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CÁC HẰNG SỐ ---
MODEL_NAME = 'llama-3.1-8b-instant'
PDF_DIR = "./PDF_KNOWLEDGE"
LOGO_PATH = "LOGO.jpg" 

# --- 2. CSS TÙY CHỈNH GIAO DIỆN (ĐÃ NÂNG CẤP) ---
st.markdown("""
<style>
    /* 1. Nền chính */
    .stApp {background-color: #f4f6f9;}
    
    /* 2. Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* 3. Box Thông tin tác giả */
    .author-box {
        background-color: #f0f8ff;
        border: 1px solid #cceeff;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.9rem;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        color: #333;
    }
    .author-title {
        font-weight: bold;
        color: #0072ff;
        margin-top: 8px;
        margin-bottom: 2px;
        font-size: 0.85rem;
    }
    .author-title:first-child { margin-top: 0; }
    .author-content { color: #333; margin-bottom: 4px; font-weight: 500; }
    .author-list { margin: 0; padding-left: 20px; color: #333; margin-bottom: 0; }

    /* 4. Tiêu đề Gradient */
    .gradient-text {
        background: linear-gradient(45deg, #004e92, #000428);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        padding-bottom: 0.5rem;
        text-align: center;
    }
    
    /* 5. Chat Bubble */
    .stChatMessage {background-color: transparent; border: none;}
    div[data-testid="stChatMessage"]:nth-child(even) { 
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 15px;
        padding: 10px 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 10px 15px;
        border: 1px solid #bbdefb;
    }

    /* 6. Disclaimer Footer */
    .footer-disclaimer {
        text-align: center;
        font-size: 0.75rem;
        color: #888;
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid #eee;
    }

    /* 7. Nút Gợi ý (Suggestion Buttons) */
    .stButton button {
        border-radius: 20px;
        border: 1px solid #0072ff;
        color: #0072ff;
        background-color: white;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #0072ff;
        color: white;
    }
    /* Riêng nút Làm mới ở Sidebar thì style khác */
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(90deg, #ff6b6b, #ff4757);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ KẾT NỐI ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ Lỗi: Chưa cấu hình GROQ_API_KEY trong .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)

@st.cache_resource(show_spinner=False)
def initialize_vector_db():
    if not os.path.exists(PDF_DIR) or not glob.glob(os.path.join(PDF_DIR, "*.pdf")):
        return None
    
    with st.spinner('🔄 Đang khởi tạo "Bộ não" kiến thức (Vector hóa dữ liệu)...'):
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for pdf_path in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
            try:
                reader = PdfReader(pdf_path)
                file_name = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            documents.append(Document(page_content=chunk, metadata={"source": file_name, "page": i + 1}))
            except Exception: pass

        if not documents: return None
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(documents, embeddings)

# --- KHỞI TẠO STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Mình là Chatbot KTC 🤖. Mình có thể giúp gì cho bạn về môn Tin học hôm nay?"}]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = initialize_vector_db()

# --- 4. SIDEBAR ---
with st.sidebar:
    # Logo lớn hơn theo yêu cầu
    col_c = st.container()
    if os.path.exists(LOGO_PATH):
        col_c.image(LOGO_PATH, use_container_width=True) # Dùng full chiều rộng
    
    st.markdown("<h2 style='text-align: center; color: #0072ff; font-size: 1.5rem;'>TRỢ LÝ KTC</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Trạng thái
    status_html = "<span style='color:green; font-weight:bold'>Đã kết nối</span>" if st.session_state.vector_db else "<span style='color:red; font-weight:bold'>Chưa nạp</span>"
    st.markdown(f"💾 Dữ liệu SGK: {status_html}", unsafe_allow_html=True)
        
    # Thông tin tác giả
    st.markdown("""
        <div class="author-box">
            <div class="author-title">🏫 Sản phẩm KHKT:</div>
            <div class="author-content"> Năm học: 2025-2026 </div>
            <div class="author-title">👨‍🏫 GV Hướng Dẫn:</div>
            <div class="author-content">Thầy Nguyễn Thế Khanh</div>
            <div class="author-title">🧑‍🎓 Nhóm tác giả:</div>
            <ul class="author-list">
                <li>Bùi Tá Tùng</li>
                <li>Cao Sỹ Bảo Chung</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    # Đổi icon thành Refresh
    if st.button("🔄 Bắt đầu cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Mình là Chatbot KTC 🤖. Mình có thể giúp gì cho bạn về môn Tin học hôm nay?"}]
        st.rerun()

# --- 5. GIAO DIỆN CHÍNH ---
col1, col2, col3 = st.columns([1, 10, 1]) # Tăng độ rộng cột giữa

with col2:
    st.markdown('<div class="gradient-text">CHATBOT HỖ TRỢ HỌC TẬP KTC</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>🚀 Ứng dụng AI hỗ trợ tra cứu kiến thức Tin học chương trình GDPT 2018</p>", unsafe_allow_html=True)
    
    # --- HIỂN THỊ LỊCH SỬ CHAT ---
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    # --- XỬ LÝ EMPTY STATE (GỢI Ý CÂU HỎI) ---
    # Chỉ hiện khi chỉ có đúng 1 tin nhắn (lời chào của Bot)
    if len(st.session_state.messages) == 1:
        st.markdown("<p style='text-align:center; color:#888; margin-top:20px;'>💡 <b>Gợi ý câu hỏi bắt đầu:</b></p>", unsafe_allow_html=True)
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        # Danh sách câu hỏi gợi ý
        questions = [
            "Cấu trúc rẽ nhánh là gì?",
            "Cách tạo mục lục trong Word?",
            "Phần mềm nguồn mở là gì?"
        ]
        
        # Logic nút bấm: Khi bấm -> thêm vào history -> rerun
        if btn_col1.button(questions[0], use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": questions[0]})
            st.rerun()
        if btn_col2.button(questions[1], use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": questions[1]})
            st.rerun()
        if btn_col3.button(questions[2], use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": questions[2]})
            st.rerun()

    # --- INPUT CHAT ---
    # Luôn hiển thị input ở dưới cùng
    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # --- LOGIC AI TRẢ LỜI (TRIGGER) ---
    # Kiểm tra: Nếu tin nhắn cuối cùng là của User -> Gọi AI
    if st.session_state.messages[-1]["role"] == "user":
        user_msg = st.session_state.messages[-1]["content"]
        
        # Hiển thị tin nhắn user (để chắc chắn nó hiện ra trước khi AI chạy)
        # (Lưu ý: Streamlit render lại từ đầu nên thực ra nó đã hiện ở vòng for trên rồi)

        # 1. Tìm kiếm RAG
        context_text = ""
        sources_list = []
        if st.session_state.vector_db:
            results = st.session_state.vector_db.similarity_search(user_msg, k=3)
            for doc in results:
                context_text += f"\n---\nNội dung: {doc.page_content}\nNguồn: {doc.metadata['source']} (Trang {doc.metadata['page']})"
                sources_list.append(f"{doc.metadata['source']} - Tr. {doc.metadata['page']}")

        # 2. Tạo Prompt
        SYSTEM_PROMPT = """Bạn là "Chatbot KTC", trợ lý ảo chuyên gia Tin học. Trả lời dựa trên SGK. Luôn trích dẫn nguồn."""
        final_prompt = f"{SYSTEM_PROMPT}\n--- BỐI CẢNH SGK ---\n{context_text}\n--- CÂU HỎI ---\n{user_msg}"

        # 3. Gọi API & Stream
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": user_msg}],
                    model=MODEL_NAME, stream=True, temperature=0.3
                )
                
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                
                # Hiển thị nguồn
                if sources_list:
                    unique_sources = list(set(sources_list))
                    citation_html = "<div style='margin-top:10px; font-size: 0.8em; color: #666; border-top: 1px dashed #ccc; padding-top: 5px;'>📚 <b>Nguồn tham khảo:</b><br>" + "<br>".join([f"- <i>{s}</i>" for s in unique_sources]) + "</div>"
                    placeholder.markdown(full_response + "\n" + citation_html, unsafe_allow_html=True)
                else:
                    placeholder.markdown(full_response)
                
                # Lưu vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": full_response + (citation_html if sources_list else "")})
            
            except Exception as e:
                st.error(f"Lỗi kết nối: {e}")

    # --- DISCLAIMER FOOTER ---
    st.markdown('<div class="footer-disclaimer">⚠️ Lưu ý: AI có thể mắc lỗi. Vui lòng kiểm tra lại thông tin quan trọng với SGK.</div>', unsafe_allow_html=True)