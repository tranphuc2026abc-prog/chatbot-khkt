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

# --- 2. CSS TÙY CHỈNH GIAO DIỆN (ĐẸP HƠN, KHÔNG ẢNH HƯỞNG LOGIC) ---
st.markdown("""
<style>
    /* 1. Nền chính */
    .stApp {background-color: #f8f9fa;}
    
    /* 2. Sidebar - Làm sạch và chuyên nghiệp */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* 3. Box Thông tin tác giả (Style mới) */
    .author-box {
        background-color: #f0f8ff; /* Màu xanh nhạt */
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 15px;
        font-size: 0.9rem;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #0f172a;
    }
    .author-header {
        font-weight: bold;
        color: #0284c7; /* Xanh đậm */
        margin-bottom: 5px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .author-content {
        margin-bottom: 8px;
        color: #334155;
    }
    .author-list {
        margin: 0;
        padding-left: 20px;
        color: #334155;
        font-weight: 500;
    }

    /* 4. Tiêu đề Gradient (Điểm nhấn chính) */
    .gradient-text {
        background: linear-gradient(90deg, #0f4c81, #1cb5e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        padding-bottom: 1rem;
        text-align: center;
        margin-bottom: 0;
    }
    
    /* 5. Chat Bubble (Bong bóng chat) */
    .stChatMessage {
        background-color: transparent; 
        border: none;
        padding: 10px;
    }
    /* Tin nhắn của Bot */
    div[data-testid="stChatMessage"]:nth-child(even) { 
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0px 15px 15px 15px; /* Bo góc kiểu chat */
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    /* Tin nhắn của User */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e0f2fe; /* Xanh rất nhạt */
        border-radius: 15px 0px 15px 15px;
        border: none;
    }

    /* 6. Button (Nút bấm) */
    .stButton>button {
        border-radius: 8px;
        background-color: #0284c7;
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        color: white;
        transform: translateY(-1px);
    }

    /* 7. Footer Disclaimer */
    .footer-note {
        text-align: center;
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 30px;
        border-top: 1px dashed #cbd5e1;
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ KẾT NỐI (GIỮ NGUYÊN) ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ Lỗi: Chưa cấu hình GROQ_API_KEY trong .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)

@st.cache_resource(show_spinner=False)
def initialize_vector_db():
    vector_db = None
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        return None
    
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        return None

    with st.spinner('🔄 Đang khởi tạo "Bộ não" kiến thức (Vector hóa dữ liệu)...'):
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
        )

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                file_name = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            documents.append(Document(
                                page_content=chunk,
                                metadata={"source": file_name, "page": i + 1}
                            ))
            except Exception as e:
                print(f"Lỗi đọc file {pdf_path}: {e}")

        if not documents:
            return None

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(documents, embeddings)
        return vector_db

# --- KHỞI TẠO STATE (GIỮ NGUYÊN) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Chào bạn! Mình là Chatbot KTC 🤖. Mình có thể giúp gì cho bạn về môn Tin học hôm nay?"
    })

if "vector_db" not in st.session_state:
    st.session_state.vector_db = initialize_vector_db()

# --- 4. GIAO DIỆN SIDEBAR (CẬP NHẬT GIAO DIỆN) ---
with st.sidebar:
    # 1. LOGO (Tối ưu hiển thị)
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True) # Dùng lệnh mới để logo full khung
    else:
        st.warning("Thiếu file LOGO.jpg")
    
    # 2. Tiêu đề Sidebar
    st.markdown("""
        <div style='text-align: center; margin-top: 10px;'>
            <h3 style='color: #0f4c81; margin: 0;'>TRỢ LÝ KTC</h3>
            <p style='font-size: 0.8rem; color: #64748b;'>Knowledge & Technology Chatbot</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. Trạng thái
    if st.session_state.vector_db:
        st.markdown("💾 Dữ liệu SGK: <span style='color:green; font-weight:bold'>● Đã kết nối</span>", unsafe_allow_html=True)
    else:
        st.markdown("💾 Dữ liệu SGK: <span style='color:red; font-weight:bold'>● Chưa nạp</span>", unsafe_allow_html=True)
        
    # 4. Thông tin Tác giả (HTML MỚI - ĐẸP HƠN)
    st.markdown("""
        <div class="author-box">
            <div class="author-header">🏫 Sản phẩm KHKT</div>
            <div class="author-content">Năm học 2025 - 2026</div>
            
            <div class="author-header">👨‍🏫 GV Hướng Dẫn</div>
            <div class="author-content">Thầy Nguyễn Thế Khanh</div>
            
            <div class="author-header">🧑‍🎓 Nhóm tác giả</div>
            <ul class="author-list">
                <li>Bùi Tá Tùng</li>
                <li>Cao Sỹ Bảo Chung</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Nút xóa lịch sử (Giữ nguyên logic)
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Làm mới hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. GIAO DIỆN CHÍNH (LOGIC GIỮ NGUYÊN - CHỈ ĐỔI GIAO DIỆN) ---
col1, col2, col3 = st.columns([1, 8, 1]) # Điều chỉnh tỷ lệ cột cho cân đối hơn

with col2:
    # Tiêu đề mới
    st.markdown('<h1 class="gradient-text">CHATBOT HỖ TRỢ HỌC TẬP KTC</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-style: italic; margin-bottom: 30px;'>🚀 Ứng dụng AI hỗ trợ tra cứu kiến thức Tin học chương trình GDPT 2018</p>", unsafe_allow_html=True)
    
    # Vòng lặp hiển thị tin nhắn (Giữ nguyên)
    for message in st.session_state.messages:
        if message["role"] == "user":
            avatar = "🧑‍🎓"
        else:
            avatar = "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    # INPUT và XỬ LÝ (LOGIC CỐT LÕI - GIỮ NGUYÊN 100%)
    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        context_text = ""
        sources_list = []
        if st.session_state.vector_db:
            results = st.session_state.vector_db.similarity_search(prompt, k=3)
            if results:
                for doc in results:
                    context_text += f"\n---\nNội dung: {doc.page_content}\nNguồn: {doc.metadata['source']} (Trang {doc.metadata['page']})"
                    sources_list.append(f"{doc.metadata['source']} - Tr. {doc.metadata['page']}")

        SYSTEM_PROMPT = """
        Bạn là "Chatbot KTC", trợ lý ảo chuyên gia về Tin học.
        Nhiệm vụ: Giải đáp thắc mắc dựa trên bối cảnh SGK được cung cấp.
        Phong cách: Thân thiện, sư phạm, khuyến khích học sinh tư duy.
        Định dạng: Sử dụng Markdown để trình bày đẹp (in đậm từ khóa, gạch đầu dòng).
        Quan trọng: Luôn trích dẫn nguồn nếu thông tin lấy từ sách.
        """
        
        final_prompt = f"""
        {SYSTEM_PROMPT}
        --- BỐI CẢNH SGK ---
        {context_text if context_text else "Không tìm thấy trong tài liệu, hãy trả lời dựa trên kiến thức chung của bạn."}
        --- CÂU HỎI ---
        {prompt}
        """

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": final_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=MODEL_NAME,
                    stream=True,
                    temperature=0.3
                )

                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        placeholder.markdown(full_response + "▌")
                
                if sources_list:
                    unique_sources = list(set(sources_list))
                    citation_html = "<div style='margin-top:10px; font-size: 0.85em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>📚 <b>Nguồn tham khảo:</b><br>"
                    for src in unique_sources:
                        citation_html += f"- <i>{src}</i><br>"
                    citation_html += "</div>"
                    full_response += "\n"
                    placeholder.markdown(full_response + "\n\n" + citation_html, unsafe_allow_html=True)
                else:
                    placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Đã xảy ra lỗi kết nối: {e}")

    # --- DISCLAIMER (PHẦN THÊM VÀO CUỐI CÙNG - KHÔNG ẢNH HƯỞNG LOGIC) ---
    st.markdown('<div class="footer-note">⚠️ Lưu ý: AI có thể mắc lỗi (hallucination). Vui lòng kiểm tra lại thông tin quan trọng với SGK.</div>', unsafe_allow_html=True)