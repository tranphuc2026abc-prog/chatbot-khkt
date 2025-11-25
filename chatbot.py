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

# --- 1. CẤU HÌNH TRANG (BẮT BUỘC Ở DÒNG ĐẦU TIÊN) ---
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

# --- 2. CSS TÙY CHỈNH GIAO DIỆN ---
st.markdown("""
<style>
    /* 1. Nền chính */
    .stApp {background-color: #f4f6f9;}
    
    /* 2. Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem; 
        padding-bottom: 1rem;
    }
    [data-testid="stSidebar"] .stMarkdown {margin-bottom: -10px;}
    [data-testid="stSidebar"] hr {margin: 15px 0;}

    /* 3. Box Thông tin tác giả (FIX LỖI HIỂN THỊ CODE) */
    .author-box {
        background-color: #f0f8ff;
        border: 1px solid #cceeff;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.9rem;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        color: #333; /* Màu chữ đen cho dễ đọc */
    }
    .author-title {
        font-weight: bold;
        color: #0072ff;
        margin-top: 8px;
        margin-bottom: 2px;
        font-size: 0.85rem;
    }
    /* Dòng đầu tiên không cần margin-top */
    .author-title:first-child { margin-top: 0; }
    
    .author-content {
        color: #333;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .author-list {
        margin: 0;
        padding-left: 20px;
        color: #333;
        margin-bottom: 0;
    }

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
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #bbdefb;
    }

    /* 6. Button */
    .stButton>button {
        border-radius: 20px;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 10px rgba(0,114,255,0.3);
        color: white;
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

# --- KHỞI TẠO STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Chào bạn! Mình là Chatbot KTC 🤖. Mình có thể giúp gì cho bạn về môn Tin học hôm nay?"
    })

if "vector_db" not in st.session_state:
    st.session_state.vector_db = initialize_vector_db()

# --- 4. GIAO DIỆN SIDEBAR (ĐÃ FIX LỖI HTML) ---
with st.sidebar:
    # 1. LOGO
    col_l, col_c, col_r = st.columns([1, 5, 1]) 
    with col_c:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=160)
        else:
            st.warning("Thiếu file LOGO.jpg")
    
    # 2. Tiêu đề
    st.markdown("""
        <div style='text-align: center; margin-top: -10px;'>
            <h2 style='color: #0072ff; margin-bottom: 5px; font-size: 1.5rem;'>TRỢ LÝ KTC</h2>
            <p style='font-size: 0.8rem; color: #666; margin-top: 0;'>
                Knowledge & Technology Chatbot
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. Trạng thái
    if st.session_state.vector_db:
        st.markdown("💾 Dữ liệu SGK: <span style='color:green; font-weight:bold'>Đã kết nối</span>", unsafe_allow_html=True)
    else:
        st.markdown("💾 Dữ liệu SGK: <span style='color:red; font-weight:bold'>Chưa nạp</span>", unsafe_allow_html=True)
        
    # 4. Thông tin Tác giả (CODE HTML ĐÃ CHỈNH SỬA)
    # Lưu ý: Viết liền mạch, không xuống dòng bừa bãi trong chuỗi string này
    st.markdown("""
        <div class="author-box">
            <div class="author-title">🏫 Sản phẩm cuộc thi KHKT cấp trường:</div>
            <div class="author-content">Năm học 2025-2026</div>
            <div class="author-title">👨‍🏫 GV Hướng Dẫn:</div>
            <div class="author-content">Thầy Nguyễn Thế Khanh</div>
            <div class="author-title">🧑‍🎓 Nhóm tác giả:</div>
            <ul class="author-list">
                <li>Bùi Tá Tùng</li>
                <li>Cao Sỹ Bảo Chung</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Nút xóa lịch sử
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Làm mới hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. GIAO DIỆN CHÍNH ---
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    st.markdown('<div class="gradient-text">CHATBOT HỖ TRỢ HỌC TẬP KTC</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>🚀 Ứng dụng AI hỗ trợ tra cứu kiến thức Tin học chương trình GDPT 2018</p>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            avatar = "🧑‍🎓"
        else:
            avatar = "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

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
