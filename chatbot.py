import streamlit as st
from groq import Groq
import os
import glob
import time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. CẤU HÌNH TRANG (PHẢI Ở DÒNG ĐẦU TIÊN) ---
st.set_page_config(
    page_title="Chatbot KTC - Trợ lý Tin học",
    page_icon="🤖",
    layout="wide", # Chuyển sang wide để thoáng hơn
    initial_sidebar_state="expanded"
)

# --- CÁC HẰNG SỐ ---
MODEL_NAME = 'llama-3.1-8b-instant'
PDF_DIR = "./PDF_KNOWLEDGE"
LOGO_PATH = "LOGO.jpg" # Đảm bảo file ảnh nằm cùng thư mục code

# --- 2. CSS TÙY CHỈNH (NÂNG CẤP GIAO DIỆN) ---
# Phong cách: Clean, Modern, Tech Blue
st.markdown("""
<style>
    /* 1. Tùy chỉnh Font và Màu nền chính */
    .stApp {
        background-color: #f4f6f9; /* Xám xanh rất nhạt, dịu mắt */
    }
    
    /* 2. Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    
    /* 3. Tùy chỉnh Tiêu đề Gradient */
    .gradient-text {
        background: linear-gradient(45deg, #004e92, #000428);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        padding-bottom: 1rem;
    }
    
    /* 4. Tùy chỉnh Bong bóng chat */
    .stChatMessage {
        background-color: transparent;
        border: none;
    }
    /* Tin nhắn của Bot */
    div[data-testid="stChatMessage"]:nth-child(even) { 
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Tin nhắn của User */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e3f2fd; /* Xanh dương nhạt */
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #bbdefb;
    }

    /* 5. Nút bấm và Input */
    .stButton>button {
        border-radius: 25px;
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
    
    /* 6. Info Box Custom */
    .info-box {
        padding: 15px;
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ API VÀ DATABASE (LOGIC CŨ) ---
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
    # Tin nhắn chào mừng mặc định
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Chào bạn! Mình là Chatbot KTC 🤖. Mình có thể giúp gì cho bạn về môn Tin học hôm nay?"
    })

if "vector_db" not in st.session_state:
    st.session_state.vector_db = initialize_vector_db()

# --- 4. GIAO DIỆN SIDEBAR (CHUYÊN NGHIỆP HÓA) ---
with st.sidebar:
    # Hiển thị Logo
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=True)
    else:
        st.warning("⚠️ Chưa tìm thấy file LOGO.jpg")
    
    st.markdown("<h2 style='text-align: center; color: #0072ff;'>TRỢ LÝ KTC</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Trạng thái hệ thống
    st.markdown("### 📡 Trạng thái hệ thống")
    if st.session_state.vector_db:
        st.success("✅ Kết nối tri thức SGK: **Sẵn sàng**")
    else:
        st.warning("⚠️ Chưa nạp dữ liệu SGK")
        
    st.markdown("---")
    
    # Thông tin dự án (Quan trọng cho KHKT)
    with st.expander("ℹ️ Thông tin dự án", expanded=True):
        st.markdown("**TRƯỜNG:** THCS VÀ THPT PHẠM KIỆT")
        st.markdown("**GVHD:** Thầy Nguyễn Thế Khanh")
        st.markdown("**Nhóm tác giả:**")
        st.markdown("- Bùi Tá Tùng")
        st.markdown("- Cao Sỹ Bảo Chung")
    
    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. GIAO DIỆN CHÍNH (MAIN COLUMN) ---
# Tạo layout 3 cột để căn giữa nội dung chính, giúp mắt tập trung hơn
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Header chính
    st.markdown('<div class="gradient-text">CHATBOT HỖ TRỢ HỌC TẬP KTC</div>', unsafe_allow_html=True)
    st.caption("🚀 Ứng dụng AI hỗ trợ tra cứu kiến thức Tin học chương trình GDPT 2018")
    
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        # Chọn Avatar
        if message["role"] == "user":
            avatar = "🧑‍🎓" # Avatar học sinh
        else:
            avatar = "🤖" # Avatar Robot (hoặc có thể dùng icon KTC nhỏ nếu muốn)
            
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Input area
    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    # --- LOGIC XỬ LÝ (GIỮ NGUYÊN) ---
    if prompt:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # Retrieval
        context_text = ""
        sources_list = []
        if st.session_state.vector_db:
            results = st.session_state.vector_db.similarity_search(prompt, k=3)
            if results:
                for doc in results:
                    context_text += f"\n---\nNội dung: {doc.page_content}\nNguồn: {doc.metadata['source']} (Trang {doc.metadata['page']})"
                    sources_list.append(f"{doc.metadata['source']} - Tr. {doc.metadata['page']}")

        # System Prompt
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

        # Generate Response
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
                
                # Hiển thị trích dẫn nguồn (Feature hay cho KHKT)
                if sources_list:
                    # Loại bỏ trùng lặp nguồn
                    unique_sources = list(set(sources_list))
                    citation_html = "<div style='margin-top:10px; font-size: 0.85em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>📚 <b>Nguồn tham khảo:</b><br>"
                    for src in unique_sources:
                        citation_html += f"- <i>{src}</i><br>"
                    citation_html += "</div>"
                    full_response += "\n" # Xuống dòng để tách text
                    placeholder.markdown(full_response + "\n\n" + citation_html, unsafe_allow_html=True) # Render HTML cho đẹp
                else:
                    placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:

                st.error(f"Đã xảy ra lỗi kết nối: {e}")
