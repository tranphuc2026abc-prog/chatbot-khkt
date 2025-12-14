import streamlit as st
import os
import glob
import nest_asyncio
import pickle
from groq import Groq

# --- CÁC THƯ VIỆN RAG NÂNG CAO ---
from llama_parse import LlamaParse  # Công cụ parse PDF xịn nhất hiện nay
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Apply asyncio fix cho LlamaParse chạy trên Streamlit
nest_asyncio.apply()

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(
    page_title="Chatbot KTC - Trợ lý Tin học",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cấu hình đường dẫn
MODEL_NAME = 'llama-3.1-8b-instant'
PDF_DIR = "./PDF_KNOWLEDGE"
CACHE_DIR = "./CACHE_DATA" # Nơi lưu dữ liệu đã xử lý để không phải parse lại
LOGO_PATH = "LOGO.jpg"

# Đảm bảo thư mục tồn tại
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- 2. CSS GIAO DIỆN (GIỮ NGUYÊN STYLE CỦA BẠN) ---
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    [data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #e0e0e0;}
    .author-box {background-color: #f0f8ff; border: 1px solid #bae6fd; border-radius: 10px; padding: 15px; margin-top: 15px; color: #0f172a;}
    .author-header {font-weight: bold; color: #0284c7; margin-bottom: 5px; font-size: 0.85rem; text-transform: uppercase; margin-top: 10px;}
    .gradient-text {background: linear-gradient(90deg, #0f4c81, #1cb5e0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem; text-align: center;}
    div[data-testid="stChatMessage"]:nth-child(even) {background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 0px 15px 15px 15px;}
    div[data-testid="stChatMessage"]:nth-child(odd) {background-color: #e0f2fe; border-radius: 15px 0px 15px 15px; border: none;}
    .footer-note {text-align: center; font-size: 0.75rem; color: #94a3b8; margin-top: 30px; border-top: 1px dashed #cbd5e1; padding-top: 10px;}
</style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ KẾT NỐI API ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    llama_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ Lỗi: Thiếu API Key trong secrets.toml (Cần cả GROQ_API_KEY và LLAMA_CLOUD_API_KEY)")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- 4. HÀM XỬ LÝ DỮ LIỆU THÔNG MINH (THEO ĐỊNH HƯỚNG MỚI) ---

@st.cache_resource(show_spinner=False)
def load_and_process_data():
    """
    Quy trình:
    1. Kiểm tra xem đã có file Markdown cache chưa.
    2. Nếu chưa -> Dùng LlamaParse chuyển PDF -> Markdown (Giữ cấu trúc bảng/hình).
    3. Lưu cache để lần sau chạy nhanh hơn.
    4. Chia nhỏ văn bản theo ngữ nghĩa (Header splitter).
    5. Tạo Vector DB.
    """
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        return None

    all_documents = []
    
    # --- GIAI ĐOẠN 1: PARSING (PDF -> MARKDOWN) ---
    with st.spinner('🔄 Đang số hóa tri thức SGK (LlamaParse)...'):
        parser = LlamaParse(
            api_key=llama_api_key,
            result_type="markdown",
            verbose=True,
            language="vi",
            gpt4o_mode=True # Chế độ thông minh nhất để hiểu layout SGK
        )

        for pdf_path in pdf_files:
            file_name = os.path.basename(pdf_path)
            cache_path = os.path.join(CACHE_DIR, f"{file_name}.md")
            
            markdown_text = ""

            # Kiểm tra cache
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
            else:
                # Nếu chưa có cache thì gọi API parse
                try:
                    documents = parser.load_data(pdf_path)
                    markdown_text = "\n".join([doc.text for doc in documents])
                    # Lưu cache lại
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(markdown_text)
                except Exception as e:
                    st.warning(f"Không thể đọc file {file_name}: {e}")
                    continue
            
            # Tạo document thô ban đầu
            if markdown_text:
                # Thêm tên nguồn vào đầu văn bản để AI biết
                markdown_text = f"Nguồn tài liệu: {file_name}\n\n" + markdown_text
                all_documents.append(Document(page_content=markdown_text, metadata={"source": file_name}))

    if not all_documents:
        return None

    # --- GIAI ĐOẠN 2: CHUNKING (CHIA NHỎ THEO CẤU TRÚC) ---
    with st.spinner('🧠 Đang tổ chức lại kiến thức (Markdown Splitting)...'):
        # 1. Cắt theo Header (Chương/Bài) trước để giữ ngữ cảnh
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        md_header_splits = []
        for doc in all_documents:
            splits = markdown_splitter.split_text(doc.page_content)
            for split in splits:
                split.metadata["source"] = doc.metadata["source"] # Copy metadata nguồn
                md_header_splits.append(split)

        # 2. Cắt mịn lại nếu đoạn văn vẫn quá dài (đảm bảo vừa context window)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        final_splits = text_splitter.split_documents(md_header_splits)

    # --- GIAI ĐOẠN 3: EMBEDDING & VECTOR DB ---
    with st.spinner('💾 Đang ghi nhớ vào não bộ...'):
        # Dùng model Multilingual để hiểu tiếng Việt tốt hơn model cũ
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vector_db = FAISS.from_documents(final_splits, embeddings)
        
    return vector_db

# --- KHỞI TẠO STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Mình là Chatbot KTC 🤖. Mình đã được nâng cấp để đọc SGK chính xác hơn. Bạn cần hỏi gì nào?"}]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_and_process_data()

# --- 5. SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    
    st.markdown("""
        <div style='text-align: center; margin-top: 10px;'>
            <h3 style='color: #0f4c81; margin: 0;'>TRỢ LÝ KTC</h3>
            <p style='font-size: 0.8rem; color: #64748b;'>Powered by LlamaParse & Groq</p>
        </div>
        <hr style="margin: 15px 0;">
    """, unsafe_allow_html=True)
    
    if st.session_state.vector_db:
        st.markdown("💾 Trạng thái: <span style='color:green; font-weight:bold'>● Sẵn sàng</span>", unsafe_allow_html=True)
    else:
        st.markdown("💾 Trạng thái: <span style='color:red; font-weight:bold'>● Chưa có dữ liệu</span>", unsafe_allow_html=True)
        st.info("Hãy bỏ file PDF vào thư mục PDF_KNOWLEDGE nhé.")

    html_info = """
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
    """
    st.markdown(html_info, unsafe_allow_html=True)
    
    if st.button("🗑️ Xóa bộ nhớ tạm", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 6. GIAO DIỆN CHÍNH & XỬ LÝ CHAT ---
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    st.markdown('<h1 class="gradient-text">CHATBOT HỖ TRỢ HỌC TẬP KTC</h1>', unsafe_allow_html=True)
    
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Input người dùng
    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        # --- LOGIC RAG ---
        context_text = ""
        sources_list = []
        
        if st.session_state.vector_db:
            # Tìm kiếm 4 đoạn văn bản liên quan nhất
            results = st.session_state.vector_db.similarity_search(prompt, k=4)
            for doc in results:
                # Lấy metadata header nếu có
                header_info = ""
                if "Header 1" in doc.metadata: header_info += f" > {doc.metadata['Header 1']}"
                if "Header 2" in doc.metadata: header_info += f" > {doc.metadata['Header 2']}"
                
                context_text += f"\n---\n[Nguồn: {doc.metadata['source']}{header_info}]\nNội dung: {doc.page_content}\n"
                sources_list.append(f"{doc.metadata['source']}{header_info}")

        # System Prompt được tinh chỉnh để trích xuất chính xác
        SYSTEM_PROMPT = """
        Bạn là trợ lý AI giáo dục (KTC Chatbot). Nhiệm vụ của bạn là trả lời câu hỏi dựa trên DỮ LIỆU ĐƯỢC CUNG CẤP (Context).
        
        QUY TẮC TRẢ LỜI:
        1. CHÍNH XÁC: Chỉ dùng thông tin trong Context. Nếu không có thông tin, hãy nói "Xin lỗi, sách giáo khoa không đề cập vấn đề này."
        2. TRÌNH BÀY: Dùng Markdown. Nếu có công thức toán/tin, hãy viết rõ ràng. Nếu có bảng biểu trong context, hãy vẽ lại bảng.
        3. NGÔN NGỮ: Tiếng Việt sư phạm, dễ hiểu, phù hợp học sinh.
        4. TRÍCH DẪN: Luôn nhắc đến thông tin này nằm ở bài nào/chương nào nếu Context có cung cấp.
        """
        
        final_prompt = f"{SYSTEM_PROMPT}\n\n--- DỮ LIỆU THAM KHẢO TỪ SGK ---\n{context_text}\n\n--- CÂU HỎI HỌC SINH ---\n{prompt}"

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
                    temperature=0.2 # Giảm nhiệt độ để tăng độ chính xác
                )

                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        placeholder.markdown(full_response + "▌")
                
                # Hiển thị nguồn tham khảo
                if sources_list:
                    unique_sources = list(set(sources_list))
                    citation_html = "<div style='margin-top:10px; font-size: 0.85em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>📚 <b>Nguồn SGK tham chiếu:</b><br>" + "<br>".join([f"- <i>{s}</i>" for s in unique_sources]) + "</div>"
                    placeholder.markdown(full_response + "\n\n" + citation_html, unsafe_allow_html=True)
                else:
                    placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Lỗi kết nối AI: {e}")

    st.markdown('<div class="footer-note">⚠️ KTC Chatbot hỗ trợ học tập - Hãy đối chiếu với SGK gốc.</div>', unsafe_allow_html=True)