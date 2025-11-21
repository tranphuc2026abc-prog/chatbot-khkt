import streamlit as st
from groq import Groq
import os
import glob
import time
from pypdf import PdfReader
# --- CÁC THƯ VIỆN RAG CHUẨN (FAISS + EMBEDDINGS) ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# --- CẤU HÌNH ---
st.set_page_config(page_title="Chatbot Tin học KTC", page_icon="🤖", layout="centered")
MODEL_NAME = 'llama-3.1-8b-instant'
PDF_DIR = "./PDF_KNOWLEDGE" # Thư mục chứa SGK PDF

# --- CSS GIAO DIỆN (Giữ nguyên phong cách của thầy) ---
st.markdown("""
<style>
    [data-testid="stSidebar"] {background-color: #f8f9fa; border-right: 1px solid #e6e6e6;}
    .main .block-container {max-width: 850px; padding-top: 2rem; padding-bottom: 5rem;}
    .stButton>button {border-radius: 20px; height: 3em; background-color: #ffffff; border: 1px solid #d0d0d0;}
    .stButton>button:hover {border-color: #4CAF50; color: #4CAF50;}
    .chat-message {padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;}
    .chat-message.user {background-color: #e6f7ff;}
    .chat-message.bot {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

# --- LẤY API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ Lỗi: Chưa cấu hình GROQ_API_KEY trong .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)

# --- HỆ THỐNG RAG: FAISS + EMBEDDINGS ---
@st.cache_resource(show_spinner=False)
def initialize_vector_db():
    """
    Hàm này đọc PDF, tạo Embeddings và xây dựng Vector Store (FAISS).
    Chạy 1 lần duy nhất khi khởi động app để tối ưu tốc độ.
    """
    vector_db = None
    
    # 1. Kiểm tra thư mục PDF
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        return None
    
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        return None

    with st.spinner('🔄 Đang khởi tạo "Bộ não" kiến thức (Vector hóa dữ liệu)...'):
        # 2. Đọc và Chia nhỏ văn bản (Chunking)
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Kích thước mỗi đoạn (khoảng 2-3 đoạn văn)
            chunk_overlap=200,  # Phần chồng lấn để giữ ngữ cảnh
            separators=["\n\n", "\n", ".", " ", ""]
        )

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                file_name = os.path.basename(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        # Lưu thêm metadata (tên sách, số trang) để trích dẫn nguồn
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

        # 3. Tạo Embeddings (Sử dụng Model thu nhỏ của HuggingFace - Chạy Offline OK)
        # Model này biến văn bản thành vector 384 chiều
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 4. Tạo FAISS Index (Vector Database)
        vector_db = FAISS.from_documents(documents, embeddings)
        
        print(f"✅ Đã khởi tạo thành công Vector DB với {len(documents)} chunks kiến thức.")
        
    return vector_db

# --- KHỞI TẠO SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = initialize_vector_db()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 Chatbot KTC")
    st.caption("Trợ lý học tập môn Tin học")
    st.markdown("---")
    
    # Hiển thị trạng thái hệ thống RAG
    if st.session_state.vector_db:
        st.success("✅ Kết nối tri thức SGK: Đã sẵn sàng", icon="📚")
    else:
        st.warning("⚠️ Chưa có dữ liệu SGK. Vui lòng chép file PDF vào thư mục PDF_KNOWLEDGE.", icon="📂")
        
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.info("**GVHD:** Thầy Nguyễn Thế Khanh\n\n**Học sinh:**\n- Bùi Tá Tùng\n- Cao Sỹ Bảo Chung")

# --- PROMPT KỸ SƯ (SYSTEM INSTRUCTION) ---
SYSTEM_PROMPT = """
Bạn là "Chatbot KTC", trợ lý ảo hỗ trợ học tập môn Tin học theo Chương trình GDPT 2018 (Bộ sách Kết nối tri thức, Cánh Diều, Chân trời sáng tạo).
Phong cách trả lời:
1. Sư phạm, dễ hiểu, thân thiện như một giáo viên giỏi.
2. Luôn ưu tiên thông tin được cung cấp trong phần "BỐI CẢNH TRA CỨU".
3. Nếu thông tin có trong BỐI CẢNH, hãy trích dẫn nguồn (Ví dụ: Theo SGK Tin học 10...).
4. Nếu BỐI CẢNH không chứa thông tin trả lời, hãy dùng kiến thức của bạn nhưng phải nói rõ: "Thông tin này không có trong tài liệu tham khảo, nhưng theo kiến thức của tôi thì...".
"""

# --- XỬ LÝ CHAT ---
# 1. Hiển thị lịch sử
for message in st.session_state.messages:
    role_icon = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=role_icon):
        st.markdown(message["content"])

# 2. Nhận câu hỏi
prompt = st.chat_input("Nhập câu hỏi về môn Tin học (VD: Mạng máy tính là gì?)...")

if prompt:
    # Thêm câu hỏi vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # --- LOGIC RAG (RETRIEVAL) ---
    context_text = ""
    sources_list = []
    
    if st.session_state.vector_db:
        # Tìm kiếm 3 đoạn văn bản tương đồng nhất (Semantic Search)
        # k=3 nghĩa là lấy 3 đoạn liên quan nhất
        results = st.session_state.vector_db.similarity_search(prompt, k=3)
        
        if results:
            for doc in results:
                context_text += f"\n---\nNội dung: {doc.page_content}\nNguồn: {doc.metadata['source']} (Trang {doc.metadata['page']})"
                sources_list.append(f"{doc.metadata['source']} (Trang {doc.metadata['page']})")

    # --- TẠO PROMPT CUỐI CÙNG GỬI CHO LLM ---
    final_prompt = f"""
    {SYSTEM_PROMPT}
    
    --- BẮT ĐẦU BỐI CẢNH TRA CỨU (THÔNG TIN TỪ SGK) ---
    {context_text if context_text else "Không tìm thấy thông tin liên quan trong tài liệu."}
    --- KẾT THÚC BỐI CẢNH ---
    
    Câu hỏi của học sinh: {prompt}
    """

    # --- GỌI API GROQ ---
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
                temperature=0.3 # Giảm độ sáng tạo để tăng độ chính xác
            )

            for chunk in chat_completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    placeholder.markdown(full_response + "▌")
            
            # Thêm phần trích dẫn nguồn vào cuối câu trả lời (Điểm cộng cho KHKT)
            if sources_list:
                citation_text = "\n\n---\nBadges: *" + ", ".join(list(set(sources_list))) + "*"
                full_response += citation_text
                
            placeholder.markdown(full_response)
            
            # Lưu vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Đã xảy ra lỗi kết nối: {e}")