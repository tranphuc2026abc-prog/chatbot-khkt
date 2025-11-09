# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: 
# pip install groq streamlit pypdf langchain langchain-text-splitters scikit-learn numpy
# (Lưu ý: Các thư viện pypdf, langchain, scikit-learn là BẮT BUỘC để RAG hoạt động)

import streamlit as st
from groq import Groq
import os
import glob
import time
from pypdf import PdfReader # <-- ĐÃ THÊM: Thư viện đọc PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- ĐÃ SỬA: Import từ gói riêng
from sklearn.feature_extraction.text import TfidfVectorizer # <-- ĐÃ THÊM: Vector hóa
from sklearn.metrics.pairwise import cosine_similarity # <-- ĐÃ THÊM: Tính tương đồng
import numpy as np # <-- ĐÃ THÊM: Hỗ trợ tính toán

# --- BƯỚC 1: LẤY API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Lỗi: Không tìm thấy GROQ_API_KEY. Vui lòng thêm vào Secrets trên Streamlit Cloud.")
    st.stop()
    
# --- BƯỚC 2: THIẾT LẬP VAI TRÒ (SYSTEM_INSTRUCTION) ---
SYSTEM_INSTRUCTION = """
---
BỐI CẢNH VAI TRÒ (ROLE CONTEXT)
---
Bạn là “Chatbook”, một Cố vấn Học tập Tin học AI toàn diện.
Vai trò của bạn được mô phỏng theo một **Giáo viên Tin học dạy giỏi cấp Quốc gia**: tận tâm, hiểu biết sâu rộng, và luôn kiên nhẫn.
Mục tiêu của bạn là đồng hành, hỗ trợ học sinh THCS và THPT (từ lớp 6 đến lớp 12) nắm vững kiến thức, phát triển năng lực Tin học theo **Chuẩn chương trình Giáo dục Phổ thông 2018** của Việt Nam.

---
📚 NỀN TẢNG TRI THỨC CỐT LÕI (CORE KNOWLEDGE BASE) - BẮT BUỘC
---
Bạn **PHẢI** nắm vững và sử dụng thành thạo toàn bộ hệ thống kiến thức trong Sách giáo khoa Tin học từ lớp 6 đến lớp 12 của **CẢ BA BỘ SÁCH HIỆN HÀNH**:
1.  **Kết nối tri thức với cuộc sống (KNTT)**
2.  **Cánh Diều (CD)**
3.  **Chân trời sáng tạo (CTST)**

Khi giải thích khái niệm hoặc hướng dẫn kỹ năng, bạn phải ưu tiên cách tiếp cận, thuật ngữ, và ví dụ được trình bày trong các bộ sách này để đảm bảo tính thống nhất và bám sát chương trình, tránh nhầm lẫn.

*** DỮ LIỆU MỤC LỤC CHUYÊN BIỆT (KHẮC PHỤC LỖI) ***
Khi học sinh hỏi về mục lục sách (ví dụ: Tin 12 KNTT), bạn PHẢI cung cấp thông tin sau:

* **Sách Tin học 12 – KẾT NỐI TRI THỨC VỚI CUỘC SỐNG (KNTT)** (ĐÃ CẬP NHẬT) gồm 5 Chủ đề chính:
    1.  **Chủ đề 1:** Máy tính và Xã hội tri thức
    2.  **Chủ đề 2:** Mạng máy tính và Internet
    3.  **Chủ đề 3:** Đạo đức, pháp luật và văn hoá trong môi trường số
    4.  **Chủ đề 4:** Giải quyết vấn đề với sự trợ giúp của máy tính
    5.  **Chủ đề 5:** Hướng nghiệp với Tin học

* **Sách Tin học 12 – CHÂN TRỜI SÁNG TẠO (CTST)** (GIỮ NGUYÊN) gồm các Chủ đề chính:
    1.  **Chủ đề 1:** Máy tính và cộng đồng
    2.  **Chủ đề 2:** Tổ chức và lưu trữ dữ liệu
    3.  **Chủ đề 3:** Đạo đức, pháp luật và văn hóa trong môi trường số
    4.  **Chủ đề 4:** Giải quyết vấn đề với sự hỗ trợ của máy tính
    5.  **Chủ đề 5:** Mạng máy tính và Internet

* **Sách Tin học 12 – CÁNH DIỀU (CD)** (GIỮ NGUYÊN) gồm các Chủ đề chính:
    1.  **Chủ đề 1:** Máy tính và Xã hội
    2.  **Chủ đề 2:** Mạng máy tính và Internet
    3.  **Chủ đề 3:** Thuật toán và Lập trình
    4.  **Chủ đề 4:** Dữ liệu và Hệ thống thông tin
    5.  **Chủ đề 5:** Ứng dụng Tin học
*** KẾT THÚC DỮ LIỆU CHUYÊN BIỆT ***

---
🌟 6 NHIỆM VỤ CỐT LÕI (CORE TASKS)
---
#... (Giữ nguyên các nhiệm vụ từ 1 đến 6) ...

**1. 👨‍🏫 Gia sư Chuyên môn (Specialized Tutor):**
    - Giải thích các khái niệm (ví dụ: thuật toán, mạng máy tính, CSGD, CSDL) một cách trực quan, sư phạm, sử dụng ví dụ gần gũi với lứa tuổi học sinh.
    - Luôn kết nối lý thuyết với thực tiễn, giúp học sinh thấy được "học cái này để làm gì?".
    - Bám sát nội dung Sách giáo khoa (KNTT, CD, CTST) và yêu cầu cần đạt của Ctr 2018.
#... (Giữ nguyên các nhiệm vụ còn lại) ...
#... (Giữ nguyên phần QUY TẮC ỨNG XỬ & PHONG CÁCH) ...
#... (Giữ nguyên phần XỬ LÝ THÔNG TIN TRA CỨU) ...
#... (GiNếu có thông tin tra cứu từ 'sổ tay' (RAG), BẠN PHẢI ưu tiên sử dụng thông tin đó.) ...
#... (Giữ nguyên phần LỚP TƯ DUY PHẢN BIỆN AI) ...
#... (Giữ nguyên phần MỤC TIÊU CUỐI CÙNG) ...
"""

# --- BƯỚC 3: KHỞI TẠO CLIENT VÀ CHỌN MÔ HÌNH ---
try:
    client = Groq(api_key=api_key) 
except Exception as e:
    st.error(f"Lỗi khi cấu hình API Groq: {e}")
    st.stop()

MODEL_NAME = 'llama-3.1-8b-instant'
PDF_DIR = "./PDF_KNOWLEDGE" # <-- ĐÃ THÊM: ĐƯỜNG DẪN ĐẾN THƯ MỤC CHỨA CÁC FILE PDF "SỔ TAY"

# --- BƯỚC 4: CẤU HÌNH TRANG VÀ CSS ---
st.set_page_config(page_title="Chatbot Tin học 2018", page_icon="✨", layout="centered")
st.markdown("""
<style>
    /* ... (Toàn bộ CSS của thầy giữ nguyên) ... */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stSidebar"] {
        background-color: #f8f9fa; border-right: 1px solid #e6e6e6;
    }
    .main .block-container { 
        max-width: 850px; padding-top: 2rem; padding-bottom: 5rem;
    }
    .welcome-message { font-size: 1.1em; color: #333; }
</style>
""", unsafe_allow_html=True)


# --- BƯỚC 4.5: THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.title("🤖 Chatbot KTC")
    st.markdown("---")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        # Xóa cache RAG để tải lại nếu cần
        st.session_state.pop("rag_system", None) 
        st.cache_resource.clear() # Xóa cache resource
        st.rerun()

    st.markdown("---")
    st.markdown(
        "Giáo viên hướng dẫn:\n"
        "**Thầy Nguyễn Thế Khanh** (GV Tin học)\n\n"
        "Học sinh thực hiện:\n"
        "*(Bùi Tá Tùng)*\n"
        "*(Cao Sỹ Bảo Chung)*"
    )
    st.markdown("---")
    st.caption(f"Model: {MODEL_NAME}")


# --- BƯỚC 4.6: CÁC HÀM RAG (ĐỌC "SỔ TAY" TỪ PDF) --- #
# <-- ĐÃ SỬA: Cập nhật các hàm RAG để hoạt động

@st.cache_resource(ttl=3600) # Dùng cache_resource cho các đối tượng (như vectorizer)
def initialize_rag_system(pdf_directory=PDF_DIR):
    """
    Hàm này sẽ quét thư mục PDF, đọc, chia nhỏ và tạo chỉ mục TF-IDF.
    Nó được cache lại để chỉ chạy một lần mỗi giờ hoặc khi cache bị xóa.
    Trả về: (vectorizer, tfidf_matrix, all_chunks) hoặc (None, None, None) nếu lỗi.
    """
    print("--- BẮT ĐẦU KHỞI TẠO HỆ THỐNG RAG (CHẠY LẦN ĐẦU) ---")
    
    # 1. Tải và chia nhỏ PDF
    all_chunks = []
    try:
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
        
        if not pdf_files:
            print(f"!!! CẢNH BÁO RAG: Không tìm thấy file PDF nào trong thư mục '{pdf_directory}'.")
            st.warning(f"Tính năng RAG (đọc sổ tay) đã bật, nhưng không tìm thấy file PDF nào trong thư mục `{pdf_directory}`. Vui lòng tạo thư mục và thêm PDF vào.", icon="⚠️")
            return None, None, None # Trả về None

        print(f"Tìm thấy {len(pdf_files)} file PDF. Đang xử lý...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=150,
            length_function=len
        )
        
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                raw_text = "".join(page.extract_text() or "" for page in reader.pages)
                if raw_text:
                    chunks = text_splitter.split_text(raw_text)
                    all_chunks.extend(chunks)
                    print(f"Đã xử lý: {os.path.basename(pdf_path)} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"Lỗi khi đọc file {pdf_path}: {e}")
                st.error(f"Lỗi đọc file PDF: {os.path.basename(pdf_path)}")

        if not all_chunks:
            print("!!! CẢNH BÁO RAG: Đã đọc file PDF nhưng không trích xuất được nội dung.")
            st.warning("Đã tìm thấy file PDF nhưng không thể trích xuất nội dung. RAG sẽ không hoạt động.", icon="⚠️")
            return None, None, None
        
        print(f"Tổng cộng {len(all_chunks)} khối kiến thức. Đang tạo chỉ mục TF-IDF...")
        
        # 2. Vector hóa (TF-IDF)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_chunks)
        print("--- HOÀN TẤT KHỞI TẠO RAG ---")
        
        # Trả về cả 3: vectorizer, ma trận tfidf, và list các chunk
        return vectorizer, tfidf_matrix, all_chunks
        
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi khởi tạo RAG: {e}")
        st.error(f"Lỗi khởi tạo RAG: {e}")
        return None, None, None

def find_relevant_knowledge(query, vectorizer, tfidf_matrix, all_chunks, num_chunks=3):
    """
    Tìm kiếm các chunk liên quan nhất bằng TF-IDF và cosine similarity.
    """
    if vectorizer is None or tfidf_matrix is None or not all_chunks:
        return None # RAG không được khởi tạo
        
    print(f"--- RAG ĐANG TÌM KIẾM CHO QUERY: '{query[:50]}...' ---")
    try:
        # 1. Vector hóa câu query
        query_vector = vectorizer.transform([query])
        
        # 2. Tính toán độ tương đồng cosine
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # 3. Lọc ra N chunk có điểm cao nhất và > 0
        # Lấy ra index của N*2 chunk cao nhất (để dự phòng)
        k = min(num_chunks * 2, len(cosine_similarities)) 
        if k <= 0: return None # Không có chunk nào

        # Lấy index của các chunk có điểm cao nhất (chưa sắp xếp)
        relevant_indices_partitioned = np.argpartition(cosine_similarities, -k)[-k:]
        
        # Lọc ra những chunk có điểm > 0.05 (ngưỡng lọc nhiễu)
        top_scores_indices = [
            i for i in relevant_indices_partitioned 
            if cosine_similarities[i] > 0.05 
        ]
        
        # Sắp xếp lại theo điểm số thực (từ cao đến thấp)
        top_scores_indices.sort(key=lambda i: cosine_similarities[i], reverse=True)
        
        # Lấy top N (num_chunks)
        final_indices = top_scores_indices[:num_chunks]
        
        if not final_indices:
            print("RAG không tìm thấy chunk nào đủ liên quan.")
            return None
            
        # 4. Trả về nội dung các chunk
        relevant_chunks = [all_chunks[i] for i in final_indices]
        print(f"RAG tìm thấy {len(relevant_chunks)} chunk liên quan.")
        return "\n---\n".join(relevant_chunks)
        
    except Exception as e:
        print(f"Lỗi khi tìm kiếm RAG: {e}")
        return None


# --- BƯỚC 5: KHỞI TẠO LỊCH SỬ CHAT VÀ "SỔ TAY" PDF --- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- ĐÃ KÍCH HOẠT RAG (ĐỌC "SỔ TAY" PDF) --- # <-- ĐÃ SỬA
# Tải và xử lý PDF khi app khởi động (hoặc khi cache bị xóa)
if "rag_system" not in st.session_state:
    with st.spinner("Đang khởi tạo và lập chỉ mục 'sổ tay' PDF (RAG)..."):
        # Hàm này trả về (vectorizer, tfidf_matrix, all_chunks)
        rag_components = initialize_rag_system(PDF_DIR)
        # Lưu cả 3 vào một biến session state
        st.session_state.rag_system = rag_components
        
# Lấy ra các thành phần RAG từ session state (dù mới khởi tạo hay đã có)
# Thêm kiểm tra phòng trường hợp rag_components là (None, None, None)
if "rag_system" in st.session_state and st.session_state.rag_system:
    vectorizer, tfidf_matrix, all_chunks = st.session_state.rag_system
    if all_chunks:
        print(f"Đã tải {len(all_chunks)} khối kiến thức vào cache.")
    else:
        print("Hệ thống RAG đã khởi tạo nhưng không có kiến thức (PDF).")
else:
    # Xử lý trường hợp không có PDF hoặc RAG lỗi
    vectorizer, tfidf_matrix, all_chunks = None, None, None
    print("RAG không hoạt động (không có file PDF hoặc lỗi khởi tạo).")
# --- KẾT THÚC KÍCH HOẠT RAG ---


# --- BƯỚC 6: HIỂN THỊ LỊCH SỬ CHAT ---
for message in st.session_state.messages:
    avatar = "✨" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- BƯỚC 7: MÀN HÌNH CHÀO MỪNG VÀ GỢI Ý ---
logo_path = "LOGO.jpg" 
col1, col2 = st.columns([1, 5])
with col1:
    try:
        st.image(logo_path, width=80)
    except Exception as e:
        st.error(f"Lỗi: Không tìm thấy file logo tên là '{logo_path}'. Vui lòng kiểm tra lại tên file trên GitHub.")
        st.stop()
with col2:
    st.title("KTC. Chatbot hỗ trợ môn Tin Học")

def set_prompt_from_suggestion(text):
    st.session_state.prompt_from_button = text

if not st.session_state.messages:
    st.markdown(f"<div class='welcome-message'>Xin chào! Thầy/em cần hỗ trợ gì về môn Tin học (Chương trình 2018)?</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # ... (Toàn bộ các nút bấm gợi ý của thầy giữ nguyên) ...
    col1_btn, col2_btn = st.columns(2)
    with col1_btn:
        st.button(
            "Giải thích về 'biến' trong lập trình?",
            on_click=set_prompt_from_suggestion, args=("Giải thích về 'biến' trong lập trình?",),
            use_container_width=True
        )
        st.button(
            "Trình bày về an toàn thông tin?",
            on_click=set_prompt_from_suggestion, args=("Trình bày về an toàn thông tin?",),
            use_container_width=True
        )
    with col2_btn:
        st.button(
            "Sự khác nhau giữa RAM và ROM?",
            on_click=set_prompt_from_suggestion, args=("Sự khác nhau giữa RAM và ROM?",),
            use_container_width=True
        )
        st.button(
            "Các bước chèn ảnh vào word",
            on_click=set_prompt_from_suggestion, args=("Các bước chèn ảnh vào word?",),
            use_container_width=True
        )


# --- BƯỚC 8: XỬ LÝ INPUT (ĐÃ KÍCH HOẠT RAG PDF) --- # <--- ĐÃ CẬP NHẬT
prompt_from_input = st.chat_input("Mời thầy hoặc các em đặt câu hỏi về Tin học...")
prompt_from_button = st.session_state.pop("prompt_from_button", None)
prompt = prompt_from_button or prompt_from_input

if prompt:
    # 1. Thêm câu hỏi của user vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Gửi câu hỏi đến Groq
    try:
        with st.chat_message("assistant", avatar="✨"):
            placeholder = st.empty()
            bot_response_text = ""

            # --- ĐÃ KÍCH HOẠT LẠI LOGIC RAG --- # <-- ĐÃ SỬA

            # 2.1. Lấy các thành phần RAG (đã được tải ở BƯỚC 5)
            # (biến vectorizer, tfidf_matrix, all_chunks đã tồn tại ở global scope của script)
            
            # 2.2. Tìm kiếm trong kho kiến thức PDF
            retrieved_context = None
            if all_chunks: # Chỉ tìm nếu có kiến thức (tức là all_chunks không rỗng)
                retrieved_context = find_relevant_knowledge(
                    prompt, vectorizer, tfidf_matrix, all_chunks, num_chunks=3
                )

            # 2.3. Chuẩn bị list tin nhắn gửi cho AI
            messages_to_send = [
                {"role": "system", "content": SYSTEM_INSTRUCTION}
            ]

            # 2.4. Xây dựng prompt dựa trên việc có tìm thấy RAG hay không
            if retrieved_context:
                print("--- RAG ĐÃ TÌM THẤY KIẾN THỨC ---")
                
                # Tạo một "bản sao" của lịch sử chat để chèn RAG
                # Chỉ lấy N tin nhắn cuối để tiết kiệm token (ví dụ 6 tin nhắn)
                temp_messages = list(st.session_state.messages[:-1])[-6:]
                
                # Lấy câu hỏi cuối cùng của user (là "prompt" hiện tại)
                last_user_message_content = st.session_state.messages[-1]['content']
                
                # Tạo prompt RAG
                rag_prompt = f"""
---
BỐI CẢNH TRA CỨU TỪ SỔ TAY (RAG):
{retrieved_context}
---
DỰA VÀO BỐI CẢNH TRÊN (nếu liên quan), hãy trả lời câu hỏi sau đây một cách sư phạm và chi tiết:
Câu hỏi: "{last_user_message_content}"
"""
                # Thêm lại các tin nhắn cũ
                messages_to_send.extend(temp_messages)
                # Thêm prompt RAG mới
                messages_to_send.append({"role": "user", "content": rag_prompt})
                
                print("Đã gửi prompt RAG cho AI.")

            else:
                # RAG không tìm thấy gì, hoặc RAG bị tắt
                print("RAG không tìm thấy gì. Trả lời bình thường.")
                # Gửi toàn bộ lịch sử chat (hoặc N tin nhắn cuối)
                messages_to_send.extend(st.session_state.messages[-10:]) # Gửi 10 tin nhắn cuối

            # --- KẾT THÚC LOGIC RAG --- #

            # 2.5. Gọi API Groq
            stream = client.chat.completions.create(
                messages=messages_to_send, # Gửi list tin nhắn đã xử lý RAG
                model=MODEL_NAME,
                stream=True,
                max_tokens=4096 # Tăng giới hạn token
            )
            
            # 2.6. Lặp qua từng "mẩu" (chunk) API trả về
            for chunk in stream:
                if chunk.choices[0].delta.content is not None: 
                    bot_response_text += chunk.choices[0].delta.content
                    placeholder.markdown(bot_response_text + "▌")
                    time.sleep(0.005) # <--- Tạo hiệu ứng
            
            placeholder.markdown(bot_response_text) # Xóa dấu ▌ khi hoàn tất

    except Exception as e:
        with st.chat_message("assistant", avatar="✨"):
            st.error(f"Xin lỗi, đã xảy ra lỗi khi kết nối Groq: {e}")
        bot_response_text = ""

    # 3. Thêm câu trả lời của bot vào lịch sử
    if bot_response_text:
        st.session_state.messages.append({"role": "assistant", "content": bot_response_text})

    # 4. Rerun nếu bấm nút
    if prompt_from_button:
        st.rerun()