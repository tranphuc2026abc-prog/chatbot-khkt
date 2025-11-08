# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: pip install groq streamlit pypdf scikit-learn
# (Lưu ý: Pypdf và Scikit-learn là BẮT BUỘC để RAG hoạt động)

import streamlit as st
from groq import Groq
import os
import glob
import time

# --- THƯ VIỆN BẮT BUỘC CHO RAG ---
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# --- KẾT THÚC THƯ VIỆN RAG ---


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
Khi học sinh hỏi về mục lục sách (ví dụ: Tin 10 KNTT, Tin 11 CD), bạn PHẢI cung cấp thông tin sau:

# --- DỮ LIỆU LỚP 10 (ĐÃ BỔ SUNG) ---
* **Sách Tin học 10 – KẾT NỐI TRI THỨC VỚI CUỘC SỐNG (KNTT)** gồm các Chủ đề chính:
    1. Chủ đề 1: Máy tính và xã hội tri thức
    2. Chủ đề 2: Mạng máy tính và Internet
    3. Chủ đề 3: Đạo đức, pháp luật và văn hoá trong môi trường số
    4. Chủ đề 4: Ứng dụng tin học (Thiết kế đồ họa)
    5. Chủ đề 5: Giải quyết vấn đề với sự trợ giúp của máy tính (Lập trình Python)
    6. Chủ đề 6: Hướng nghiệp với tin học

* **Sách Tin học 10 – CÁNH DIỀU (CD)** gồm các Chủ đề chính:
    1. Chủ đề A: Máy tính và xã hội tri thức
    2. Chủ đề B: Mạng máy tính và Internet
    3. Chủ đề D: Đạo đức, pháp luật và văn hóa trong môi trường số
    4. Chủ đề E: Ứng dụng tin học (Thiết kế đồ họa)
    5. Chủ đề F: Giải quyết vấn đề với sự trợ giúp của máy tính (Lập trình Python)
    6. Chủ đề G: Hướng nghiệp với tin học

* **Sách Tin học 10 – CHÂN TRỜI SÁNG TẠO (CTST)** gồm các Chủ đề chính:
    1. Chủ đề 1: Máy tính và xã hội
    2. Chủ đề 2: Mạng máy tính và Internet
    3. Chủ đề 3: Đạo đức, pháp luật và văn hóa trong môi trường số
    4. Chủ đề 4: Ứng dụng tin học (Phần mềm đồ họa)
    5. Chủ đề 5: Giải quyết vấn đề với sự trợ giúp của máy tính (Lập trình Python)
    6. Chủ đề 6: Hướng nghiệp

# --- DỮ LIỆU LỚP 11 (ĐÃ BỔ SUNG) ---
* **Sách Tin học 11 – KẾT NỐI TRI THỨC VỚI CUỘC SỐNG (KNTT)** gồm các Chủ đề chính:
    1. Chủ đề 1: Máy tính và xã hội tri thức (Hệ điều hành, Phần mềm...)
    2. Chủ đề 2: Tổ chức lưu trữ, tìm kiếm và trao đổi thông tin
    3. Chủ đề 3: Đạo đức, pháp luật và văn hóa trong môi trường số
    4. Chủ đề 4: Giới thiệu các hệ cơ sở dữ liệu (CSDL)
    5. (Và các chuyên đề CS/ICT như Lập trình, Đồ họa/Đa phương tiện)

* **Sách Tin học 11 – CÁNH DIỀU (CD)** gồm các Chủ đề chính:
    1. Chủ đề A: Máy tính và xã hội tri thức (Bên trong máy tính, HĐH...)
    2. Chủ đề C: Tổ chức lưu trữ, tìm kiếm và trao đổi thông tin
    3. Chủ đề F: Giới thiệu các hệ cơ sở dữ liệu (CSDL)
    4. (Và các chuyên đề CS/ICT)

* **Sách Tin học 11 – CHÂN TRỜI SÁNG TẠO (CTST)** gồm các Chủ đề chính:
    1. Chủ đề 1: Máy tính và xã hội tri thức (Hệ điều hành...)
    2. Chủ đề 2: Tổ chức lưu trữ, tìm kiếm và trao đổi thông tin
    3. Chủ đề 3: Đạo đức, pháp luật và văn hóa trong môi trường số
    4. Chủ đề 4: Giới thiệu các hệ cơ sở dữ liệu (CSDL)
    5. (Và các chuyên đề CS/ICT)

# --- DỮ LIỆU LỚP 12 (CÓ SẴN) ---
* **Sách Tin học 12 – KẾT NỐI TRI THỨC VỚI CUỘC SỐNG (KNTT)** gồm 5 Chủ đề chính:
    1.  Chủ đề 1: Máy tính và xã hội tri thức (Ví dụ: Công nghệ, AI)
    2.  Chủ đề 2: Đạo đức, pháp luật và văn hóa trong không gian số
    3.  Chủ đề 3: Hệ cơ sở dữ liệu (Ví dụ: CSDL, Hệ quản trị CSDL)
    4.  Chủ đề 4: Lập trình và ứng dụng (Ví dụ: Cấu trúc dữ liệu cơ bản, Thư viện lập trình)
    5.  Chủ đề 5: Mạng máy tính và Internet (Ví dụ: Mạng máy tính, Bảo mật mạng)
(Và các sách khác...)
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

---
[PHẦN QUAN TRỌNG] XỬ LÝ THÔNG TIN TRA CỨU (RAG)
---
Khi nhận được thông tin trong một tin nhắn hệ thống bắt đầu bằng "--- BẮT ĐẦU DỮ LIỆU TRA CỨU TỪ 'SỔ TAY' (RAG) ---", bạn **PHẢI** tuân thủ các quy tắc sau:

1.  **ƯU TIÊN TUYỆT ĐỐI:** Dữ liệu này là nguồn "chân lý" (ground truth) từ Sổ tay Tin học. Bạn **PHẢI** ưu tiên sử dụng thông tin này để trả lời câu hỏi của người dùng.
2.  **TRÍCH DẪN (NẾU CÓ THỂ):** Nếu câu trả lời của bạn dựa trực tiếp vào "NGUỒN" được cung cấp, hãy cố gắng trích dẫn ngắn gọn (ví dụ: "Theo tài liệu,..." hoặc "Như trong Sổ tay có đề cập...").
3.  **TỔNG HỢP:** Nếu các NGUỒN cung cấp thông tin rời rạc, hãy tổng hợp chúng lại thành một câu trả lời mạch lạc.
4.  **KHÔNG BỊA ĐẶT:** Nếu thông tin tra cứu có vẻ không liên quan đến câu hỏi, hãy lịch sự thông báo rằng bạn không tìm thấy thông tin chính xác trong Sổ tay và trả lời dựa trên kiến thức chung của bạn.

#... (Giữ nguyên các phần còn lại của System Prompt) ...
"""

# --- BƯỚC 3: KHỞI TẠO CLIENT VÀ CHỌN MÔ HÌNH ---
try:
    client = Groq(api_key=api_key) 
except Exception as e:
    st.error(f"Lỗi khi cấu hình API Groq: {e}")
    st.stop()

MODEL_NAME = 'llama-3.1-8b-instant'

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
        # st.session_state.pop("knowledge_data", None) # Không cần xóa cache RAG mỗi lần chat mới
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


# --- BƯỚC 4.6: CÁC HÀM RAG (ĐÃ KÍCH HOẠT) --- #

@st.cache_data(ttl=3600) 
def load_and_process_pdfs(pdf_folder="data_pdf"):
    """
    Tải tất cả file PDF từ một thư mục, trích xuất văn bản theo từng trang,
    và tạo ra ma trận TF-IDF cũng như vectorizer.
    """
    print(f"Bắt đầu quét thư mục: {pdf_folder}")
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print("CẢNH BÁO: Không tìm thấy file PDF nào trong thư mục 'data_pdf'. RAG sẽ không hoạt động.")
        return [], None, None # Trả về bộ rỗng

    chunks = []
    for pdf_path in pdf_files:
        print(f"Đang xử lý file: {pdf_path}")
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Thêm thông tin nguồn (tên file, số trang) vào chunk
                    source_info = f"[Nguồn: {os.path.basename(pdf_path)}, Trang {page_num + 1}]"
                    chunks.append(f"{source_info}\n\n{text}")
        except Exception as e:
            print(f"Lỗi khi đọc file {pdf_path}: {e}")

    if not chunks:
        print("Không thể trích xuất nội dung từ các file PDF.")
        return [], None, None

    print(f"Đã trích xuất {len(chunks)} trang PDF. Bắt đầu vector hóa (TF-IDF)...")
    
    # Bắt đầu vector hóa
    try:
        vectorizer = TfidfVectorizer(
            stop_words=None, # Có thể thêm stop_words tiếng Việt nếu muốn
            ngram_range=(1, 2) # Xem xét cả cụm 1 và 2 từ
        )
        tfidf_matrix = vectorizer.fit_transform(chunks)
        print("Vector hóa hoàn tất.")
        
        # Trả về 3 đối tượng: danh sách chunks, ma trận TF-IDF, và bộ vector hóa
        return chunks, tfidf_matrix, vectorizer
    
    except ValueError as e:
        if "empty vocabulary" in str(e):
            st.error(f"Lỗi RAG: Các file PDF có thể không chứa văn bản (chỉ chứa ảnh). Vui lòng kiểm tra file.")
            return [], None, None
        else:
            raise e


def find_relevant_knowledge(query, chunks, tfidf_matrix, vectorizer, num_chunks=3):
    """
    Tìm các chunks (trang) liên quan nhất đến câu hỏi bằng TF-IDF và Cosine Similarity.
    """
    # 1. Kiểm tra xem RAG có dữ liệu không
    if not chunks or tfidf_matrix is None or vectorizer is None:
        return [] # Không có dữ liệu RAG để tìm kiếm

    # 2. Vector hóa câu hỏi của người dùng
    query_vector = vectorizer.transform([query])
    
    # 3. Tính toán độ tương đồng cosine
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # 4. Lấy N chunks có điểm cao nhất
    # Chỉ lấy những chunks có điểm > 0 (có liên quan)
    relevant_indices = np.where(cosine_similarities > 0)[0]
    
    # Sắp xếp các chỉ mục này theo điểm số giảm dần
    sorted_indices = sorted(relevant_indices, key=lambda i: cosine_similarities[i], reverse=True)
    
    # Lấy N chỉ mục hàng đầu (hoặc ít hơn nếu không đủ)
    top_indices = sorted_indices[:num_chunks]

    if not top_indices:
        return [] # Không tìm thấy chunk nào có độ tương đồng > 0
        
    # 5. Trả về nội dung của các chunks đó
    relevant_chunks = [chunks[i] for i in top_indices]
    return relevant_chunks


# --- BƯỚC 5: KHỞI TẠO LỊCH SỬ CHAT VÀ "SỔ TAY" PDF (RAG ĐÃ MỞ) --- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- KÍCH HOẠT RAG ---
# Tải và xử lý PDF khi app khởi động (chỉ chạy 1 lần)
if "knowledge_data" not in st.session_state:
    with st.spinner("👩‍🏫 Em đang đọc 'Sổ tay Tin học' (PDF)..."):
        # Hàm load_and_process_pdfs trả về (chunks, matrix, vectorizer)
        st.session_state.knowledge_data = load_and_process_pdfs()
        print("RAG (Đọc PDF) đã được tải và xử lý.")
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

    # 2. Gửi câu hỏi đến Groq (ĐÃ BAO GỒM RAG)
    try:
        with st.chat_message("assistant", avatar="✨"):
            placeholder = st.empty()
            bot_response_text = ""

            # --- PHẦN RAG ĐÃ KÍCH HOẠT --- #
            
            # 2.1. Lấy dữ liệu RAG đã cache
            chunks, tfidf_matrix, vectorizer = st.session_state.knowledge_data
            
            # 2.2. Tìm kiếm kiến thức liên quan
            # (Hàm này sẽ trả về [] nếu không có dữ liệu RAG)
            retrieved_context = find_relevant_knowledge(prompt, chunks, tfidf_matrix, vectorizer, num_chunks=3)
            
            # 2.3. Chuẩn bị danh sách tin nhắn gửi cho AI
            messages_to_send = [
                {"role": "system", "content": SYSTEM_INSTRUCTION}
            ]
            
            # 2.4. (QUAN TRỌNG) Chèn Context RAG vào tin nhắn
            if retrieved_context:
                print(f"Đã tìm thấy {len(retrieved_context)} mẩu kiến thức RAG cho câu hỏi.")
                # Tạo một tin nhắn "system" đặc biệt để chứa kiến thức
                context_message = (
                    "--- BẮT ĐẦU DỮ LIỆU TRA CỨU TỪ 'SỔ TAY' (RAG) ---\n"
                    "Đây là thông tin bổ sung từ 'Sổ tay Tin học' của bạn. "
                    "Hãy sử dụng thông tin này làm NGUỒN ƯU TIÊN để trả lời câu hỏi của người dùng.\n\n"
                )
                for i, chunk_text in enumerate(retrieved_context):
                    context_message += f"--- NGUỒN {i+1} ---\n{chunk_text}\n\n"
                context_message += "--- KẾT THÚC DỮ LIỆU TRA CỨU ---\n"
                
                # Thêm tin nhắn context này vào *trước* lịch sử chat
                messages_to_send.append({"role": "system", "content": context_message})
            else:
                print("Không tìm thấy kiến thức RAG liên quan. Trả lời bình thường.")

            # 2.5. Thêm toàn bộ lịch sử chat (bao gồm cả câu hỏi mới nhất)
            messages_to_send.extend(st.session_state.messages)
            
            # --- KẾT THÚC PHẦN RAG --- #

            # 2.6. Gọi API Groq
            stream = client.chat.completions.create(
                messages=messages_to_send, # Gửi tin nhắn ĐÃ BAO GỒM RAG
                model=MODEL_NAME,
                stream=True
            )
            
            # 2.7. Lặp qua từng "mẩu" (chunk) API trả về
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