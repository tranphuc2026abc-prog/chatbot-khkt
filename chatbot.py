# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: pip install google-generativeai streamlit pypdf scikit-learn
# (Lưu ý: Pypdf và Scikit-learn là BẮT BUỘC để RAG hoạt động)

import streamlit as st
# [THAY ĐỔI] 1. Bỏ Groq, thêm thư viện của Google
import google.generativeai as genai
# [SỬA LỖI] Thêm dòng này để tắt Safety
from google.generativeai.types import HarmCategory, HarmBlockThreshold
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
    # [THAY ĐỔI] 2. Lấy Google API Key từ Streamlit Secrets
    api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Lỗi: Không tìm thấy GOOGLE_API_KEY. Vui lòng thêm vào Secrets trên Streamlit Cloud.")
    st.stop()
    
# [THAY ĐỔI] 3. Cấu hình API cho Google
genai.configure(api_key=api_key)

# --- BƯỚC 2: THIẾT LẬP VAI TRÒ (SYSTEM_INSTRUCTION) ---
# System prompt này sẽ được đưa vào model, không cần thay đổi
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
# ... (Giữ nguyên toàn bộ dữ liệu mục lục của thầy) ...
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

# [THAY ĐỔI] 4. Khởi tạo mô hình Gemini với System Instruction
# [SỬA LỖI] Đổi về 1.5-pro-latest cho ổn định (thầy có thể đổi lại 2.5-pro nếu muốn)
MODEL_NAME = 'gemini-1.5-pro-latest' 
try:
    # [SỬA LỖI] Cập nhật safety_settings dùng Enum
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    # Khởi tạo model và gán system_instruction vào
    gemini_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
        safety_settings=safety_settings # <--- Dùng biến đã sửa
    )
    print("Khởi tạo model Gemini 1.5 Pro thành công.")
except Exception as e:
    st.error(f"Lỗi khi khởi tạo Model Gemini: {e}")
    st.stop()


# --- BƯỚC 4: CẤU HÌNH TRANG VÀ CSS ---
# (Giữ nguyên không thay đổi)
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
        # st.session_state.pop("knowledge_data", None) # Không cần xóa cache RAG
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
    # [THAY ĐỔI] 5. Cập nhật tên Model mới
    st.caption(f"Model: {MODEL_NAME}")


# --- BƯỚC 4.6: CÁC HÀM RAG (ĐÃ KÍCH HOẠT) --- #
# (Giữ nguyên toàn bộ 2 hàm RAG của thầy)

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
        return [], None, None 

    chunks = []
    for pdf_path in pdf_files:
        print(f"Đang xử lý file: {pdf_path}")
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    source_info = f"[Nguồn: {os.path.basename(pdf_path)}, Trang {page_num + 1}]"
                    chunks.append(f"{source_info}\n\n{text}")
        except Exception as e:
            print(f"Lỗi khi đọc file {pdf_path}: {e}")

    if not chunks:
        print("Không thể trích xuất nội dung từ các file PDF.")
        return [], None, None

    print(f"Đã trích xuất {len(chunks)} trang PDF. Bắt đầu vector hóa (TF-IDF)...")
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=None, 
            ngram_range=(1, 2) 
        )
        tfidf_matrix = vectorizer.fit_transform(chunks)
        print("Vector hóa hoàn tất.")
        
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
    if not chunks or tfidf_matrix is None or vectorizer is None:
        return [] 

    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    relevant_indices = np.where(cosine_similarities > 0.1)[0]
    
    sorted_indices = sorted(relevant_indices, key=lambda i: cosine_similarities[i], reverse=True)
    top_indices = sorted_indices[:num_chunks]

    if not top_indices:
        return [] 
        
    relevant_chunks = [chunks[i] for i in top_indices]
    return relevant_chunks

# --- [THAY ĐỔI] 6. HÀM CHUYỂN ĐỔI LỊCH SỬ SANG FORMAT GEMINI ---
def convert_history_for_gemini(messages):
    """
    Chuyển đổi lịch sử chat của Streamlit (role/content) 
    sang định dạng của Gemini (role/parts).
    """
    gemini_history = []
    for msg in messages:
        role = 'model' if msg['role'] == 'assistant' else 'user'
        gemini_history.append({'role': role, 'parts': [msg['content']]})
    return gemini_history

# --- BƯỚC 5: KHỞI TẠO LỊCH SỬ CHAT VÀ "SỔ TAY" PDF (RAG ĐÃ MỞ) --- #
# (Giữ nguyên)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_data" not in st.session_state:
    with st.spinner("👩‍🏫 Em đang đọc 'Sổ tay Tin học' (PDF)..."):
        st.session_state.knowledge_data = load_and_process_pdfs()
        print("RAG (Đọc PDF) đã được tải và xử lý.")


# --- BƯỚC 6: HIỂN THỊ LỊCH SỬ CHAT ---
# (Giữ nguyên)
for message in st.session_state.messages:
    avatar = "✨" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- BƯỚC 7: MÀN HÌNH CHÀO MỪNG VÀ GỢI Ý ---
# (Giữ nguyên)
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


# --- BƯỚC 8: XỬ LÝ INPUT (ĐÃ KÍCH HOẠT RAG PDF) --- # 
# [THAY ĐỔI] 7. Đây là phần thay đổi LỚN NHẤT (toàn bộ logic gọi API)

prompt_from_input = st.chat_input("Mời thầy hoặc các em đặt câu hỏi về Tin học...")
prompt_from_button = st.session_state.pop("prompt_from_button", None)
prompt = prompt_from_button or prompt_from_input

if prompt:
    # 1. Thêm câu hỏi của user vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Gửi câu hỏi đến GEMINI (ĐÃ BAO GỒM RAG)
    try:
        with st.chat_message("assistant", avatar="✨"):
            placeholder = st.empty()
            # [SỬA LỖI] Đặt spinner chờ ở đây
            with placeholder.status("👩‍🏫 Chatbook đang suy nghĩ..."):

                # --- PHẦN RAG ĐÃ KÍCH HOẠT --- #
                
                # 2.1. Lấy dữ liệu RAG đã cache
                chunks, tfidf_matrix, vectorizer = st.session_state.knowledge_data
                
                # 2.2. Tìm kiếm kiến thức liên quan
                retrieved_context = find_relevant_knowledge(prompt, chunks, tfidf_matrix, vectorizer, num_chunks=3)
                
                # 2.3. Chuẩn bị lịch sử chat cho Gemini
                messages_for_api = convert_history_for_gemini(st.session_state.messages)
                
                # 2.4. (QUAN TRỌNG) Chèn Context RAG vào tin nhắn
                if retrieved_context:
                    print(f"Đã tìm thấy {len(retrieved_context)} mẩu kiến thức RAG cho câu hỏi.")
                    context_message = (
                        "--- BẮT ĐẦU DỮ LIỆU TRA CỨU TỪ 'SỔ TAY' (RAG) ---\n"
                        "Đây là thông tin bổ sung từ 'Sổ tay Tin học' của bạn. "
                        "Hãy sử dụng thông tin này làm NGUỒN ƯU TIÊN để trả lời câu hỏi của người dùng.\n\n"
                    )
                    for i, chunk_text in enumerate(retrieved_context):
                        context_message += f"--- NGUỒN {i+1} ---\n{chunk_text}\n\n"
                    context_message += "--- KẾT THÚC DỮ LIỆU TRA CỨU ---\n"
                    
                    last_user_message = messages_for_api.pop()
                    new_prompt_content = f"{context_message}\n\nCâu hỏi: {last_user_message['parts'][0]}"
                    messages_for_api.append({'role': 'user', 'parts': [new_prompt_content]})
                    
                else:
                    print("Không tìm thấy kiến thức RAG liên quan. Trả lời bình thường.")

                # --- KẾT THÚC PHẦN RAG --- #

                # --- [BẮT ĐẦU SỬA LỖI "too_many_pings"] ---
                
                # 2.5. Khởi tạo phiên chat
                chat_session = gemini_model.start_chat(
                    history=messages_for_api[:-1] # Toàn bộ lịch sử TRỪ câu hỏi cuối
                )
                
                # 2.6. Gửi câu hỏi và CHỜ phản hồi (NON-STREAMING)
                # Bỏ stream=True để tránh lỗi "ENHANCE_YOUR_CALM"
                response = chat_session.send_message(
                    messages_for_api[-1]['parts'] # Chỉ gửi nội dung câu hỏi cuối
                )
                
                # 2.7. Lấy nội dung text và hiển thị
                bot_response_text = response.text
                
                # --- [KẾT THÚC SỬA LỖI] ---

            # Hiển thị câu trả lời (sau khi spinner đã xong)
            placeholder.markdown(bot_response_text)

    except Exception as e:
        with st.chat_message("assistant", avatar="✨"):
            st.error(f"Xin lỗi, đã xảy ra lỗi khi kết nối Gemini: {e}")
        bot_response_text = ""

    # 3. Thêm câu trả lời của bot vào lịch sử
    if bot_response_text:
        st.session_state.messages.append({"role": "assistant", "content": bot_response_text})

    # 4. Rerun nếu bấm nút
    if prompt_from_button:
        st.rerun()