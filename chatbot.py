# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: pip install google-generativeai streamlit pypdf scikit-learn
# (Lưu ý: Pypdf và Scikit-learn là BẮT BUỘC để RAG hoạt động)

import streamlit as st
import google.generativeai as genai
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
    api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Lỗi: Không tìm thấy GOOGLE_API_KEY. Vui lòng thêm vào Secrets trên Streamlit Cloud.")
    st.stop()
    
genai.configure(api_key=api_key)

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

#... (Giữ nguyên toàn bộ dữ liệu mục lục và phần còn lại của System Prompt) ...

---
[PHẦN QUAN TRỌNG] XỬ LÝ THÔNG TIN TRA CỨU (RAG)
---
Khi nhận được thông tin trong một tin nhắn hệ thống bắt đầu bằng "--- BẮT ĐẦU DỮ LIỆU TRA CỨU TỪ 'SỔ TAY' (RAG) ---", bạn **PHẢI** tuân thủ các quy tắc sau:

1.  **ƯU TIÊN TUYỆT ĐỐI:** Dữ liệu này là nguồn "chân lý" (ground truth) từ Sổ tay Tin học. Bạn **PHẢI** ưu tiên sử dụng thông tin này để trả lời câu hỏi của người dùng.
2.  **TRÍCH DẪN (NẾU CÓ THỂ):** Nếu câu trả lời của bạn dựa trực tiếp vào "NGUỒN" được cung cấp, hãy cố gắng trích dẫn ngắn gọn (ví dụ: "Theo tài liệu,..." hoặc "Như trong Sổ tay có đề cập...").
3.  **TỔNG HỢP:** Nếu các NGUỒN cung cấp thông tin rời rạc, hãy tổng hợp chúng lại thành một câu trả lời mạch lạc.
4.  **KHÔNG BỊA ĐẶT:** Nếu thông tin tra cứu có vẻ không liên quan đến câu hỏi, hãy lịch sự thông báo rằng bạn không tìm thấy thông tin chính xác trong Sổ tay và trả lời dựa trên kiến thức chung của bạn.
"""

# --- BƯỚC 3: KHỞI TẠO CLIENT VÀ CHỌN MÔ HÌNH ---

# [SỬA LỖI] Dùng 'gemini-pro' (cơ bản) để đảm bảo API Key có quyền
# Thầy có thể thử lại 'gemini-1.5-pro-latest' sau khi app chạy
MODEL_NAME = 'gemini-pro' 
try:
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    gemini_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
        safety_settings=safety_settings
    )
    print("Khởi tạo model Gemini thành công.") # DEBUG
except Exception as e:
    st.error(f"Lỗi khi khởi tạo Model Gemini: {e}")
    st.stop()


# --- BƯỚC 4: CẤU HÌNH TRANG VÀ CSS ---
# (Giữ nguyên không thay đổi)
st.set_page_config(page_title="Chatbot Tin học 2018", page_icon="✨", layout="centered")
st.markdown("""<style> ... (Giữ nguyên CSS) ... </style>""", unsafe_allow_html=True)


# --- BƯỚC 4.5: THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.title("🤖 Chatbot KTC")
    st.markdown("---")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
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


# --- BƯỚC 4.6: CÁC HÀM RAG ---

@st.cache_data(ttl=3600) 
def load_and_process_pdfs(pdf_folder="data_pdf"):
    # Hàm này sẽ được gọi bên trong BƯỚC 5 (có bẫy lỗi)
    print(f"--- BẮT ĐẦU HÀM load_and_process_pdfs ---") # DEBUG
    print(f"Bắt đầu quét thư mục: {pdf_folder}")
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print("CẢNH BÁO: Không tìm thấy file PDF nào.")
        return None # Trả về None nếu không có file

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
            # Báo lỗi nếu 1 file cụ thể bị hỏng
            st.error(f"Lỗi khi đọc file {pdf_path}: {e}. Vui lòng kiểm tra file này trên GitHub.")
            print(f"Lỗi khi đọc file {pdf_path}: {e}") # DEBUG
            # Vẫn tiếp tục xử lý các file khác
            continue 

    if not chunks:
        print("Không thể trích xuất nội dung từ các file PDF.")
        return None # Trả về None nếu không có nội dung

    print(f"Đã trích xuất {len(chunks)} trang PDF. Bắt đầu vector hóa...")
    
    try:
        vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(chunks)
        print("Vector hóa hoàn tất.")
        # Trả về 3 đối tượng
        return (chunks, tfidf_matrix, vectorizer)
    
    except ValueError as e:
        if "empty vocabulary" in str(e):
            st.error(f"Lỗi RAG: Các file PDF có thể không chứa văn bản (chỉ chứa ảnh).")
        else:
            st.error(f"Lỗi Vectorizer: {e}")
        return None # Trả về None nếu lỗi
    

def find_relevant_knowledge(query, knowledge_data, num_chunks=3):
    # Lấy dữ liệu từ knowledge_data
    chunks, tfidf_matrix, vectorizer = knowledge_data
    
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

def convert_history_for_gemini(messages):
    gemini_history = []
    for msg in messages:
        role = 'model' if msg['role'] == 'assistant' else 'user'
        gemini_history.append({'role': role, 'parts': [msg['content']]})
    return gemini_history

# --- [SỬA LỖI] BƯỚC 5: KHỞI TẠO RAG VÀ BẪY LỖI ---
# Chúng ta sẽ bẫy lỗi ở đây để nó không "chết" âm thầm

if "knowledge_data" not in st.session_state:
    try:
        print("--- BƯỚC 5: BẮT ĐẦU TẢI RAG ---") # DEBUG
        with st.spinner("👩‍🏫 Em đang đọc 'Sổ tay Tin học' (PDF)..."):
            # Gọi hàm load PDF
            knowledge_result = load_and_process_pdfs()
            
            # Kiểm tra xem hàm có trả về dữ liệu không
            if knowledge_result is None:
                st.error("Lỗi: Không thể tải hoặc xử lý các file PDF. RAG sẽ bị tắt.")
                # Gán dữ liệu rỗng để app không bị crash
                st.session_state.knowledge_data = ([], None, None) 
            else:
                # Gán dữ liệu thành công
                st.session_state.knowledge_data = knowledge_result
                print("--- BƯỚC 5: TẢI RAG THÀNH CÔNG ---") # DEBUG
                
    except Exception as e:
        # Bắt mọi lỗi xảy ra trong quá trình load RAG
        print(f"--- LỖI NGHIÊM TRỌNG Ở BƯỚC 5 ---: {e}") # DEBUG
        st.error(f"Lỗi nghiêm trọng khi tải RAG: {e}")
        st.session_state.knowledge_data = ([], None, None) # Gán rỗng

# --- BƯỚC 6: HIỂN THỊ LỊCH SỬ CHAT ---
# (Giữ nguyên)
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        st.error(f"Lỗi: Không tìm thấy file logo tên là '{logo_path}'.")
        # Không st.stop() để app vẫn chạy
with col2:
    st.title("KTC. Chatbot hỗ trợ môn Tin Học")

if not st.session_state.messages:
    st.markdown(f"<div class='welcome-message'>Xin chào! Thầy/em cần hỗ trợ gì về môn Tin học (Chương trình 2018)?</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1_btn, col2_btn = st.columns(2)
    # ... (Giữ nguyên các nút bấm) ...
    with col1_btn:
        st.button("Giải thích về 'biến' trong lập trình?", on_click=set_prompt_from_suggestion, args=("Giải thích về 'biến' trong lập trình?",), use_container_width=True)
        st.button("Trình bày về an toàn thông tin?", on_click=set_prompt_from_suggestion, args=("Trình bày về an toàn thông tin?",), use_container_width=True)
    with col2_btn:
        st.button("Sự khác nhau giữa RAM và ROM?", on_click=set_prompt_from_suggestion, args=("Sự khác nhau giữa RAM và ROM?",), use_container_width=True)
        st.button("Các bước chèn ảnh vào word", on_click=set_prompt_from_suggestion, args=("Các bước chèn ảnh vào word?",), use_container_width=True)


# --- BƯỚC 8: XỬ LÝ INPUT (ĐÃ SỬA LỖI TREO) --- 

prompt_from_input = st.chat_input("Mời thầy hoặc các em đặt câu hỏi về Tin học...")
prompt_from_button = st.session_state.pop("prompt_from_button", None)
prompt = prompt_from_button or prompt_from_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant", avatar="✨"):
            placeholder = st.empty()
            
            with placeholder.status("👩‍🏫 Chatbook đang suy nghĩ..."):
                print("--- BƯỚC 8: BẮT ĐẦU XỬ LÝ PROMPT ---") # DEBUG
                
                # --- PHẦN RAG ---
                # Kiểm tra xem RAG có dữ liệu không
                if st.session_state.knowledge_data and st.session_state.knowledge_data[0]:
                    print("Đang tìm kiến thức liên quan...") # DEBUG
                    retrieved_context = find_relevant_knowledge(prompt, st.session_state.knowledge_data, num_chunks=3)
                else:
                    retrieved_context = []
                
                print("Đang chuyển đổi lịch sử chat...") # DEBUG
                messages_for_api = convert_history_for_gemini(st.session_state.messages)
                
                if retrieved_context:
                    print(f"Đã tìm thấy {len(retrieved_context)} mẩu kiến thức RAG.") # DEBUG
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
                    print("Không tìm thấy kiến thức RAG liên quan.") # DEBUG

                # --- [SỬA LỖI TREO] ---
                print("ĐANG GỌI API GEMINI...") # DEBUG
                response = gemini_model.generate_content(
                    messages_for_api # Gửi toàn bộ
                )
                print("ĐÃ NHẬN PHẢN HỒI TỪ GEMINI.") # DEBUG
                
                # [SỬA LỖI] Gemini có thể chặn và không trả về 'text'
                if not response.parts:
                    # Kiểm tra xem có bị chặn không
                    if response.candidates and response.candidates[0].finish_reason == "SAFETY":
                        bot_response_text = "Xin lỗi, câu trả lời của tôi đã bị chặn vì lý do an toàn. Thầy/em vui lòng hỏi khác đi."
                    else:
                        bot_response_text = "Xin lỗi, tôi không thể tạo câu trả lời cho câu hỏi này."
                else:
                    bot_response_text = response.text
                
                # --- [KẾT THÚC SỬA LỖI] ---

            placeholder.markdown(bot_response_text)

    except Exception as e:
        with st.chat_message("assistant", avatar="✨"):
            st.error(f"Xin lỗi, đã xảy ra lỗi khi kết nối Gemini: {e}")
            print(f"--- LỖI XẢY RA Ở BƯỚC 8 ---: {e}") # DEBUG
        bot_response_text = ""

    # Thêm câu trả lời của bot vào lịch sử
    if bot_response_text:
        st.session_state.messages.append({"role": "assistant", "content": bot_response_text})

    # Rerun nếu bấm nút
    if prompt_from_button:
        st.rerun()