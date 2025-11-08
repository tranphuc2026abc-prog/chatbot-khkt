# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: pip install google-generativeai streamlit

import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import time

# --- [ĐÃ XÓA] Toàn bộ thư viện RAG (pypdf, sklearn, numpy) ---


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
(RAG HIỆN ĐANG TẮT)
"""

# --- BƯỚC 3: KHỞI TẠO CLIENT VÀ CHỌN MÔ HÌNH ---
MODEL_NAME = 'gemini-pro' # Dùng model cơ bản nhất để test
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
    print("--- KHỞI TẠO MODEL GEMINI THÀNH CÔNG ---") # DEBUG
except Exception as e:
    st.error(f"Lỗi khi khởi tạo Model Gemini: {e}")
    st.stop()


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


# --- BƯỚC 4.6: CÁC HÀM RAG (ĐÃ XÓA) ---

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

# --- BƯỚC 5: KHỞI TẠO RAG (ĐÃ TẮT) ---
print("--- BƯỚC 5: RAG ĐÃ BỊ TẮT ---") # DEBUG

# --- BƯỚC 6: HIỂN THỊ LỊCH SỬ CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        st.error(f"Lỗi: Không tìm thấy file logo tên là '{logo_path}'.")
with col2:
    st.title("KTC. Chatbot hỗ trợ môn Tin Học")

# Thêm lại hàm bị thiếu
def set_prompt_from_suggestion(text):
    st.session_state.prompt_from_button = text

if not st.session_state.messages:
    st.markdown(f"<div class='welcome-message'>Xin chào! Thầy/em cần hỗ trợ gì về môn Tin học (Chương trình 2018)?</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1_btn, col2_btn = st.columns(2)
    with col1_btn:
        st.button(
            "Giải thích về 'biến' trong lập trình?",
            on_click=set_prompt_from_suggestion, 
            args=("Giải thích về 'biến' trong lập trình?",), 
            use_container_width=True
        )
        st.button(
            "Trình bày về an toàn thông tin?",
            on_click=set_prompt_from_suggestion, 
            args=("Trình bày về an toàn thông tin?",), 
            use_container_width=True
        )
    with col2_btn:
        st.button(
            "Sự khác nhau giữa RAM và ROM?",
            on_click=set_prompt_from_suggestion, 
            args=("Sự khác nhau giữa RAM và ROM?",), 
            use_container_width=True
        )
        st.button(
            "Các bước chèn ảnh vào word",
            on_click=set_prompt_from_suggestion, 
            args=("Các bước chèn ảnh vào word?",), 
            use_container_width=True
        )


# --- BƯỚC 8: XỬ LÝ INPUT (KHÔNG CÓ RAG) --- 

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
                print("--- BƯỚC 8: BẮT ĐẦU XỬ LÝ PROMPT (KHÔNG RAG) ---") # DEBUG
                
                messages_for_api = convert_history_for_gemini(st.session_state.messages)
                
                print("ĐANG GỌI API GEMINI...") # DEBUG
                response = gemini_model.generate_content(
                    messages_for_api # Gửi toàn bộ
                )
                print("ĐÃ NHẬN PHẢN HỒI TỪ GEMINI.") # DEBUG
                
                if not response.parts:
                    if response.candidates and response.candidates[0].finish_reason == "SAFETY":
                        bot_response_text = "Xin lỗi, câu trả lời của tôi đã bị chặn vì lý do an toàn. Thầy/em vui lòng hỏi khác đi."
                    else:
                        bot_response_text = "Xin lỗi, tôi không thể tạo câu trả lời cho câu hỏi này."
                else:
                    bot_response_text = response.text

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