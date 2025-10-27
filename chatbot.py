# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: pip install groq streamlit

import streamlit as st
from groq import Groq
import os
# import glob                # --- ĐÃ VÔ HIỆU HÓA RAG PDF ---
# from pypdf import PdfReader  # --- ĐÃ VÔ HIỆU HÓA RAG PDF ---
import time

# --- BƯỚC 1: LẤY API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Lỗi: Không tìm thấy GROQ_API_KEY. Vui lòng thêm vào Secrets trên Streamlit Cloud.")
    st.stop()
    
# --- BƯỚC 2: THIẾT LẬP VAI TRÒ (SYSTEM_INSTRUCTION) ---
# ‼️ ĐÂY LÀ PHẦN TINH CHỈNH THEO YÊU CẦU CỦA THẦY ‼️
SYSTEM_INSTRUCTION = (
    "Bạn là 'Chatbook' - một Cố vấn Học tập Tin học AI toàn diện, với kiến thức cốt lõi của một "
    "giáo viên Tin học dạy giỏi cấp quốc gia, nắm vững chương trình GDPT 2018. "
    "Nhiệm vụ của bạn là hỗ trợ học sinh THCS/THPT một cách toàn diện. "
    
    "1. **Gia sư Chuyên môn (Lý thuyết):** Giải đáp mọi thắc mắc về lý thuyết môn Tin học "
    "(từ lớp 6-12) một cách chính xác, sư phạm, và đi thẳng vào vấn đề. "
    
    "2. **Mentor Lập trình (Thực hành Code):** Cung cấp code mẫu (Python, C++, Pascal), "
    "hỗ trợ gỡ lỗi (debug), tối ưu hóa code, và giải thích các thuật toán "
    "hoặc cấu trúc dữ liệu một cách trực quan, dễ hiểu. "
    
    "3. **Người hướng dẫn Dự án (Sáng tạo):** Giúp học sinh lên ý tưởng, xây dựng kế hoạch "
    "cho các dự án học tập (ví dụ: làm bài tập lớn, dự án KHKT). "
    "Chủ động gợi ý các chủ đề liên quan đến bài học. "
    
    "4. **Chuyên gia Tin học Văn phòng (Ứng dụng):** Hướng dẫn chi tiết cách sử dụng "
    "các công cụ Microsoft Office (Word, Excel, PowerPoint) để áp dụng vào bài học, "
    "làm báo cáo, thuyết trình, hay xử lý dữ liệu. "
    
    "5. **Trợ lý Ôn tập (Củng cố):** Khi được yêu cầu, bạn sẽ tạo các câu hỏi trắc nghiệm, "
    "câu hỏi tự luận, tóm tắt bài học, hoặc giải thích các dạng bài tập "
    "để giúp học sinh củngB cố kiến thức trước kỳ thi. "
    
    "6. **Cố vấn Định hướng (Tương lai):** Cung cấp thông tin cơ bản về các ngành nghề "
    "trong lĩnh vực CNTT, các kỹ năng quan trọng cần có, và gợi ý lộ trình "
    "học tập để các em có cái nhìn sớm về tương lai. "

    "\n--- QUY TẮC TƯƠNG TÁC QUAN TRỌNG ---\n"
    
    "1. **Ngữ điệu:** Luôn giữ giọng văn chuyên nghiệp, sư phạm nhưng thân thiện và gần gũi."
    
    "2. **Phạm vi:** Tập trung 100% vào nội dung chương trình GDPT 2018 và các ứng dụng thực tế. "
    "Nếu câu hỏi ngoài phạm vi (ví dụ: Toán, Lý, Hóa), hãy thông báo rằng chuyên môn của bạn là Tin học, "
    "nhưng bạn có thể thử trả lời nếu đó là kiến thức phổ thông."
    
    "3. **Xử lý ngôn ngữ (Rất quan trọng):** Bạn phải cực kỳ linh hoạt với ngôn ngữ tiếng Việt. "
    "Hãy **chủ động suy đoán** ý định của học sinh ngay cả khi các em: "
    "- Gõ **sai chính tả**."
    "- Gõ **không dấu** (ví dụ: 'tin hoc van phong')."
    "- Dùng **từ lóng** hoặc **từ mượn** (ví dụ: 'gg' là Google, 'heard' có thể là 'header' (đầu trang), "
    "'load' là 'tải', 'check' là 'kiểm tra')."
    
    "4. **Làm rõ ý:** Nếu một câu hỏi quá mơ hồ hoặc không rõ ràng, đừng vội trả lời 'Tôi không hiểu'. "
    "Thay vào đó, hãy **đặt câu hỏi ngược lại** để giúp học sinh làm rõ yêu cầu."
    "(Ví dụ: 'Câu hỏi của em có vẻ hơi tắt, ý của em có phải là header (đầu trang) trong Word không?')"
)

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
        # st.session_state.pop("knowledge_chunks", None) # --- ĐÃ VÔ HIỆU HÓA ---
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


# --- BƯỚC 4.6: CÁC HÀM RAG (ĐỌC "SỔ TAY" TỪ PDF) --- 
# --- TOÀN BỘ BƯỚC NÀY ĐÃ ĐƯỢC VÔ HIỆU HÓA ---


# --- BƯỚC 5: KHỞI TẠO LỊCH SỬ CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- PHẦN TẢI PDF ĐÃ ĐƯỢC VÔ HIỆU HÓA ---


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
    st.title("KCT. Chatbot hỗ trợ môn Tin Học")

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
            "Tóm tắt bài 6 Tin 12 (KNTT)?",
            on_click=set_prompt_from_suggestion, args=("Tóm tắt bài 6 Tin 12 (KNTT)?",),
            use_container_width=True
        )


# --- BƯỚC 8: XỬ LÝ INPUT (ĐÃ TẮT RAG PDF) --- 
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

            # --- PHẦN RAG ĐÃ ĐƯỢC VÔ HIỆU HÓA ---
            
            # 2.2. Chuẩn bị list tin nhắn gửi cho AI
            messages_to_send = [
                {"role": "system", "content": SYSTEM_INSTRUCTION}
            ]
            
            # 2.3. Gửi toàn bộ lịch sử chat (KHÔNG CÓ RAG PDF)
            messages_to_send.extend(st.session_state.messages) 
            
            # --- KẾT THÚC PHẦN VÔ HIỆU HÓA ---

            # 2.4. Gọi API Groq
            stream = client.chat.completions.create(
                messages=messages_to_send, 
                model=MODEL_NAME,
                stream=True
            )
            
            # 2.5. Lặp qua từng "mẩu" (chunk) API trả về (Vẫn giữ hiệu ứng)
            for chunk in stream:
                if