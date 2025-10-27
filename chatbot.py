# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: pip install groq streamlit
# (Lưu ý: Pypdf không còn cần thiết nếu thầy tắt RAG, nhưng để đó cũng không sao)
import streamlit as st
from groq import Groq
import os
import glob
import time
#
# *** LƯU Ý: Thầy có thể comment out (thêm #) dòng import pypdf ở đầu file nếu có
# vì chúng ta không còn dùng đến nó.
# Ví dụ: # from pypdf import PdfReader
#

# --- BƯỚC 1: LẤY API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Lỗi: Không tìm thấy GROQ_API_KEY. Vui lòng thêm vào Secrets trên Streamlit Cloud.")
    st.stop()
    
# --- BƯỚC 2: THIẾT LẬP VAI TRÒ (SYSTEM_INSTRUCTION) ---
SYSTEM_INSTRUCTION = (
    "Bạn là 'Chatbook' - một Cố vấn Học tập Tin học AI toàn diện, với kiến thức cốt lõi của một "
    "giáo viên Tin học dạy giỏi cấp quốc gia, nắm vững chương trình GDPT 2018. "
    "Nhiệm vụ của bạn là hỗ trợ học sinh THCS/THPT một cách toàn diện. "
    
    # ... (Toàn bộ 6 nhiệm vụ của thầy vẫn giữ nguyên ở đây) ...
    "1. **Gia sư Chuyên môn (Lý thuyết):** ... " # (Giữ nguyên)
    "2. **Mentor Lập trình (Thực hành Code):** ... " # (Giữ nguyên)
    "3. **Người hướng dẫn Dự án (Sáng tạo):** ... " # (Giữ nguyên)
    "4. **Chuyên gia Tin học Văn phòng (Ứng dụng):** ... " # (Giữ nguyên)
    "5. **Trợ lý Ôn tập (Củng cố):** ... " # (Giữ nguyên)
    "6. **Cố vấn Định hướng (Tương lai):** ... " # (Giữ nguyên)
    
    "Khi tương tác, hãy luôn giữ giọng văn chuyên nghiệp nhưng thân thiện, "
    "tập trung 100% vào nội dung chương trình 2018 và các ứng dụng thực tế của nó."
    "Nếu câu hỏi KHÔNG liên quan đến Tin học, lập trình, hoặc Office, hãy trả lời rằng "
    "chuyên môn chính của bạn là Tin học."
    
    # --- PHẦN SỬA LỖI QUAN TRỌNG NẰM Ở ĐÂY ---
    # (Phần này không còn tác dụng vì RAG đã bị tắt, nhưng giữ lại cũng không sao)
    "TRỪ KHI: Nếu bạn được cung cấp 'Thông tin tra cứu' (context) từ tài liệu: "
    "1. Đầu tiên, hãy **KIỂM TRA** xem thông tin tra cứu đó có **LIÊN QUAN TRỰC TIẾP** đến câu hỏi của học sinh không."
    "2. **Nếu CÓ liên quan:** Hãy dựa vào thông tin đó để trả lời."
    "3. **Nếu KHÔNG liên quan:** (Ví dụ: học sinh hỏi về Excel nhưng thông tin tra cứu lại nói về PowerPoint) "
    "Hãy **BỎ QUA** thông tin tra cứu đó và trả lời câu hỏi bằng kiến thức chung của bạn mà **KHÔNG ĐƯỢC PHÊ PHÁN** hay đề cập đến sự không liên quan của tài liệu."
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
        st.session_state.pop("knowledge_chunks", None) # Xóa cache kiến thức
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
# (Các hàm này vẫn được định nghĩa, nhưng sẽ không được gọi nữa)

@st.cache_data(ttl=3600) 
def load_and_chunk_pdfs():
    # Sẽ không chạy vì chúng ta đã vô hiệu hóa ở BƯỚC 5
    print("HÀM 'load_and_chunk_pdfs' SẼ KHÔNG ĐƯỢC GỌI.")
    return []

def find_relevant_knowledge(query, knowledge_chunks, num_chunks=3):
    # Sẽ không chạy vì chúng ta đã vô hiệu hóa ở BƯỚC 8
    print("HÀM 'find_relevant_knowledge' SẼ KHÔNG ĐƯỢC GỌI.")
    return None


# --- BƯỚC 5: KHỞI TẠO LỊCH SỬ CHAT VÀ "SỔ TAY" PDF --- # <--- ĐÃ VÔ HIỆU HÓA RAG
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- ĐÃ VÔ HIỆU HÓA RAG THEO YÊU CẦU ---
# Tải và xử lý PDF khi app khởi động
if "knowledge_chunks" not in st.session_state:
    # Chúng ta không gọi hàm load_and_chunk_pdfs() nữa
    # Thay vào đó, chỉ cần khởi tạo một danh sách rỗng
    st.session_state.knowledge_chunks = []
    print("RAG (Đọc PDF) đã bị tắt. Bỏ qua việc tải file.")
# --- KẾT THÚC VÔ HIỆU HÓA ---


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
            "Tóm tắt bài 6 Tin 12 (KNTT)?",
            on_click=set_prompt_from_suggestion, args=("Tóm tắt bài 6 Tin 12 (KNTT)?",),
            use_container_width=True
        )


# --- BƯỚC 8: XỬ LÝ INPUT (ĐÃ VÔ HIỆU HÓA RAG PDF) --- # <--- ĐÃ CẬP NHẬT
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

            # --- PHẦN RAG MỚI ĐÃ BỊ VÔ HIỆU HÓA --- #
            
            # 2.1. (BỎ QUA) Tìm kiếm trong kho kiến thức PDF
            # retrieved_context = find_relevant_knowledge(prompt, st.session_state.knowledge_chunks)
            
            # 2.2. Chuẩn bị list tin nhắn gửi cho AI (Không dùng RAG)
            messages_to_send = [
                {"role": "system", "content": SYSTEM_INSTRUCTION}
            ]
            
            # 2.3. (BỎ QUA) logic 'if retrieved_context:'
            
            # Thay vào đó, chúng ta gửi toàn bộ lịch sử chat như bình thường
            print("RAG đã tắt. Trả lời bình thường dựa trên lịch sử chat.")
            messages_to_send.extend(st.session_state.messages)
            
            # --- KẾT THÚC PHẦN RAG BỊ VÔ HIỆU HÓA --- #

            # 2.4. Gọi API Groq
            stream = client.chat.completions.create(
                messages=messages_to_send, # Gửi lịch sử chat tiêu chuẩn
                model=MODEL_NAME,
                stream=True
            )
            
            # 2.5. Lặp qua từng "mẩu" (chunk) API trả về
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