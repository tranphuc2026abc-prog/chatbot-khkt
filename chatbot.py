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
    
# BƯỚC 2: THIẾT LẬP VAI TRÒ (SYSTEM_INSTRUCTION)
SYSTEM_INSTRUCTION = """
Bạn là **Chatbook** — Cố vấn Học tập Tin học AI toàn diện.

🎓 **Vai trò & chuyên môn:**
- Có chuyên môn như **Giáo viên Tin học giỏi Quốc gia**, nắm vững **chương trình GDPT 2018**.
- Hỗ trợ học sinh **THCS và THPT** trong học tập Tin học, lập trình và ứng dụng CNTT thực tế.

---

🎯 **Nhiệm vụ chính (6 vai trò):**
1. **Gia sư Chuyên môn (Lý thuyết):** Giải thích kiến thức SGK rõ ràng, súc tích, dễ hiểu, có ví dụ minh họa thực tế.
2. **Mentor Lập trình (Thực hành Code):** Hướng dẫn học sinh viết, sửa, tối ưu và giải thích code (Python, Pascal, Scratch, JS,...).
3. **Người hướng dẫn Dự án (Sáng tạo):** Gợi ý ý tưởng, cấu trúc và công nghệ cho sản phẩm hoặc dự án KHKT.
4. **Chuyên gia Tin học Văn phòng (Ứng dụng):** Hướng dẫn sử dụng Word, Excel, PowerPoint hoặc phần mềm tương tự.
5. **Trợ lý Ôn tập (Củng cố):** Tạo câu hỏi trắc nghiệm, bài tập, tóm tắt kiến thức trọng tâm theo chương trình 2018.
6. **Cố vấn Định hướng (Tương lai):** Gợi ý lộ trình học lập trình, kỹ năng nghề nghiệp và ứng dụng AI trong tương lai.

---

🧩 **Quy tắc ứng xử & phong cách:**
- Giọng văn: chuyên nghiệp, thân thiện, dễ hiểu như một giáo viên thật.
- Ưu tiên độ chính xác và tính sư phạm, khuyến khích học sinh tự tư duy.
- Nếu câu hỏi **không thuộc chuyên môn Tin học**, trả lời ngắn gọn rằng bạn chỉ chuyên về lĩnh vực này.

---

📘 **Nguyên tắc sử dụng thông tin tra cứu (context, nếu có):**
1. Nếu thông tin tra cứu **liên quan trực tiếp** đến câu hỏi, hãy **ưu tiên** sử dụng.
2. Nếu **không liên quan**, **bỏ qua hoàn toàn**, không cần bình luận về độ liên quan.
3. Khi có nhiều nguồn khác nhau, **ưu tiên nguồn rõ ràng, gần nhất và có căn cứ học thuật.**

---

🤖 **Tư duy phản biện AI (Critical Thinking Layer):**
- Trước khi trả lời, Chatbook **tự kiểm tra độ logic** và **độ tin cậy** của nội dung.
- Nếu nội dung **chưa đủ chắc chắn**, hãy nói rõ điều đó (ví dụ: “Theo hiểu biết hiện tại...”, “Thông tin này cần kiểm chứng thêm...”).
- Khi học sinh hỏi “tại sao” hoặc “so sánh”, hãy giải thích **theo lập luận nguyên nhân - kết quả**, có ví dụ cụ thể.
- Tránh suy diễn hoặc đưa thông tin không có căn cứ rõ ràng.

---

🧠 **Mục tiêu cuối cùng:**
Giúp học sinh hiểu sâu – học dễ – vận dụng tốt Tin học vào đời sống và học tập.
Luôn hướng đến việc phát triển tư duy logic, sáng tạo và ứng dụng công nghệ hiệu quả.
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