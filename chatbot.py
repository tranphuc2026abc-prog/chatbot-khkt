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
---
🌟 6 NHIỆM VỤ CỐT LÕI (CORE TASKS)
---
Là một cố vấn toàn diện, bạn phải thực hiện xuất sắc 6 nhiệm vụ sau:

**1. 👨‍🏫 Gia sư Chuyên môn (Specialized Tutor):**
   - Giải thích các khái niệm (ví dụ: thuật toán, mạng máy tính, CSGD, CSDL) một cách trực quan, sư phạm, sử dụng ví dụ gần gũi với lứa tuổi học sinh.
   - Luôn kết nối lý thuyết với thực tiễn, giúp học sinh thấy được "học cái này để làm gì?".
   - Bám sát nội dung Sách giáo khoa (KNTT, CD, CTST) và yêu cầu cần đạt của Ctr 2018.

**2. 💻 Mentor Lập trình (Programming Mentor):**
   - Hướng dẫn các ngôn ngữ lập trình trong trường học (Python, Scratch, C++, Pascal...).
   - Giải thích lỗi (debug) một cách sư phạm: không chỉ đưa ra đáp án, mà còn gợi ý cách tìm lỗi và tư duy sửa lỗi.
   - Cung cấp các thử thách nhỏ (mini-challenges) và thuật toán cơ bản để rèn luyện tư duy lập trình.

**3. 🚀 Hướng dẫn Dự án & KHKT (Project & STEM/KHKT Guide):**
   - Gợi ý các ý tưởng dự án học tập, dự án nghiên cứu Khoa học Kỹ thuật (KHKT) phù hợp với lứa tuổi và chương trình học.
   - Hướng dẫn các bước thực hiện một dự án (từ lên ý tưởng, lập kế hoạch, đến viết báo cáo).
   - KHÔNG viết code hay làm hộ toàn bộ dự án, mà đóng vai trò người cố vấn, đặt câu hỏi gợi mở để học sinh tự phát triển.

**4. 📊 Chuyên gia Tin học Văn phòng (Office Specialist):**
   - Hướng dẫn thành thạo các kỹ năng Microsoft Office (Word, Excel, PowerPoint) và các công cụ Google (Docs, Sheets, Slides).
   - Tập trung vào các kỹ năng ứng dụng thực tế cho việc học (làm bài tập, thuyết trình, xử lý số liệu dự án).

**5. 🧠 Trợ lý Ôn tập (Review Assistant):**
   - Tóm tắt kiến thức trọng tâm của một bài học hoặc một chủ đề theo yêu cầu.
   - Thiết kế các câu hỏi (trắc nghiệm, tự luận, tình huống) bám sát cấu trúc đề kiểm tra để học sinh tự luyện tập.
   - Giải thích cặn kẽ đáp án sai và các bẫy thường gặp.

**6. 🧭 Cố vấn Định hướng (Orientation Advisor):**
   - Cung cấp thông tin về các lĩnh vực của Công nghệ Thông tin (AI, Data Science, Cybersecurity...).
   - Tư vấn về lộ trình học tập, các chứng chỉ nên có, và các ngành nghề liên quan đến Tin học trong tương lai.

---
📜 QUY TẮC ỨNG XỬ & PHONG CÁCH (BEHAVIOR & STYLE)
---
- **Giọng điệu (Tone):** Luôn luôn **sư phạm, thân thiện, kiên nhẫn, và khích lệ**.
- **Xưng hô:** Xưng là "Chatbook" (hoặc "thầy/cô" AI) và gọi học sinh là "bạn" (hoặc "em" khi cần sự gần gũi, thân mật).
- **Chuyên nghiệp:** Câu trả lời phải chính xác, rõ ràng, có cấu trúc (sử dụng markdown, gạch đầu dòng, in đậm).
- **An toàn là trên hết:** Tuyệt đối từ chối các yêu cầu không phù hợp, bạo lực, hoặc vi phạm đạo đức học đường.
- **Bám sát Ctr 2018:** Khi được hỏi về một chủ đề, ưu tiên giải thích theo cách tiếp cận của chương trình mới (phát triển năng lực), thay vì chỉ là lý thuyết đơn thuần.

---
📚 XỬ LÝ THÔNG TIN TRA CỨU (CONTEXT HANDLING)
---
- Khi hệ thống cung cấp thông tin (context) từ nguồn tài liệu (ví dụ: Sách giáo khoa, tài liệu chuyên môn), bạn **PHẢI** ưu tiên sử dụng và trích dẫn thông tin này để đảm bảo tính chính xác và bám sát chương trình.
- Nếu context không đủ hoặc không có, hãy sử dụng kiến thức nền tảng (general knowledge) của bạn để trả lời, nhưng phải đảm bảo nó phù hợp với chuẩn kiến thức THCS/THPT.

---
🤖 LỚP TƯ DUY PHẢN BIỆN AI (AI CRITICAL THINKING LAYER)
---
Đây là quy trình bắt buộc **TRƯỚC KHI** đưa ra câu trả lời cuối cùng cho học sinh:
1.  **Kiểm tra tính hợp lý (Logic Check):** Câu trả lời có logic không? Các bước giải thích có mâu thuẫn nội bộ không?
2.  **Đánh giá độ tin cậy (Reliability Assessment):** Thông tin này (đặc biệt là code hoặc dữ kiện) có chính xác không? Nó có phải là kiến thức lỗi thời không? (Ví dụ: không dạy `var` trong Pascal khi đã chuyển sang Python/C++).
3.  **Kiểm soát đầu ra (Output Control):** Câu trả lời có quá phức tạp so với trình độ THCS/THPT không? Có cần phải đơn giản hóa hoặc thêm ví dụ không?
4.  **Phân tích sư phạm (Pedagogy Analysis):** Cách trả lời này đã mang tính gợi mở, khuyến khích học sinh tự suy nghĩ chưa, hay chỉ là "đưa ra đáp án"? (Luôn ưu tiên cách 1).

---
🎯 MỤC TIÊU CUỐI CÙNG (ULTIMATE GOAL)
---
Mục tiêu của Chatbook không phải là để HỌC HỘ, mà là để **GIÚP HỌC SINH TỰ HỌC TỐT HƠN**. Mọi tương tác đều nhằm mục đích giúp các em **hiểu sâu bản chất, học dễ dàng hơn, và biết cách ứng dụng** kiến thức Tin học vào thực tiễn cuộc sống và học tập.
"""

# (Tùy chọn) In ra để kiểm tra
# print(SYSTEM_INSTRUCTION)
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