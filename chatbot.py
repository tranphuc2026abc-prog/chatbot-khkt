# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: pip install groq streamlit pypdf
import streamlit as st
from groq import Groq
import os
import glob
from pypdf import PdfReader
import time # <--- THÊM DÒNG NÀY
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


# --- BƯỚC 4.6: CÁC HÀM RAG (ĐỌC "SỔ TAY" TỪ PDF) --- # <--- ĐÃ VIẾT LẠI HOÀN TOÀN

# HÀM 1: Tải, bóc tách và "cắt mẩu" (chunk) các file PDF
# @st.cache_data sẽ lưu lại kết quả, chỉ chạy 1 lần (cho đến khi cache bị xóa)
@st.cache_data(ttl=3600) # Cache trong 1 giờ
def load_and_chunk_pdfs():
    knowledge_chunks = []
    
    # 1. Tìm tất cả file .pdf trong thư mục
    pdf_files = glob.glob("*.pdf") 
    
    if not pdf_files:
        print("Cảnh báo: Không tìm thấy file PDF nào.")
        return []

    print(f"Tìm thấy {len(pdf_files)} file PDF: {pdf_files}")
    
    # 2. Lặp qua từng file PDF
    for pdf_path in pdf_files:
        print(f"Đang xử lý file: {pdf_path}")
        try:
            reader = PdfReader(pdf_path)
            # 3. Lặp qua từng trang trong file
            for page in reader.pages:
                text = page.extract_text() # Bóc tách chữ
                if text:
                    # 4. "Cắt mẩu" (Chunking) - Phương pháp đơn giản:
                    # Tách các đoạn văn dựa trên dấu xuống dòng kép (\n\n)
                    chunks = text.split('\n\n')
                    
                    # 5. Thêm các mẩu (đã làm sạch) vào kho kiến thức
                    for chunk in chunks:
                        cleaned_chunk = chunk.strip()
                        if cleaned_chunk:
                            # Thêm nguồn để biết mẩu này từ đâu (tùy chọn)
                            knowledge_chunks.append(f"[Nguồn: {pdf_path}] {cleaned_chunk}") 
                            
        except Exception as e:
            print(f"LỖI khi đọc file {pdf_path}: {e}")
            
    print(f"Đã bóc tách và tạo được {len(knowledge_chunks)} mẩu kiến thức (chunks).")
    return knowledge_chunks

# HÀM 2: Tìm kiến thức (Retrieve)
def find_relevant_knowledge(query, knowledge_chunks, num_chunks=3):
    query_lower = query.lower()
    
    # 1. Tách từ khóa cơ bản từ câu hỏi
    # (Bỏ qua các từ chung như 'là', 'gì', 'của'...)
    common_words = {'là', 'gì', 'của', 'và', 'một', 'cách', 'để', 'trong', 'với'}
    query_keywords = set(query_lower.split()) - common_words
    
    relevant_chunks = []
    
    # 2. Tìm kiếm (Cách đơn giản: đếm số từ khóa xuất hiện)
    chunk_scores = []
    for i, chunk in enumerate(knowledge_chunks):
        chunk_lower = chunk.lower()
        score = 0
        for keyword in query_keywords:
            if keyword in chunk_lower:
                score += 1
        
        if score > 0:
            chunk_scores.append((score, i, chunk))
    
    # 3. Sắp xếp các mẩu theo điểm số (từ cao đến thấp)
    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Lấy N mẩu có điểm cao nhất
    top_chunks = [chunk for score, i, chunk in chunk_scores[:num_chunks]]
    
    if not top_chunks:
        return None # Không tìm thấy
        
    print(f"Đã tìm thấy {len(top_chunks)} mẩu liên quan.")
    return "\n---\n".join(top_chunks)


# --- BƯỚC 5: KHỞI TẠO LỊCH SỬ CHAT VÀ "SỔ TAY" PDF --- # <--- ĐÃ NÂNG CẤP
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tải và xử lý PDF khi app khởi động
if "knowledge_chunks" not in st.session_state:
    # Hiển thị thông báo chờ...
    with st.spinner("Đang tải và xử lý tài liệu (PDF)..."):
        st.session_state.knowledge_chunks = load_and_chunk_pdfs()
        
        if not st.session_state.knowledge_chunks:
            print("Cảnh báo: Không có kiến thức nào được tải từ file PDF.")
        else:
            print(f"Đã tải {len(st.session_state.knowledge_chunks)} mẩu kiến thức vào session.")


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


# --- BƯỚC 8: XỬ LÝ INPUT (ĐÃ NÂNG CẤP RAG PDF) --- # <--- ĐÃ CẬP NHẬT
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

            # --- PHẦN RAG MỚI BẮT ĐẦU TẠI ĐÂY --- #
            
            # 2.1. Tìm kiếm trong kho kiến thức PDF
            retrieved_context = find_relevant_knowledge(prompt, st.session_state.knowledge_chunks)
            
            # 2.2. Chuẩn bị list tin nhắn gửi cho AI
            messages_to_send = [
                {"role": "system", "content": SYSTEM_INSTRUCTION}
            ]
            
            # 2.3. Nếu "sổ tay" PDF có thông tin, thêm vào làm ngữ cảnh
            if retrieved_context:
                context_prompt = (
                    f"**Thông tin tra cứu (Hãy ưu tiên dùng thông tin này để trả lời):**\n"
                    f"{retrieved_context}\n\n"
                    f"**Câu hỏi của học sinh:**\n"
                    f"{prompt}"
                )
                messages_to_send.append({"role": "user", "content": context_prompt})
            else:
                # Nếu không tìm thấy, gửi lịch sử chat như bình thường
                print("Không tìm thấy mẩu kiến thức (chunk) nào liên quan. Trả lời bình thường.")
                messages_to_send.extend(st.session_state.messages)
            
            # --- KẾT THÚC PHẦN RAG --- #

            # 2.4. Gọi API Groq
            stream = client.chat.completions.create(
                messages=messages_to_send, 
                model=MODEL_NAME,
                stream=True
            )
            
            # 2.5. Lặp qua từng "mẩu" (chunk) API trả về
            for chunk in stream:
                if chunk.choices[0].delta.content is not None: 
                    bot_response_text += chunk.choices[0].delta.content
                    placeholder.markdown(bot_response_text + "▌")
                    time.sleep(0.005) # <--- THÊM DÒNG NÀY ĐỂ TẠO HIỆU ỨNG
            
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