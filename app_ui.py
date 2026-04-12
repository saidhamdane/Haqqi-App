import streamlit as st
import requests

# إعدادات الصفحة
st.set_page_config(page_title="ShadowAI Chat", page_icon="🤖")
st.title("🤖 ShadowAI: توأمك الرقمي")

# الرابط الخاص بالسيرفر (تأكد أن السيرفر الأساسي يعمل على 8080)
BASE_URL = "http://127.0.0.1:8080"

# شريط جانبي للهوية
with st.sidebar:
    st.header("إعدادات الهوية")
    employee_id = st.text_input("معرف الموظف (ID):", value="Said")
    st.info("تأكد من استخدام نفس الـ ID الذي استخدمته عند رفع الملفات.")

# تهيئة سجل الدردشة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال سؤال المستخدم
if prompt := st.chat_input("اسأل توأمك الرقمي..."):
    # إضافة سؤال المستخدم للواجهة
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # إرسال الطلب للسيرفر
    with st.chat_message("assistant"):
        try:
            params = {"employee_id": employee_id, "question": prompt}
            response = requests.get(f"{BASE_URL}/ask", params=params)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "لا يوجد رد")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("خطأ في الاتصال بالسيرفر. تأكد أن main.py يعمل.")
        except Exception as e:
            st.error(f"حدث خطأ: {e}")
