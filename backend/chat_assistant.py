import streamlit as st
import google.generativeai as genai
import speech_recognition as sr

import os
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
genai.configure(api_key=api_key)

def chat_tab():

    st.header("💬 Chat Assistant")

    if "report_text" not in st.session_state:
        st.warning("⚠️ Upload and analyze a lab report first")
        return

    # -------- LANGUAGE --------
    language = st.selectbox("🌍 Choose Language", ["English", "Hindi", "Telugu"])

    # -------- CHAT HISTORY --------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # -------- DISPLAY CHAT --------
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -------- VOICE INPUT --------
    if st.button("🎤 Speak"):
        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening...")
                audio = recognizer.listen(source)

            user_input = recognizer.recognize_google(audio)
            st.success(f"You said: {user_input}")

        except Exception as e:
            st.error(f"Voice input failed: {str(e)}")
            user_input = ""
    else:
        user_input = st.chat_input("Ask something about your report...")

    # -------- PROCESS INPUT --------
    if user_input:

        # Show user message
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                model = genai.GenerativeModel("gemini-2.5-flash")

                report = st.session_state["report_text"]

                prompt = f"""
You are a friendly health assistant.

User question:
{user_input}

Lab report:
{report}

Instructions:
- Answer in {language}
- Use VERY simple language
- Max 2 lines only
- Make it conversational
- Avoid medical jargon
- If needed, explain in simple words

Example style:
"Your hemoglobin is a bit low. This means your body may feel tired easily."

Keep it short and human-like.
"""

                response = model.generate_content(prompt)
                reply = response.text

                st.write(reply)

        # Save assistant response
        st.session_state["messages"].append(
            {"role": "assistant", "content": reply}
        )