import streamlit as st
import os

# Try to import API client for backend mode
try:
    from frontend.api_client import send_chat_message as api_send_chat
    from frontend.api_client import get_api_key, check_backend_health
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False

# Import direct Groq client as fallback
client = None
_Groq = None
_sr = None
try:
    import speech_recognition as _sr
except ImportError:
    pass
try:
    from groq import Groq as _Groq
except ImportError:
    pass

def _init_groq():
    global client
    if client is None and _Groq is not None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            try:
                client = _Groq(api_key=api_key)
            except Exception:
                pass

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
        if _sr is None:
            st.error("Speech recognition not available. Please install SpeechRecognition package.")
        else:
            try:
                recognizer = _sr.Recognizer()
                with _sr.Microphone() as source:
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

                use_api = HAS_API_CLIENT and get_api_key() and check_backend_health()
                if use_api:
                    reply = api_send_chat(user_input, report, language)
                else:
                    _init_groq()
                    if client is None:
                        st.error("No Groq client available. Configure backend or GROQ_API_KEY.")
                        reply = "Sorry, I cannot process your request right now."
                    else:
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=1024
                        )
                        reply = response.choices[0].message.content

                st.write(reply)

        # Save assistant response
        st.session_state["messages"].append(
            {"role": "assistant", "content": reply}
        )