import speech_recognition as sr
import pyaudio

try:
    print(f"PyAudio version: {pyaudio.get_portaudio_version_text()}")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Microphone successfully initialized! Ready to listen.")
except Exception as e:
    print(f"MICROPHONE ERROR: {str(e)}")
