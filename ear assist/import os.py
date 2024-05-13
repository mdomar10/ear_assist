import os
import pyaudio
import vosk
import json

# Define the Vosk model directory
MODEL_DIR = "C:/Users/omarb/Downloads/vosk-model-small-en-us-0.15 (1)/vosk-model-small-en-us-0.15"

# Initialize the Vosk recognizer with the model
vosk_model = vosk.Model(MODEL_DIR)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Define audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Start audio stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")

# Continuously listen and transcribe speech
while True:
    data = stream.read(CHUNK)
    if len(data) == 0:
        break
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        print("Transcription:", result["text"])

# Stop and close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()