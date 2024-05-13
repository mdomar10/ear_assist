import pyaudio
import vosk
import json
import numpy as np
from scipy.signal import stft, istft

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

# Continuously listen, apply noise cancellation, and transcribe speech
while True:
    data = stream.read(CHUNK)
    if len(data) == 0:
        break
    
    # Convert raw bytes into 16-bit integer samples
    data_np = np.frombuffer(data, dtype=np.int16)
    
    # Apply noise cancellation (spectral subtraction)
    f, t, Zxx = stft(data_np, fs=RATE, nperseg=256)
    Zxx_denoised = Zxx - np.mean(Zxx, axis=1)[:, np.newaxis]
    _, data_denoised = istft(Zxx_denoised, fs=RATE)

    # Convert denoised data back to raw bytes
    data_denoised_bytes = data_denoised.astype(np.int16).tobytes()

    if recognizer.AcceptWaveform(data_denoised_bytes):
        result = json.loads(recognizer.Result())
        print("Transcription:", result["text"])

# Stop and close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()
