import socket
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

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('172.16.51.124', 65432)  # Replace <Raspberry Pi IP> with the actual IP address of your Raspberry Pi
client_socket.connect(server_address)

print("Listening...")

# Continuously listen and transcribe speech
while True:
    data = stream.read(CHUNK)
    if len(data) == 0:
        break
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        transcription = result["text"]
        print("Transcription:", transcription)
        
        # Send the transcription to the Raspberry Pi for display
        client_socket.sendall(transcription.encode('utf-8'))

# Stop and close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()
