import time
import numpy as np
import Levenshtein as lev
import pyaudio
import vosk
import json  
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

# Define the Vosk model directory
MODEL_DIR =  "C:/Users/omarb/Downloads/vosk-model-small-en-us-0.15 (1)/vosk-model-small-en-us-0.15"

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

# Start accuracy checking and efficiency evaluation
start_time = time.time()
total_words = 0
total_errors = 0
elapsed_times = []
word_error_rates = []
transcriptions_per_seconds = []
latencies = []  # List to store latency values
transcribed_texts = []  # List to store transcribed texts

print("Listening...")

while time.time() - start_time < 60:  # Listen for 10 seconds for demonstration purposes
    data = stream.read(CHUNK)
    if len(data) == 0:
        break

    # Record start time when speech is detected
    speech_start_time = time.time()

    # Perform speech recognition
    if recognizer.AcceptWaveform(data):
        result = recognizer.Result()
        result_dict = json.loads(result)
        transcribed_text = result_dict["text"]
        
        # Calculate latency
        latency = time.time() - speech_start_time
        latencies.append(latency)
        
        # Print transcribed text, WER, and latency
        print(f"Transcribed Text: {transcribed_text}")
        reference_text = "The world is so beautiful and communication skills and knowledge of the team about this code"  # Provide the actual spoken words
        wer = lev.distance(reference_text.split(), transcribed_text.split()) / len(reference_text.split()) * 100
        print(f"Word Error Rate (WER): {wer:.2f}%")
        print(f"Latency: {latency:.2f} seconds\n")
        
        # Append transcribed text, WER, and latency to lists
        transcribed_texts.append(transcribed_text)
        word_error_rates.append(wer)
        
        # Manual verification
        total_errors += lev.distance(reference_text.split(), transcribed_text.split())
        total_words += len(reference_text.split())
        
        # Calculate transcriptions per second
        elapsed_time = time.time() - start_time
        transcriptions_per_second = total_words / elapsed_time
        transcriptions_per_seconds.append(transcriptions_per_second)

        # Append elapsed time for plotting
        elapsed_times.append(elapsed_time)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(elapsed_times, word_error_rates, label='Word Error Rate (WER)')
plt.plot(elapsed_times, transcriptions_per_seconds, label='Transcriptions per second')
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.title('Accuracy and Efficiency Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Print average WER and average latency
if transcribed_texts:
    avg_wer = sum(word_error_rates) / len(word_error_rates)
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Word Error Rate (WER): {avg_wer:.2f}%")
    print(f"Average Latency: {avg_latency:.2f} seconds")

# Stop and close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()