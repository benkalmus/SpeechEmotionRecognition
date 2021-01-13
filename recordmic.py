import os 
import pyaudio
import soundfile as sf
import sounddevice as sd

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = f"{os.getcwd()}\\data\\User\\file.wav"

def recordAudio(filePath, lengthSeconds):
    print("Recording started")
    recording = sd.rec(int(lengthSeconds * RATE), samplerate=RATE,
                         channels=CHANNELS, blocking=True, dtype='float32')
    
    print("Recording finished")
    #save to file
    sf.write(filePath, recording, RATE)
    return filePath
    
#recordAudio(WAVE_OUTPUT_FILENAME, RECORD_SECONDS)