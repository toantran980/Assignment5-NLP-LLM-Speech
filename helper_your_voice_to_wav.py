import sounddevice as sd
import soundfile as sf
import queue
import sys

# Output file
OUTPUT_FILE = "recording.wav"

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 1

# A thread-safe queue to store recorded audio blocks
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each block of audio."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def main():
    print("Recording... Press Ctrl+C to stop and save.")

    # Open audio input stream
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        callback=audio_callback):

        # Prepare WAV file for writing
        with sf.SoundFile(OUTPUT_FILE, mode='w',
                          samplerate=SAMPLE_RATE,
                          channels=CHANNELS) as file:

            try:
                while True:
                    file.write(q.get())
            except KeyboardInterrupt:
                print("\nRecording stopped.")
                print(f"Saved WAV file to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
