import librosa # type: ignore
import numpy as np
import sounddevice as sd # type: ignore
from scipy.io.wavfile import write # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Load pre-trained sentiment analysis model
model = load_model("sentiment_model.h5")  # Replace with your trained model path

# Constants
SAMPLE_RATE = 22050  # Sample rate for audio recording
DURATION = 5  # Duration of audio recording in seconds
THRESHOLD = 0.7  # Threshold for detecting distress/offensive content

def record_audio(duration, sample_rate):
    """
    Record audio from the microphone.
    """
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return audio.flatten()

def extract_features(audio, sample_rate):
    """
    Extract audio features using Librosa.
    """
    # Extract MFCC (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    # Extract Chroma STFT
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_scaled = np.mean(chroma.T, axis=0)

    # Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_scaled = np.mean(mel.T, axis=0)

    # Combine all features
    features = np.hstack((mfccs_scaled, chroma_scaled, mel_scaled))
    return features

def predict_sentiment(features):
    """
    Predict sentiment using the pre-trained model.
    """
    # Reshape features for model input
    features = np.expand_dims(features, axis=0)
    features = StandardScaler().fit_transform(features)  # Normalize features

    # Predict sentiment
    prediction = model.predict(features)
    return prediction[0][0]  # Return the probability of distress/offensive content

def trigger_sos():
    """
    Trigger an SOS alert.
    """
    print("Distress detected! Triggering SOS alert...")
    # Add code to notify emergency contacts or nearby users here

def main():
    # Step 1: Record audio
    audio = record_audio(DURATION, SAMPLE_RATE)

    # Step 2: Extract features
    features = extract_features(audio, SAMPLE_RATE)

    # Step 3: Predict sentiment
    sentiment_score = predict_sentiment(features)
    print(f"Sentiment Score: {sentiment_score}")

    # Step 4: Check for distress/offensive content
    if sentiment_score > THRESHOLD:
        trigger_sos()
    else:
        print("No distress detected. You are safe.")

if __name__ == "__main__":
    main()