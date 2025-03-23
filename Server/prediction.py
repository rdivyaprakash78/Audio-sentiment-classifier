import librosa
import io
import numpy as np
import torch
from load_pretrained_speech_models import load_all_models
import torch.nn.functional as F


def preprocess_audio(file_path):

    """This function preprocess the input audio in a format suitable for the model."""

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    TARGET_SAMPLE_RATE = 16000
    MAX_LENGTH = 48000
    
    waveform, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
    resampled_waveform = librosa.resample(np.array(waveform), orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

    num_samples = resampled_waveform.shape[0]
    waveform = torch.tensor(resampled_waveform).unsqueeze(0)

    if num_samples < MAX_LENGTH:
        pad = torch.zeros((1, MAX_LENGTH - num_samples))
        waveform = torch.cat((waveform, pad), dim=1)
    elif num_samples > MAX_LENGTH:
        waveform = waveform[:, :MAX_LENGTH]

    return waveform, TARGET_SAMPLE_RATE

def scale_embeddings(x):
    mean = x.mean(dim=1, keepdim=True) 
    std = x.std(dim=1, keepdim=True) 

    scaled_embeddings = (x - mean) / std

    return scaled_embeddings

def get_predictions(waveform):
    hubert_feature_extractor, hubert_model, classifier = load_all_models()
    hubert_features = hubert_feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")["input_values"]

    with torch.no_grad():
        outputs = hubert_model(hubert_features)

    embeddings = outputs.last_hidden_state
    mean_pooled_embeddings = embeddings.mean(dim = 1)
    scaled_mean_pooled_embeddings = scale_embeddings(mean_pooled_embeddings)

    with torch.no_grad():
        logits = classifier(scaled_mean_pooled_embeddings)

    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    max_probability = probabilities[0][predicted_class].item()
    confidence = f"{max_probability * 100:.2f}"

    return probabilities, predicted_class, confidence

