import os
from transformers import  AutoFeatureExtractor, AutoModel
import torch.nn as nn
import torch.optim as optim
import torch

LOCAL_DIR = "models/hubert"
MODEL_NAME = "facebook/hubert-base-ls960"
INPUT_SIZE = 768
NUM_CLASSES = 8

class AudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioClassifier, self).__init__()
        self.fc0 = nn.Linear(input_size, 512)
        self.fc1 = nn.Linear(512, 256)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128) 
        self.fc3 = nn.Linear(128, num_classes) 
        self.dropout = nn.Dropout(0.3)  

    def forward(self, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

def load_hubert():

    """Loads HuBERT Base model from local storage or downloads it if missing."""
    
    if os.path.exists(LOCAL_DIR):
        print("Loading HuBERT model from local storage...")
        processor =  AutoFeatureExtractor.from_pretrained(LOCAL_DIR)
        model = AutoModel.from_pretrained(LOCAL_DIR)
    else:
        print("Downloading HuBERT model for the first time...")
        processor =  AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)

        # Save locally for future use
        processor.save_pretrained(LOCAL_DIR)
        model.save_pretrained(LOCAL_DIR)
        print("HuBERT model downloaded and saved locally.")

    # Set model to evaluation mode
    model.eval()

    return processor, model

def load_classifier():

    """Loads the sentiment classifier."""

    classifier = AudioClassifier(input_size = INPUT_SIZE, num_classes =  NUM_CLASSES)
    classifier.load_state_dict(torch.load("C:\ARAN\Working directory\Server\models\hubert_mean_pool_model_weights.pth",map_location=torch.device('cpu')))
    classifier.eval()
    return classifier

def load_all_models():
    processor, model = load_hubert()
    classifier = load_classifier()
    return processor, model, classifier
