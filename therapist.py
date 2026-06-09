# =========================================
# INSTALL
# =========================================
#!pip install gradio transformers timm librosa accelerate torchvision deepface

# =========================================
# IMPORTS
# =========================================
import torch
import torch.nn as nn
import numpy as np
import librosa
import timm
import gradio as gr
from PIL import Image
import torchvision.transforms as T

from transformers import (
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    pipeline
)

from deepface import DeepFace

# =========================================
# DEVICE
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================
# LOAD LLaVA
# =========================================
model_id = "llava-hf/llava-1.5-7b-hf"

llava_model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# =========================================
# TEXT EMOTION MODEL
# =========================================
text_emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# =========================================
# AUDIO MODEL
# =========================================
audio_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

audio_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).cpu()

# =========================================
# IMAGE ENCODER (for feature fusion)
# =========================================
image_encoder = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=0
).cpu()

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================================
# CROSS MODAL FUSION
# =========================================
class CrossModalFusion(nn.Module):

    def __init__(self, dim=768):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )

        self.fc = nn.Linear(dim, dim)

    def forward(self, text, image, audio):

        modalities = torch.cat(
            [text.unsqueeze(1), image.unsqueeze(1), audio.unsqueeze(1)],
            dim=1
        )

        fused,_ = self.attention(
            modalities,
            modalities,
            modalities
        )

        fused = fused.mean(dim=1)

        return self.fc(fused)

fusion_model = CrossModalFusion()

# =========================================
# FEATURE EXTRACTORS
# =========================================

def extract_text_emotion(text):

    result = text_emotion_model(text)[0][0]["label"]

    return result


def extract_audio_emotion(audio_path):

    if audio_path is None:
        return "Unknown"

    speech, sr = librosa.load(audio_path, sr=16000)

    energy = np.mean(librosa.feature.rms(y=speech))

    if energy > 0.1:
        return "Excited"
    elif energy > 0.05:
        return "Neutral"
    else:
        return "Sad"


def extract_face_emotion(image):

    if image is None:
        return "Unknown"

    try:
        result = DeepFace.analyze(
            np.array(image),
            actions=["emotion"],
            enforce_detection=False
        )

        return result[0]["dominant_emotion"]

    except:
        return "Unknown"


def extract_image_features(image):

    if image is None:
        return torch.zeros((1,768))

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = image_encoder(image)

    return features


def extract_audio_features(audio_path):

    if audio_path is None:
        return torch.zeros((1,768))

    speech, sr = librosa.load(audio_path, sr=16000)

    inputs = audio_processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = audio_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1)

# =========================================
# MULTIMODAL EMOTION REASONING
# =========================================

def combine_emotions(text_e, audio_e, face_e):

    emotions = [text_e, audio_e, face_e]

    if "sad" in emotions or "Sad" in emotions:
        return "Distressed"

    if "happy" in emotions or "joy" in emotions:
        return "Happy"

    if "angry" in emotions:
        return "Frustrated"

    return "Neutral"

# =========================================
# MAIN RESPONSE
# =========================================

def generate_response(image, text, audio):

    try:

        # emotion detection
        text_emotion = extract_text_emotion(text)
        audio_emotion = extract_audio_emotion(audio)
        face_emotion = extract_face_emotion(image)

        combined_emotion = combine_emotions(
            text_emotion,
            audio_emotion,
            face_emotion
        )

        # feature extraction for fusion
        image_feat = extract_image_features(image)
        audio_feat = extract_audio_features(audio)

        text_feat = torch.zeros((1,768))

        fused = fusion_model(
            text_feat,
            image_feat,
            audio_feat
        )

        # prompt
        if image is not None:

            prompt = f"""
<image>

User message: {text}

Emotion Analysis

Text Emotion: {text_emotion}
Voice Emotion: {audio_emotion}
Face Emotion: {face_emotion}

Combined Multimodal Emotion: {combined_emotion}

You are a compassionate AI therapist.

Respond with:
1. Empathy
2. What the user may be feeling
3. One coping suggestion
4. One comforting closing
"""

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device)

        else:

            prompt = f"""
User message: {text}

Emotion Analysis

Text Emotion: {text_emotion}
Voice Emotion: {audio_emotion}
Face Emotion: {face_emotion}

Combined Multimodal Emotion: {combined_emotion}

You are a compassionate AI therapist.

Respond with:
1. Empathy
2. What the user may be feeling
3. One coping suggestion
4. One comforting closing
"""

            inputs = processor(
                text=prompt,
                return_tensors="pt"
            ).to(device)

        output = llava_model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True
        )

        response = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return response

    except Exception as e:

        return f"Error occurred: {str(e)}"

# =========================================
# GRADIO UI
# =========================================

app = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Image(type="pil", label="Face Image (optional)"),
        gr.Textbox(label="Your Message"),
        gr.Audio(type="filepath", label="Voice Emotion (optional)")
    ],
    outputs=gr.Textbox(
        label="Therapist Reply",
        lines=29
    ),
   

    title="Multimodal AI Therapist",
    description="Uses text, facial cues and voice emotion."
)

app.launch()