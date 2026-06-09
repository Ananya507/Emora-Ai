# 🧠 Emora AI: Multimodal Emotion-Aware Virtual Therapist

### Understanding Emotions Beyond Words 💙

Emora AI is an intelligent therapeutic assistant that leverages **Multimodal Emotion Recognition** to understand users through their **text, facial expressions, and voice cues**. By combining insights from multiple modalities and utilizing a powerful **Vision-Language Model (LLaVA-1.5)**, Emora AI delivers empathetic, context-aware, and emotionally supportive responses.

The system integrates state-of-the-art models for emotion detection, multimodal feature fusion, and response generation, creating a more human-centered conversational experience than traditional text-only chatbots.

---

## 🌟 Key Features

* 📝 **Text Emotion Recognition** using DistilRoBERTa
* 😊 **Facial Emotion Analysis** using DeepFace
* 🎤 **Voice Emotion Detection** using Wav2Vec2
* 🔗 **Cross-Modal Attention-Based Feature Fusion**
* 🤖 **LLaVA-Powered Therapeutic Conversations**
* 💙 **Emotion-Aware Personalized Responses**
* 🌐 **Interactive Gradio Interface**
* ⚡ Real-time Multimodal Processing

---

## 🎯 Problem Statement

Most conversational AI systems rely solely on textual information, ignoring valuable emotional cues expressed through facial expressions and vocal tones. As a result, responses often lack emotional awareness and personalization.

Emora AI addresses this challenge by combining information from three modalities—text, image, and audio—to better understand a user's emotional state and provide more compassionate, contextually relevant support.

---

## 🏗️ System Architecture

```text
                    User Input
              ┌────────┼────────┐
              │        │        │
            Text     Image    Audio
              │        │        │
              ▼        ▼        ▼

      DistilRoBERTa  DeepFace  Wav2Vec2
      Emotion Model  Emotion   Audio Features
                     Analysis

              └────────┼────────┘
                       ▼

         Vision Transformer (ViT)
              Image Embeddings

                       ▼

      Cross-Modal Attention Fusion
             Multi-Head Attention

                       ▼

         Unified Emotional State

                       ▼

              LLaVA-1.5 (7B)

                       ▼

     Empathetic Therapeutic Response
```

---

## 🚀 Technologies Used

### Deep Learning & AI

* PyTorch
* Hugging Face Transformers
* TIMM

### Models

* LLaVA-1.5-7B
* DistilRoBERTa Emotion Classifier
* Wav2Vec2
* Vision Transformer (ViT)
* DeepFace

### Libraries

* Gradio
* Librosa
* NumPy
* TorchVision
* PIL

---

## 📂 Project Structure

```bash
Emora-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── assets/
│   ├── screenshots/
│   └── demo/
│
└── models/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/emora-ai.git

cd emora-ai
```

### Install Dependencies

```bash
pip install gradio transformers timm librosa accelerate torchvision deepface
```

or

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python app.py
```

After launching, Gradio will generate a local URL where users can:

1. Upload a facial image (optional)
2. Enter a text message
3. Upload or record an audio sample (optional)
4. Receive an emotionally aware therapeutic response

---

## 🧩 Methodology

### 1. Text Emotion Recognition

The user's textual input is analyzed using a pretrained DistilRoBERTa emotion classification model to identify emotions such as:

* Happiness
* Sadness
* Anger
* Fear
* Surprise
* Neutral

---

### 2. Facial Emotion Recognition

Facial expressions are analyzed using DeepFace to determine the dominant emotional state present in the uploaded image.

Examples include:

* Happy
* Sad
* Angry
* Fearful
* Neutral
* Surprise

---

### 3. Voice Emotion Analysis

Audio inputs are processed using Wav2Vec2 embeddings and speech energy features to estimate emotional characteristics and vocal intensity.

---

### 4. Feature Extraction

#### Image Features

* Vision Transformer (ViT)
* 768-dimensional visual embeddings

#### Audio Features

* Wav2Vec2 contextual embeddings
* 768-dimensional audio representations

#### Text Features

* Emotion-aware textual representations

---

### 5. Cross-Modal Fusion

A custom CrossModalFusion module combines information from:

* Text
* Image
* Audio

using Multi-Head Self-Attention.

This enables the model to learn relationships between modalities and generate a unified emotional representation.

---

### 6. Therapeutic Response Generation

The detected emotional context is integrated into prompts supplied to LLaVA-1.5.

LLaVA generates responses that:

* Demonstrate empathy
* Acknowledge emotional states
* Suggest coping strategies
* Provide supportive guidance

---

## 📊 Key Contributions

* Developed a multimodal emotion-aware conversational system.
* Integrated text, audio, and facial emotion recognition into a unified framework.
* Implemented attention-based multimodal feature fusion.
* Leveraged LLaVA for context-aware therapeutic response generation.
* Designed an interactive Gradio interface for real-time user interaction.

---

## 💡 Future Enhancements

* Real-time webcam emotion detection
* Live speech conversation support
* Personalized emotional memory
* Emotion trend visualization dashboard
* Crisis detection and escalation support
* Multilingual emotion recognition
* Mobile application deployment
* Fine-tuned multimodal emotion classification model

---

## 🎯 Applications

* Mental Health Support Systems
* AI Therapy Assistants
* Emotionally Intelligent Chatbots
* Human-Computer Interaction Research
* Affective Computing
* Healthcare AI Solutions
* Educational Well-being Platforms

---

## 📸 Demo

Add screenshots or GIF demonstrations here.

```text
Example:
assets/screenshots/homepage.png
assets/demo/emora_demo.gif
```

---

## 📈 Results

The system successfully demonstrates:

* Multimodal emotion understanding
* Emotion-aware reasoning
* Context-sensitive therapeutic conversations
* Improved emotional awareness compared to text-only systems

---

## 👩‍💻 Author

### Ananya Majumdar

Final Year Information Technology Student

Research Interests:

* Artificial Intelligence
* Multimodal Learning
* Emotion Recognition
* Generative AI
* Human-Centered AI

---

## 🙏 Acknowledgements

Special thanks to:

* Hugging Face Transformers
* PyTorch
* DeepFace
* TIMM
* Gradio
* Meta AI (LLaVA)

for providing the open-source tools and models that made this project possible.

---

## ⭐ Support

If you found this project useful, consider giving it a star.

**Emora AI — Understanding Emotions Beyond Words.** 💙
