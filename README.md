# 🎗️ AI Breast Cancer Advisor

An Arabic-language AI assistant for breast cancer awareness, combining a RAG-based Q&A chatbot with an image classification model for mammogram analysis.

---

## ✨ Features

- 💬 **Arabic Chatbot** — Answers questions about breast self-examination and prevention using RAG (Retrieval-Augmented Generation)
- 🖼️ **Image Classifier** — Classifies mammogram images as **Normal**, **Benign**, or **Malignant**
- 🗂️ **Local Vector Store** — Uses ChromaDB to retrieve relevant content from medical text files
- 🌐 **Gradio UI** — Simple web interface with RTL Arabic support

---

## 🗂️ Project Structure

```
AI-Breast-Cancer-Advisor/
├── main2.py                  # Main app (Gradio UI + chatbot + image classifier)
├── image.py                  # Image preprocessing and prediction logic
├── vector.py                 # ChromaDB vector store setup and retriever
├── breast_prevention.txt     # Prevention tips (Arabic)
├── self_examination.txt      # Self-examination guide (Arabic)
├── normal.jpg                # Sample normal mammogram
├── benign.jpg                # Sample benign tumor mammogram
├── malignent.jpg             # Sample malignant tumor mammogram
├── breast_xception (1).h5    # ⚠️ Not included (see below)
└── README.md
```

> ⚠️ **The `.h5` model file is excluded from this repo** due to its large size. See the setup instructions below to obtain it.

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-Breast-Cancer-Advisor.git
cd AI-Breast-Cancer-Advisor
```

### 2. Install dependencies
```bash
pip install langchain langchain-ollama langchain-chroma gradio opencv-python keras tensorflow
```

### 3. Install and run Ollama models
```bash
# Install Ollama: https://ollama.com
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
```

### 4. Add the model file
Place your `breast_xception (1).h5` model file in the project root directory.  
*(The model is trained on mammogram images with 3 classes: Normal, Benign, Malignant)*

### 5. Run the app
```bash
python main2.py
```

Then open your browser at `http://localhost:7860`

---

## 🧠 How It Works

### Chatbot (RAG)
1. User asks a question in Arabic
2. ChromaDB retrieves relevant chunks from the `.txt` knowledge files
3. Retrieved context + question is sent to the local LLM (`qwen3:0.6b` via Ollama)
4. The model responds in Arabic only

### Image Classifier
1. User uploads a mammogram image
2. Image is resized to 224×224 and normalized
3. The Xception-based model predicts the class with confidence score
4. Result is displayed in Arabic

---

## 📋 Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally
- GPU recommended for faster inference

---

## ⚠️ Disclaimer

This tool is for **educational and awareness purposes only**. It is **not a substitute** for professional medical diagnosis. Always consult a qualified healthcare provider.

---

## 📄 License

See [LICENSE](LICENSE) for details.
