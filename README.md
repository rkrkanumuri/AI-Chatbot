# AI-Enhanced Hybrid Chatbot

## Overview

This project implements a hybrid chatbot that combines traditional Natural Language Processing (NLP) techniques with Microsoft Azure AI Language Services. 

The chatbot uses:
- TF-IDF vectorization for text representation
- Cosine similarity for intent classification
- Azure AI sentiment analysis for emotional awareness

The result is a rule-based chatbot enhanced with cloud-based AI services.

---

## Project Purpose

This project was developed as part of the course:

**MSAI631 – Artificial Intelligence for Human–Computer Interaction**  
University of the Cumberlands

The objective was to integrate a traditional chatbot with a cloud-based AI-as-a-service platform using the free tier of Azure AI Services.

---

## Features

- Intent classification using TF-IDF
- Cosine similarity confidence scoring
- Azure AI sentiment analysis integration
- Emotion-aware response adaptation
- Fallback mechanism for low-confidence predictions
- Modern web-based user interface (Flask)
- Secure API key management using environment variables

---

## System Architecture

User Input  
→ Text Preprocessing (NLTK)  
→ TF-IDF Vectorization  
→ Cosine Similarity Matching  
→ Azure AI Sentiment Analysis  
→ Response Enhancement  
→ JSON Response to Frontend  

This architecture demonstrates hybrid AI integration.

---

## Technologies Used

- Python
- Flask
- Scikit-learn
- NLTK
- Azure AI Language Service
- HTML / CSS / JavaScript

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Enhanced-Chatbot.git
cd AI-Enhanced-Chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Azure Credentials
Create a .env file in the root directory

```bash
AZURE_ENDPOINT=your_azure_endpoint_here
AZURE_KEY=your_azure_key_here
```

Do NOT hardcode API keys in the source code.

### 4. Run the application

```bash
python app.py
```

Open you browser and Navigate to:

```
http://127.0.0.1:5000
```

---

## 👨‍💻 Author

Ravi Kiran Raju Kanumuri  
MSAI631 – Artificial Intelligence for Human-Computer Interaction  
University of the Cumberlands  

---

## 📄 License

This project was developed for academic purposes.
