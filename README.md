# 🧠 SkillNet AI — Job Skill Intelligence & Career Roadmap
SkillNet AI is an end-to-end AI-powered job intelligence system that analyzes real-world job postings, predicts required skills using deep learning, and generates personalized learning roadmaps using generative AI.

## 🚀 Features
### 📊 Job Market Insights
- Analyze 10K+ real job postings from multiple websites
- Identify top skills in demand
- Visualize domain-wise job distribution
- Track company-specific skill demand

### 🤖 Skill Predictor (Deep Learning)
- Built using DistilBERT (Transformer Model)
- Multi-label classification of job descriptions
- Hybrid inference:
  - Transformer predictions
  - Semantic similarity (Sentence Transformers)

### 🎓 AI Learning Roadmap Generator

- Powered by Google Gemini
- Generates structured 8-week learning plans
- Includes:
  - Weekly topics
  - Subtopics
  - Practical tasks

## 🏗️ System Architecture

Job Scraper -> Raw Dataset (CSV) -> Data Cleaning & Preprocessing -> Skill Labeling (SentenceTransformer) -> Training Dataset -> DistilBERT Multi-label Classifier -> Hybrid Prediction Layer -> Streamlit Dashboard -> Gemini AI Roadmap Generator

## 🧪 Tech Stack
### 🧠 AI / ML
- Transformers (DistilBERT)
- Sentence Transformers
- Scikit-learn
- PyTorch

### 📊 Data & Visualization
- Pandas
- Plotly

### 🌐 App
- Streamlit

### 🤖 Generative AI
- Google Gemini API

## 📈 Key Highlights

- 🔥 Hybrid AI system (Transformer + Semantic Similarity)
- 📊 Real-world dataset (10K+ jobs)
- 🤖 Multi-label classification model
- 🎯 Structured GenAI outputs (controlled prompting)
- 🖥️ Interactive dashboard

## 🧠 Future Improvements
- Skill gap analysis (user → target role)
- Resume-based skill extraction
- Trend detection (emerging skills)
- PDF export for learning plans

## 👨‍💻 Author

Rishitha Raj

AI | Data Science | Product Thinking
