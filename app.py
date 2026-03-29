import streamlit as st
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px # type: ignore
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="SkillNet AI", layout="wide")

skill_descriptions = {

"Data Engineering":
"building ETL pipelines data warehouses spark airflow kafka",

"DevOps":
"CI CD pipelines docker kubernetes infrastructure automation",

"Cloud Computing":
"AWS Azure GCP cloud infrastructure deployment",

"Platform Engineering":
"internal developer platforms infrastructure tooling scalability",

"Big Data":
"hadoop spark distributed data processing",

"Machine Learning":
"training machine learning models pytorch tensorflow",

"AI / NLP":
"natural language processing transformers language models",

"Cybersecurity":
"security encryption authentication vulnerability detection",

"Distributed Systems":
"distributed computing scalable microservices architecture",

"Data Science":
"statistical analysis predictive modelling experimentation"

}


def format_plan(plan):

    weeks = re.split(r"(Week \d+:)", plan)

    formatted = ""

    for i in range(1, len(weeks), 2):
        week_title = weeks[i]
        content = weeks[i+1]

        formatted += f"### {week_title}\n{content.strip()}\n\n"

    return formatted

# -----------------------------
# Load analytics data
# -----------------------------

skill_summary = pd.read_csv("data/skill_summary.csv")
domain_summary = pd.read_csv("data/domain_summary.csv")
company_skills = pd.read_csv("data/company_skill_demand.csv")

skill_names = list(skill_descriptions.keys())

skill_embeddings = semantic_model.encode(list(skill_descriptions.values()))

# -----------------------------
# Load trained model
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("model/skill_classifier")
model = AutoModelForSequenceClassification.from_pretrained("model/skill_classifier")

model.to(device)
model.eval()

labels = skill_summary["Skill"].tolist()

# -----------------------------
# Skill prediction function
# -----------------------------

def predict_skills(text):

    # -------- BERT prediction --------
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    bert_probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    # -------- semantic similarity --------
    job_embedding = semantic_model.encode([text])

    similarity = cosine_similarity(job_embedding, skill_embeddings)[0]

    # -------- combine scores --------
    combined_scores = 0.6 * bert_probs[:len(skill_names)] + 0.4 * similarity

    # -------- select top skills --------
    top_indices = combined_scores.argsort()[-3:][::-1]

    predicted = [skill_names[i] for i in top_indices]

    return predicted

# -----------------------------
# Dashboard title
# -----------------------------

st.title("🚀 SkillNet AI – Job Skill Intelligence")

tabs = st.tabs(["📊 Job Market Insights", "🤖 Skill Predictor", "🎓 Learning Roadmap"])

# -----------------------------
# TAB 1 – Market Insights
# -----------------------------

with tabs[0]:

    st.subheader("Top Skills in Job Market")

    fig1 = px.bar(skill_summary.head(10), x="Skill", y="Job_Count")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Skill Domains")

    fig2 = px.pie(domain_summary, values="Job_Count", names="Domain")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top Skills by Company")

    fig3 = px.bar(company_skills.head(10), x="company", y="job_count", color="skill_labels")
    st.plotly_chart(fig3, use_container_width=True)


# -----------------------------
# TAB 2 – Skill Predictor
# -----------------------------

with tabs[1]:

    st.subheader("Predict Skills From Job Description")

    user_input = st.text_area("Paste a job description")

    if st.button("Predict Skills"):
        with st.spinner("Analyzing job description..."):
            skills = predict_skills(user_input)

        if skills:
            # st.success(f"Predicted Skills: {', '.join(skills)}")
            st.success("Predicted Skills")
            for s in skills:
                st.markdown(f"- **{s}**")
        else:
            st.warning("No skills detected")


# -----------------------------
# TAB 3 – Learning Roadmap
# -----------------------------

import re

with tabs[2]:

    valid_skills = [
        "Machine Learning", "Data Engineering", "DevOps",
        "Cloud Computing", "AI / NLP", "Big Data",
        "Cybersecurity", "Distributed Systems"
    ]

    st.subheader("Generate Learning Roadmap")

    skill_input = st.multiselect(
        "Select skills to learn",
        options=valid_skills,
        key="tab3_skill_selector"
    )

    if st.button("Generate Roadmap", key="tab3_generate_btn"):

        if not skill_input:
            st.warning("Please select at least one skill")
            st.stop()

        from utils.roadmap import learning_planner

        skills = skill_input

        with st.spinner("Generating AI roadmap..."):
            plan = learning_planner(skills)

        st.markdown("### 🎯 Personalized Learning Plan")

        # Extract each week block properly
        week_blocks = re.findall(r"(Week \d+:.*?)(?=Week \d+:|$)", plan, re.DOTALL)

        for block in week_blocks:

            # Extract full title (Week + Main Topic)
            week_title_match = re.search(r"(Week \d+:\s*.*)", block)
            subtopics_match = re.search(r"Subtopics:(.*)", block)
            task_match = re.search(r"Practical Task:(.*)", block)

            week_title = week_title_match.group(1).strip() if week_title_match else ""
            subtopics = subtopics_match.group(1).strip() if subtopics_match else "N/A"
            task = task_match.group(1).strip() if task_match else "N/A"

            st.markdown(
                f"""
                <div style="background:;padding:15px;border-radius:10px;margin-bottom:12px">
                    <h4 style="color:#4CAF50;">📅 {week_title}</h4>
                    <p><b>📚 Subtopics:</b> {subtopics}</p>
                    <p><b>🛠 Practical Task:</b> {task}</p>
                </div>
                """,
                unsafe_allow_html=True
            )