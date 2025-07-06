import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.title("🔍 AI Resume Matcher")
st.write("Upload your resume and get matched with top job descriptions!")

resume_file = st.file_uploader("📄 Upload your resume (.txt or .pdf)", type=["pdf", "txt"])
top_k = st.slider("🔢 Select top K matches", 1, 10, 5)

# Use hardcoded job descriptions
job_data = [
  {
    "Job Title": "Software Engineer",
    "Job Description": "Design, develop, and maintain scalable software systems. Collaborate with product and design teams to deliver high-quality applications.",
    "Required Skills": "Java, Python, REST APIs, Git, Agile, SQL"
  },
  {
    "Job Title": "Frontend Developer",
    "Job Description": "Build responsive user interfaces using modern JavaScript frameworks. Ensure accessibility, performance, and cross-browser compatibility.",
    "Required Skills": "HTML, CSS, JavaScript, React, Tailwind, Git, Figma"
  },
  {
    "Job Title": "Backend Developer",
    "Job Description": "Develop RESTful APIs and server-side logic. Optimize database queries and manage backend infrastructure.",
    "Required Skills": "Node.js, Express, MongoDB, PostgreSQL, Docker, AWS"
  },
  {
    "Job Title": "Full-Stack Developer",
    "Job Description": "Handle both client-side and server-side components. Collaborate across teams to build end-to-end web applications.",
    "Required Skills": "MERN Stack, Django, APIs, CI/CD, MySQL, Cloud Deployment"
  },
  {
    "Job Title": "Data Analyst",
    "Job Description": "Analyze business data, generate insights, and create dashboards. Work closely with stakeholders to support data-driven decisions.",
    "Required Skills": "Excel, SQL, Power BI, Tableau, Python (Pandas), Statistics"
  },
  {
    "Job Title": "Data Scientist",
    "Job Description": "Build machine learning models, clean large datasets, and develop predictive analytics. Communicate findings with visualizations and reports.",
    "Required Skills": "Python, Scikit-learn, Pandas, NumPy, Matplotlib, ML Algorithms, SQL"
  },
  {
    "Job Title": "Machine Learning Engineer",
    "Job Description": "Design and deploy machine learning pipelines in production. Fine-tune models and work on model optimization at scale.",
    "Required Skills": "Python, TensorFlow, PyTorch, MLOps, Docker, Cloud (AWS/GCP)"
  },
  {
    "Job Title": "AI Engineer (NLP Focus)",
    "Job Description": "Build and fine-tune NLP models for tasks like sentiment analysis, chatbots, and summarization.",
    "Required Skills": "Hugging Face Transformers, BERT, spaCy, NLTK, Python, FastAPI"
  },
  {
    "Job Title": "DevOps Engineer",
    "Job Description": "Automate infrastructure, monitor CI/CD pipelines, and ensure deployment reliability.",
    "Required Skills": "Linux, Docker, Kubernetes, Jenkins, Terraform, AWS, Bash"
  },
  {
    "Job Title": "Technical Support Engineer",
    "Job Description": "Troubleshoot software issues, assist clients with installations and configurations, and escalate technical problems when needed.",
    "Required Skills": "Networking, Windows/Linux, SQL, Shell Scripting, Customer Communication, Jira"
  }
]

jobs_df = pd.DataFrame(job_data)
jobs_df["combined"] = jobs_df["Job Title"] + ". " +  jobs_df["Job Description"] + " Skills: " +   jobs_df["Required Skills"]
job_descriptions = jobs_df["combined"].tolist()

def load_resume_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if ext == ".pdf":
        return extract_text(uploaded_file)
    elif ext == ".txt":
        return uploaded_file.read().decode("utf-8")
    else:
        return None

if resume_file:
    with st.spinner("🔍 Matching your resume with job descriptions..."):
        resume_text = load_resume_text(resume_file)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
        cosine_scores = util.cos_sim(resume_embedding, job_embeddings)[0]
        top_k = min(top_k, len(cosine_scores))  # safeguard
        top_results = torch.topk(cosine_scores, k=top_k)
        st.success("✅ Matching Complete!")
        st.subheader("🎯 Top Job Matches:")
        for score, idx in zip(top_results[0], top_results[1]):
           i = idx.item()  # convert tensor to int
           st.markdown(f"**🧠 Job Title**: {jobs_df.iloc[i]['Job Title']}")
           st.markdown(f"**📄 Description**: {jobs_df.iloc[i]['Job Description']}")
           st.markdown(f"**🛠 Required Skills**: {jobs_df.iloc[i]['Required Skills']}")
           st.markdown(f"**🎯 Match Score**: `{score.item():.4f}`")
           st.markdown("---")
