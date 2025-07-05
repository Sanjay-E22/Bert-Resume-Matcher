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
        "Job Title": "Data Scientist",
        "Job Description": "Analyze data trends and build predictive models.",
        "Required Skills": "Python, SQL, Machine Learning, Statistics"
    },
    {
        "Job Title": "Web Developer",
        "Job Description": "Develop responsive web applications and maintain front-end features.",
        "Required Skills": "HTML, CSS, JavaScript, React"
    },
    {
        "Job Title": "AI Engineer",
        "Job Description": "Build NLP models using BERT and deploy scalable solutions.",
        "Required Skills": "Python, PyTorch, BERT, NLP"
    }
]

jobs_df = pd.DataFrame(job_data)
jobs_df["combined"] = jobs_df["Job Title"] + ". " +                       jobs_df["Job Description"] + " Skills: " +                       jobs_df["Required Skills"]
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
