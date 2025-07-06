# 🧠 AI Resume Matcher – BERT + Streamlit

Hi there! 👋

This is a smart little tool I built to help job seekers (like me!) see which job descriptions best match their resume — using **AI**, not just keywords.

🚀 **Try it live**: 
https://sanjay-e22-bert-resume-matcher-app-zlz0tl.streamlit.app/



upload your resume (PDF or TXT), and the app:
1. Reads your resume using `pdfminer` or `txt` reader
2. Compares it to a list of job descriptions using **BERT embeddings**
3. Returns the **top matching jobs**, scored by semantic similarity — not just keywords!


## 🛠️ Tech Stack

- 🧠 **Sentence-BERT** (`all-MiniLM-L6-v2`)
- 🧾 **pdfminer.six** (to extract text from PDFs)
- 📊 **Pandas** (for handling job data)
- ⚙️ **PyTorch** (for similarity scoring)
- 🌐 **Streamlit** (for the interactive web UI)


## 📂 How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/bert-resume-matcher.git
   cd bert-resume-matcher


Let me know if you'd like this customized further with:
EMAIL: sanjayelangovan22@gmail.com
