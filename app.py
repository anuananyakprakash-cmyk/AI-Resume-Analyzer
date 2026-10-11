import streamlit as st
import re
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume to find suitable job roles and skills to improve.")

job_roles = {
    "Data Analyst": [
        "python", "sql", "excel", "pandas", "statistics",
        "data analysis", "visualization", "power bi"
    ],
    "Machine Learning Engineer": [
        "python", "machine learning", "scikit learn",
        "statistics", "pandas", "numpy", "model training"
    ],
    "AI Engineer": [
        "python", "artificial intelligence", "machine learning",
        "deep learning", "neural networks", "tensorflow"
    ],
    "NLP Engineer": [
        "python", "nlp", "natural language processing",
        "text processing", "transformers", "machine learning"
    ],
    "Computer Vision Engineer": [
        "python", "computer vision", "opencv", "deep learning",
        "image processing", "tensorflow", "pytorch"
    ]
}

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            resume_text = " ".join(
                page.extract_text() or "" for page in reader.pages
            )
        else:
            document = Document(uploaded_file)
            resume_text = " ".join(
                paragraph.text for paragraph in document.paragraphs
            )

        resume_text = re.sub(r"\s+", " ", resume_text).strip()

        if not resume_text:
            st.error("No text found. Please upload a text-based resume.")
        else:
            st.subheader("Resume Text Extracted")
            st.write(resume_text[:1500])

            resume_lower = resume_text.lower()
            all_skills = sorted(set(
                skill
                for skills in job_roles.values()
                for skill in skills
            ))

            detected_skills = [
                skill for skill in all_skills
                if re.search(
                    r"(?<!\w)" + re.escape(skill) + r"(?!\w)",
                    resume_lower
                )
            ]

            st.subheader("Skills Detected")
            if detected_skills:
                st.write(", ".join(detected_skills))
            else:
                st.write("No listed skills detected.")

            results = []

            for role, skills in job_roles.items():
                role_text = " ".join(skills)
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(
                    [resume_text, role_text]
                )
                

                matched = [
                    skill for skill in skills
                    if skill in detected_skills
                ]
                missing = [
                    skill for skill in skills
                    if skill not in detected_skills
                ]
                score = (len(matched) / len(skills) * 100) if skills else 0
                

                results.append({
                    "Job Role": role,
                    "Match Score (%)": round(score, 2),
                    "Matched Skills": ", ".join(matched) or "None",
                    "Skills to Learn": ", ".join(missing) or "None"
                })

            results.sort(
                key=lambda item: item["Match Score (%)"],
                reverse=True
            )

            st.subheader("Top 3 Recommended Job Roles")
            st.dataframe(pd.DataFrame(results[:3]), hide_index=True)

            selected_role = st.selectbox(
                "Choose a job role",
                list(job_roles.keys())
            )

            selected_result = next(
                item for item in results
                if item["Job Role"] == selected_role
            )

            st.subheader("Selected Role Analysis")
            st.metric(
                "Estimated Text Similarity",
                f'{selected_result["Match Score (%)"]}%'
            )

            st.write("**Skills to learn:**")
            missing_skills = [
                skill for skill in job_roles[selected_role]
                if skill not in detected_skills
            ]

            if missing_skills:
                for skill in missing_skills:
                    st.write(f"- {skill}")
            else:
                st.write("All listed skills were detected.")

            st.subheader("Basic Learning Roadmap")
            if missing_skills:
                for number, skill in enumerate(missing_skills, 1):
                    st.write(f"{number}. Learn and practise **{skill}**.")
            else:
                st.write("Continue building projects and practising interview questions.")

            csv_data = pd.DataFrame(results).to_csv(index=False)
            st.download_button(
                "Download Analysis as CSV",
                data=csv_data,
                file_name="resume_analysis.csv",
                mime="text/csv"
            )

            st.caption(
                "For educational and career guidance only. "
                "Match scores are estimates, not hiring decisions."
            )

    except Exception as error:
        st.error(f"Could not analyze this file: {error}")
