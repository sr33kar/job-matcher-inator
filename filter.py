from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import streamlit as st
import os
import PyPDF2  # Add PyPDF2 for PDF support

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

# Function to extract years of experience using regex
def extract_experience(text):
    # Expanded pattern to cover various ways of stating experience
    pattern = r'(\d+)\s*\+?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:experience|work\s*experience|professional\s*experience|relevant\s*experience|in\s*(?:the\s*field|[a-zA-Z\s]*))?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit App
def main():
    st.title("Job Matching System")
    st.write("Upload your resume and find eligible jobs!")

    # Dropdown for file type selection
    file_type = st.selectbox("Select resume file type", ["Markdown (MD)", "PDF"])

    # Upload resume based on selected file type
    if file_type == "Markdown (MD)":
        uploaded_file = st.file_uploader("Upload your resume (Markdown)", type="md")
    else:
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    if uploaded_file:
        # Read text from the uploaded file
        if file_type == "PDF":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode('utf-8')
        st.write("Resume text extracted successfully!")

        # Load job data
        jobs = pd.read_csv('jobs.csv')
        st.write(f"Loaded {len(jobs)} jobs.")

        # Handle missing descriptions
        jobs['description'] = jobs['description'].fillna('')

        # Remove jobs containing the word "citizenship"
        filter_title = ['citizenship', 'senior', 'lead', 'Sr', '.Net', 'Clearance', 'Secret', 'Manager', 'Mgr', 'US Citizen']
        filter_description = ['citizenship', 'Clearance', 'Secret', 'TS/SCI', 'Citizen']
        filter_companies = ['Dice']

        jobs = jobs[~(
            jobs['title'].str.contains('|'.join(filter_title), case=False, na=False) |
            jobs['description'].str.contains('|'.join(filter_description), case=False, na=False) |
            jobs['company'].str.contains('|'.join(filter_companies), case=False, na=False)
        )]

        # Remove jobs where 'location' does not contain a comma
        jobs = jobs[jobs['location'].str.contains(',', na=False)]

        st.write(f"Filtered out jobs with titles containing {filter_title}.")
        st.write(f"Filtered out jobs with description containing {filter_description}.")
        st.write(f"Filtered out jobs with company name containing {filter_companies}.")
        st.write(f"Remaining jobs: {len(jobs)}")
        
        # Preprocess resume text
        resume_embedding = get_bert_embedding(resume_text)
        resume_experience = extract_experience(resume_text)

        
        # Cache BERT embeddings for each job
        if 'job_embeddings' not in st.session_state:
            st.session_state.job_embeddings = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (_, job) in enumerate(jobs.iterrows()):
                job_text = job['title'] + ' ' + job['description']
                st.session_state.job_embeddings[job['id']] = get_bert_embedding(job_text)

                # Update progress bar
                progress = int((i + 1) / len(jobs) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Computing embeddings for job {i + 1} of {len(jobs)}...")

            st.write("BERT embeddings computed and cached!")

        # Compute similarity for each job using cached embeddings
        similarity_scores = []
        job_experiences = []
        for _, job in jobs.iterrows():
            job_embedding = st.session_state.job_embeddings[job['id']]
            job_experience = extract_experience(job['description'])

            # BERT similarity
            bert_similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]

            # Experience match
            experience_match = 1 if resume_experience >= job_experience else 0

            # Combine scores (70% BERT similarity, 30% experience match)
            final_score = 0.4 * bert_similarity + 0.6 * experience_match
            similarity_scores.append(final_score)
            job_experiences.append(job_experience)

        # Add similarity scores to jobs dataframe
        jobs['similarity_score'] = similarity_scores
        jobs['job_experience'] = job_experiences
        # Filter jobs based on a threshold
        threshold = st.slider("Set similarity threshold", 0.0, 1.0, 0.9)
        eligible_jobs = jobs[jobs['similarity_score'] > threshold]

        # Display eligible jobs
        st.write(f"Found {len(eligible_jobs)} eligible jobs:")
        st.dataframe(eligible_jobs, column_config={"job_url": st.column_config.LinkColumn()})

        # Save eligible jobs to a CSV file
        if len(eligible_jobs) > 0:
            csv = eligible_jobs.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Eligible Jobs as CSV",
                data=csv,
                file_name="eligible_jobs.csv",
                mime="text/csv",
            )

# Run the Streamlit app
if __name__ == "__main__":
    main()