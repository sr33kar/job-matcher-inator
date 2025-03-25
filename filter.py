from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import streamlit as st
import os
import PyPDF2  # Add PyPDF2 for PDF support
from rake_nltk import Rake  # For keyword extraction

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

# Function to extract keywords using RAKE
def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()  # Returns a list of ranked keywords

# Function to save applied jobs to a CSV file
def save_applied_jobs(applied_jobs):
    applied_df = pd.DataFrame(applied_jobs)
    applied_df.to_csv('applied.csv', index=False)

# Function to load applied jobs from a CSV file
def load_applied_jobs():
    if os.path.exists('applied.csv'):
        return pd.read_csv('applied.csv').to_dict('records')
    return []

# Streamlit App
def main():
    st.title("Job Matching System")
    st.write("Upload your resume and find eligible jobs!")

    # Initialize session state for applied jobs
    if 'applied_jobs' not in st.session_state:
        st.session_state.applied_jobs = load_applied_jobs()

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
        filter_title = ['citizenship', 'senior', 'lead', 'Sr', '.Net', 'Clearance', 'Secret', 'Manager', 'Mgr', 'US Citizen', 'Principal', 'Embedded', 'HVAC', 'Staff']
        filter_description = ['citizenship', 'Clearance', 'Secret', 'TS/SCI', 'Citizen']
        filter_companies = ['Sreekar\'s company']

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
        resume_keywords = extract_keywords(resume_text)
        resume_experience = extract_experience(resume_text)

        # Toggle for similarity method
        similarity_method = st.radio(
            "Select similarity method",
            ["BERT Embeddings + Cosine Similarity", "TF-IDF + Cosine Similarity"],
            index=1  # Default to BERT Embeddings + Cosine Similarity
        )

        # Compute similarity based on selected method
        if similarity_method == "BERT Embeddings + Cosine Similarity":
            # Get BERT embeddings for resume keywords
            resume_embedding = get_bert_embedding(' '.join(resume_keywords))

            # Cache BERT embeddings for each job's keywords
            if 'job_embeddings' not in st.session_state:
                st.session_state.job_embeddings = {}
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (_, job) in enumerate(jobs.iterrows()):
                    job_text = job['title'] + ' ' + job['description']
                    job_keywords = extract_keywords(job_text)
                    st.session_state.job_embeddings[job['id']] = get_bert_embedding(' '.join(job_keywords))

                    # Update progress bar
                    progress = int((i + 1) / len(jobs) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Computing embeddings for job {i + 1} of {len(jobs)}...")

                st.write("BERT embeddings computed and cached!")

            # Compute similarity for each job using cached BERT embeddings
            similarity_scores = []
            for _, job in jobs.iterrows():
                job_embedding = st.session_state.job_embeddings[job['id']]
                job_experience = extract_experience(job['description'])

                # BERT similarity
                bert_similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]

                # Experience match
                experience_match = 1 if resume_experience >= job_experience else 0

                final_score = bert_similarity * experience_match
                similarity_scores.append(final_score)

        else:  # TF-IDF + Cosine Similarity
            # Combine resume keywords and job keywords
            all_keywords = [' '.join(resume_keywords)] + [
                ' '.join(extract_keywords(job['title'] + ' ' + job['description'])) for _, job in jobs.iterrows()
            ]

            # Compute TF-IDF matrix
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_keywords)

            # Compute cosine similarity between resume and job keywords
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Add similarity scores to jobs dataframe
            similarity_scores = cosine_similarities

        # Add similarity scores to jobs dataframe
        jobs['similarity_score'] = similarity_scores

        # Filter jobs based on a threshold
        threshold = st.slider("Set similarity threshold", 0.0, 1.0, 0.9)
        eligible_jobs = jobs[jobs['similarity_score'] > threshold]

        # Add a checkbox column to the DataFrame
        eligible_jobs['Apply'] = eligible_jobs['id'].apply(
            lambda job_id: job_id in [applied_job['id'] for applied_job in st.session_state.applied_jobs]
        )

        # Reorder columns to place the checkbox next to the job name
        column_order = ['id', 'Apply', 'title', 'company', 'location', 'job_url', 'similarity_score', 'date_posted', 'company_num_employees']
        eligible_jobs = eligible_jobs[column_order]

        # Display the DataFrame with checkboxes
        st.write(f"Found {len(eligible_jobs)} eligible jobs:")
        edited_df = st.data_editor(
            eligible_jobs,
            column_config={
                "Apply": st.column_config.CheckboxColumn("Apply", help="Check to apply to this job"),
                "job_url": st.column_config.LinkColumn("Job URL"),
            },
            hide_index=True,
        )

        # Update applied jobs based on checkbox changes
        if len(eligible_jobs) > 0:
            applied_job_ids = edited_df[edited_df['Apply']]['id'].tolist()
            st.session_state.applied_jobs = [
                {
                    'id': job['id'],
                    'title': job['title'],
                    'company': job['company'],
                    'job_url': job['job_url'],
                    'date_posted': job['date_posted'],
                    'similarity_score': job['similarity_score']
                }
                for job in eligible_jobs.to_dict('records') if job['id'] in applied_job_ids
            ]

        # Save applied jobs to a CSV file
        if st.button("Save Applied Jobs"):
            save_applied_jobs(st.session_state.applied_jobs)
            st.write("Applied jobs saved to `applied.csv`.")

        # Display applied jobs
        st.write("### Applied Jobs")
        if st.session_state.applied_jobs:
            applied_df = pd.DataFrame(st.session_state.applied_jobs)
            st.dataframe(applied_df, column_config={"job_url": st.column_config.LinkColumn()})
        else:
            st.write("No jobs applied yet.")

# Run the Streamlit app
if __name__ == "__main__":
    main()