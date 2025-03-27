# Job Matching System

https://github.com/user-attachments/assets/10cd5963-bf1a-41c5-a177-ec0f52054d65

## Overview

The Job Matching System is a Python-based application that helps users find eligible jobs based on their resume. The system uses BERT embeddings for semantic matching and cosine similarity to compare the resume with job descriptions. It also filters out jobs based on specific criteria (e.g., excluding jobs with certain keywords or invalid locations).

## Features
Job Scraping:
- Pulls all the jobs on the internet from the job portals which we can customize.
- Allows user to provide a timeline to scrape jobs posted in that duration.
  
Resume Parsing:
- Supports Markdown (.md) and PDF (.pdf) resume formats.
- Extracts text from the resume for processing.

Job Matching:
- Uses BERT embeddings for semantic understanding of job descriptions and resumes.
- Computes cosine similarity to rank jobs based on relevance.

Filtering:
- Excludes jobs containing specific keywords (e.g., "citizenship", "visa", "sponsorship").
- Removes jobs with invalid locations (e.g., locations without a comma).
- I assume all of the locations on US has a comma, you can remove/update location filtering to your need.

Interactive Interface:
- Built using Streamlit, providing a user-friendly web interface.
- Displays eligible jobs in a DataFrame-like format with clickable URLs.

Cache System:
- Caches BERT embeddings to improve performance.
- Allows clearing the cache manually or automatically on app restart.

Export Results:
- Allows users to download eligible jobs as a CSV file.

## Setup Instructions
Prerequisites
- Python <4.0, >=3.10

Required Libraries:

- Install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

NLTK Data:
- Download NLTK data files (stopwords and tokenizers):
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
## Job Data:

Prepare a CSV file (jobs.csv) containing job data with the following columns:
- id: Unique identifier for each job.
- title: Job title.
- description: Job description.
- location: Job location (must contain a comma for valid locations).
- job_url: URL to the job posting.
- Other optional columns (e.g., company, date_posted, job_type).

## Running the Application
Clone the Repository:
```bash
git clone https://github.com/sr33kar/job-matcher-inator.git
cd job-matching-system
```

## Scrape jobs from all over the internet
```bash
python scrape.py
```

## Run the Streamlit App:

```bash
streamlit run job_matching_app.py
```

## Access the App:

Open your browser and navigate to http://localhost:8501.

## Usage
Upload Resume:
- Upload your resume in Markdown (.md) or PDF (.pdf) format.

Adjust Threshold:
- Use the slider to set the similarity threshold (default: 0.5).

View Results:
- The app displays eligible jobs in a tabular format.

Click on the job_url to open the job posting in a new tab.

Download Results:
- Click the Download Eligible Jobs as CSV button to save the results.

File Structure
```bash
job-matching-system/
├── job_matching_app.py       # Main application script
├── jobs.csv                  # Job data (CSV file)
├── README.md                 # Project documentation
├── requirements.txt          # List of dependencies
└── resume.md                 # Example resume (Markdown format)
```
## Customizations

**Filter Words:**

Modify the filter_words list in the code to exclude jobs containing specific keywords:
```bash
filter_words = ['citizenship', 'visa', 'sponsorship']
```
**Location Format:**

Adjust the location filter to match your desired format (e.g., City, State or City, Country).

**Advanced Matching:**

Replace BERT with other pre-trained models (e.g., RoBERTa, DistilBERT) for better performance.

**Styling:**

Customize the Streamlit interface using CSS or Streamlit's theming options.

### Example Job Data
Here’s an example of the jobs.csv file:
```bash
id	title	description	location	job_url
1	Software Engineer	Develop software applications.	San Francisco, CA	https://example.com/job/1
2	Data Scientist	Analyze data and build models.	New York, NY	https://example.com/job/2
3	Product Manager	Manage product development.	Remote	https://example.com/job/3
4	Business Analyst	Work with stakeholders.	Chicago, IL	https://example.com/job/4
5	UX Designer	Design user interfaces.	London	https://example.com/job/5
```

### Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with a detailed description of your changes.


### Contact
For questions or feedback, please contact:

**Sreekar Gadasu**

Email: gadasusreekar@gmail.com
GitHub: sr33kar

## License

[MIT](https://choosealicense.com/licenses/mit/)
