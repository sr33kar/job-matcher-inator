import csv
from jobspy import scrape_jobs

jobs = scrape_jobs(
    site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
    # site_name=["linkedin"],
    search_term="software engineer",
    google_search_term="entry level and mid level software engineer jobs",
    location="United States",
    results_wanted=500,
    hours_old=10,
    country_indeed='USA',
    job_type='fulltime',
    linkedin_fetch_description=True, # gets more info such as description, direct job url (slower)
    # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],

    verbose=2
)
print(f"Found {len(jobs)} jobs")
print(jobs.head())
jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False) # to_excel