import streamlit as st
import os
from io import StringIO
import pandas as pd
import numpy as np
import shutil # For cleaning up temp directory

# Import custom utility functions
from utils.pdf_parser import parse_pdf
from utils.docx_parser import parse_docx
from utils.resume_parser import parse_resume, parse_job_description
from utils.similarity_models import get_bert_embedding, calculate_cosine_similarity_bert, calculate_tfidf_cosine_similarity

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Resume Screening AI",
    layout="wide", # Use wide layout for better display of dataframes
    initial_sidebar_state="expanded"
)

# --- Title and Subheader ---
st.title("🎯 Resume Screening AI")
st.subheader("Match Resumes to Job Descriptions Using NLP")

# --- Sidebar for settings ---
st.sidebar.header("Configuration")
similarity_metric = st.sidebar.selectbox(
    "Choose Similarity Metric:",
    ("BERT Embeddings (Recommended)", "TF-IDF + Cosine Similarity"),
    help="BERT embeddings provide more semantic understanding, while TF-IDF is simpler and faster."
)
min_match_score = st.sidebar.slider(
    "Minimum Match Score (%)",
    0, 100, 60, # Min, Max, Default
    help="Only resumes with a match score equal to or higher than this percentage will be displayed."
)
st.sidebar.markdown("---")
st.sidebar.info("Upload resume files (PDF/DOCX) and a Job Description (TXT) to start the screening process.")

# --- Create a temporary directory for file uploads if it doesn't exist ---
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Function to clean up the temporary directory
def clean_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR) # Recreate empty directory
        st.success("Temporary files cleaned up.")

st.sidebar.button("Clean Temporary Files", on_click=clean_temp_dir, help="Deletes all uploaded files from the temporary directory.")


# --- Main Application Area ---

# 1. Job Description Upload
st.header("1. Upload Job Description")
jd_file = st.file_uploader(
    "Upload Job Description (TXT file)",
    type=["txt"],
    help="Please upload a plain text file containing the job description."
)
jd_text = ""
parsed_jd = None

if jd_file:
    try:
        # Read the content of the uploaded JD file
        stringio = StringIO(jd_file.getvalue().decode("utf-8"))
        jd_text = stringio.read()
        st.text_area("Job Description Content (Preview):", jd_text, height=200, disabled=True)

        # Parse the job description
        parsed_jd = parse_job_description(jd_text)
        
        # Display extracted JD details
        with st.expander("Extracted Job Description Details"):
            st.json({
                "Job Title": parsed_jd["job_title"],
                "Required Skills": parsed_jd["required_skills"],
                "Required Experience (Years)": parsed_jd["required_experience"]
            })
        st.success("Job Description parsed successfully!")
    except Exception as e:
        st.error(f"Error reading or parsing Job Description: {e}")
        jd_file = None # Reset jd_file to prevent further processing issues
else:
    st.warning("Please upload a Job Description to proceed with resume screening.")


# 2. Resume Upload
st.header("2. Upload Resumes")
resume_files = st.file_uploader(
    "Upload Resume Files (PDF, DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    help="You can upload multiple PDF or DOCX resume files at once."
)

if jd_file and parsed_jd and resume_files:
    st.write(f"Processing {len(resume_files)} resume(s)... This may take a moment.")

    results = []
    jd_embedding = None

    # Pre-calculate JD embedding if BERT is selected, to avoid recalculating in loop
    if similarity_metric == "BERT Embeddings (Recommended)":
        with st.spinner("Generating Job Description embedding..."):
            jd_embedding = get_bert_embedding(parsed_jd["processed_text"])
            if jd_embedding is None or jd_embedding.numel() == 0:
                st.error("Failed to generate BERT embedding for Job Description. Please check model download.")
                st.stop() # Stop execution if JD embedding fails

    progress_text = "Processing resumes..."
    progress_bar = st.progress(0, text=progress_text)

    for i, resume_file in enumerate(resume_files):
        resume_name = resume_file.name
        
        # Save the uploaded file to a temporary location for parsing
        temp_file_path = os.path.join(TEMP_DIR, resume_name)
        try:
            with open(temp_file_path, "wb") as f:
                f.write(resume_file.getbuffer())
        except Exception as e:
            st.error(f"Error saving {resume_name} to temporary directory: {e}. Skipping.")
            continue

        resume_text = ""
        if resume_name.endswith(".pdf"):
            resume_text = parse_pdf(temp_file_path)
        elif resume_name.endswith(".docx"):
            resume_text = parse_docx(temp_file_path)
        else:
            st.error(f"Unsupported file type for {resume_name}. Skipping.")
            continue

        if not resume_text:
            st.warning(f"Could not extract text from {resume_name}. It might be empty or corrupted. Skipping.")
            continue

        # Parse the resume
        parsed_resume = parse_resume(resume_text)

        # Calculate similarity score based on selected metric
        score = 0.0
        if similarity_metric == "BERT Embeddings (Recommended)":
            resume_embedding = get_bert_embedding(parsed_resume["processed_text"])
            if resume_embedding is not None and resume_embedding.numel() > 0:
                score = calculate_cosine_similarity_bert(jd_embedding, resume_embedding)
            else:
                st.warning(f"Could not generate BERT embedding for {resume_name}. Score set to 0.")
        else: # TF-IDF
            score = calculate_tfidf_cosine_similarity(parsed_jd["processed_text"], parsed_resume["processed_text"])

        match_score = round(score * 100, 2)

        # Calculate skill match details
        matched_skills = list(set(parsed_jd["required_skills"]) & set(parsed_resume["skills"]))
        missing_skills = list(set(parsed_jd["required_skills"]) - set(parsed_resume["skills"]))
        
        # Determine experience match status
        experience_match_status = "N/A"
        if parsed_jd["required_experience"] > 0:
            if parsed_resume["years_experience"] >= parsed_jd["required_experience"]:
                experience_match_status = "Meets/Exceeds requirement"
            else:
                experience_match_status = f"Below requirement (has {parsed_resume['years_experience']} yrs, needs {parsed_jd['required_experience']} yrs)"
        else:
            experience_match_status = "No specific requirement in JD"


        results.append({
            "Resume File": resume_name,
            "Candidate Name": parsed_resume["name"] if parsed_resume["name"] else "Unknown Candidate",
            "Match Score (%)": match_score,
            "Years Experience": parsed_resume["years_experience"],
            "Experience Match": experience_match_status,
            "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
            "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
            "Raw Resume Text (for debug)": resume_text # Include for debugging if needed
        })
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(resume_files), text=f"Processing {resume_name}...")

    progress_bar.empty() # Clear the progress bar after completion

    # Sort results by match score in descending order
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

    st.header("3. Screening Results")

    # Filter by minimum match score
    filtered_df = results_df[results_df["Match Score (%)"] >= min_match_score]

    if not filtered_df.empty:
        st.success(f"Found {len(filtered_df)} candidate(s) matching the criteria (Match Score >= {min_match_score}%).")
        
        # Display the dataframe with conditional styling for match score
        st.dataframe(
            filtered_df.drop(columns=["Raw Resume Text (for debug)"]).style.applymap(
                lambda x: 'background-color: #d4edda' if isinstance(x, (int, float)) and x >= min_match_score else '',
                subset=["Match Score (%)"]
            ),
            use_container_width=True
        )

        # Output Example: Top Candidates Overview
        st.subheader("Top Candidates Overview:")
        num_to_show = min(5, len(filtered_df)) # Show up to 5 top candidates
        for idx in range(num_to_show):
            row = filtered_df.iloc[idx]
            st.markdown(f"#### {idx+1}. {row['Candidate Name']} (Match Score: {row['Match Score (%)']}%)")
            st.markdown(f"- **Resume File:** `{row['Resume File']}`")
            st.markdown(f"- **Years of Experience:** {row['Years Experience']} ({row['Experience Match']})")
            st.markdown(f"- **Skills Matched:** :green[{row['Matched Skills']}]")
            if row['Missing Skills'] != "None":
                st.markdown(f"- **Missing Skills:** :red[{row['Missing Skills']}]")
            st.markdown("---")
            
        # Download Results button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Results as CSV",
            data=csv,
            file_name="resume_screening_results.csv",
            mime="text/csv",
            help="Download the table above as a CSV file."
        )
    else:
        st.warning(f"No candidates found with a match score of {min_match_score}% or higher. Try adjusting the minimum match score in the sidebar.")

else:
    st.info("Please upload a Job Description and at least one Resume file to see the screening results.")

