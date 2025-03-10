from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import os
from datetime import datetime
import google.generativeai as genai
import PyPDF2 as pdf
from fpdf import FPDF
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load environment variables
load_dotenv()

# Configure Generative AI API
genai.configure(api_key=("AIzaSyC-qSvpq44LP0hYgr7EZLZaOIxlPzezP3g"))


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vectorstore
def setup_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory="cv_vectordb", embedding_function=embeddings)
    return vectorstore

# Convert PDF to text
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Retrieve relevant content from vectorstore
def retrieve_from_vectorstore(vectorstore, query):
    retriever = vectorstore.as_retriever()
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

# Get response from Generative AI
def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text if response else None

# Generate PDF report
def generate_pdf_report(candidate_name, report_content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, txt=f"Candidate Report: {candidate_name}", ln=True, align="L")
    pdf.ln(5)

    numbered_sections = {
        1: "Candidate Name and Email",
        2: '"Can Do" list:',
        3: '"Should Do" list',
        4: "Skill Comparison Table:",
        5: "Overall Matching Score:",
        6: "Analysis of Strengths and Weaknesses",
        7: "Recommendations for Improvement",
        8: "Conclusion on Fitment",
    }

    lines = report_content.splitlines()
    current_section = None
    bullet_point = "\u2022 "

    for line in lines:
        stripped_line = line.strip().replace("*", "")

        if stripped_line in numbered_sections.values():
            for number, section in numbered_sections.items():
                if stripped_line == section:
                    current_section = number
                    pdf.set_font("Arial", style="", size=11)
                    pdf.cell(0, 6, txt=f"{number}. {section}", ln=True, align="L")
                    pdf.ln(3)
                    break
        elif current_section and stripped_line.startswith("- "):
            pdf.set_font("Arial", size=10)
            pdf.cell(5)
            pdf.cell(0, 5, txt=f"{bullet_point}{stripped_line[2:]}", ln=True)
        elif "|" in stripped_line:
            cells = [cell.strip() for cell in stripped_line.split("|")[1:-1]]
            if len(cells) == 4:
                pdf.set_font("Arial", size=9)
                pdf.cell(50, 6, cells[0], border=1)
                pdf.cell(35, 6, cells[1], border=1, align="C")
                pdf.cell(35, 6, cells[2], border=1, align="C")
                pdf.cell(35, 6, cells[3], border=1, align="C")
                pdf.ln()
        else:
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, stripped_line)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{candidate_name}_report_{timestamp}.pdf"
    pdf.output(file_name)
    return file_name

# Setup vectorstore
vectorstore = setup_vectorstore()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Fitment Report API. Use the /generate_fitment_reports/ endpoint to generate reports."}

@app.post("/generate_fitment_reports/")
async def generate_fitment_reports(resumes: List[UploadFile] = File(...), job_description: UploadFile = File(...)):
    if not resumes or not job_description:
        raise HTTPException(status_code=400, detail="Please upload resumes and a job description.")

    try:
        job_description_text = input_pdf_text(job_description.file)
        company_culture_content = retrieve_from_vectorstore(vectorstore, "company culture match")

        fitment_results = []
        for resume_file in resumes:
            candidate_name = os.path.splitext(resume_file.filename)[0]
            resume_text = input_pdf_text(resume_file.file)

            input_prompt = f"""
### Task: Generate a candidate shortlisting report.
### Instructions:
You are a highly intelligent and unbiased system designed to shortlist candidates for a job based on:
1. The candidate's resume.
2. A provided job description.
3. Relevant company culture data retrieved from the vector database.
### Key Objectives:
- Analyze skills, qualifications, and experiences in the resume.
- Evaluate alignment with the job description.
- Assess cultural fit using company culture data.
- Provide detailed scoring, strengths, weaknesses, and recommendations.
### Required Sections in the Report:
- Candidate Name and Email
- Parse the job description and create a 'Should Do' list, categorizing required skills into levels: Beginner, Competent, Intermediate, Expert.
- Parse the candidate's resume and create a 'Can Do' list, categorizing listed skills into the same levels: Beginner, Competent, Intermediate, Expert.
- Matching score: A detailed table showing alignment of skills.
- Analysis of strengths and weaknesses.
- Recommendations for improvement.
- Overall conclusion.
### Input Data:
- **Resume**: {resume_text}
- **Job Description**: {job_description_text}
- **Company Culture Data**: {company_culture_content}
### Output Format:
1. Candidate Name and Email
2."Can Do" list:
3. "Should Do" list
4. Skill Comparison Table:
   | Skill                   | "Can Do" Level  | "Should Do" Level  | Matching Score |
   |--------------------------|----------------|--------------------|----------------|
5. Overall Matching Score: [Percentage]
6. Analysis of Strengths and Weaknesses
7. Recommendations for Improvement
8. Conclusion on Fitment
Note:Remove or do not generate the words 'Ok','Okay'and the sentence like 'Okay, I will generate a candidate shortlisting report for ' from the generated pdf of the  fitment report
            """

            report_content = get_gemini_response(input_prompt)

            if report_content:
                try:
                    matching_score = float(report_content.split("Overall Matching Score:")[1].split("%")[0].strip())
                except (IndexError, ValueError):
                    matching_score = 0.0
                    report_content += "\n\n[ERROR: Matching Score could not be parsed]"

                report_file = generate_pdf_report(candidate_name, report_content)
                fitment_results.append((candidate_name, matching_score, report_file))

        fitment_results.sort(key=lambda x: x[1], reverse=True)

        response_data = []
        for rank, (candidate_name, matching_score, report_file) in enumerate(fitment_results, start=1):
            response_data.append({
                "candidate_name": candidate_name,
                "matching_score": matching_score,
                "rank": rank,
                "report_file": report_file
            })

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
