import os
import json
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
import PyPDF2

# ==============================
# CONFIGURATION
# ==============================
# 1. SETUP API KEY:
# Ideally, set this in your environment variables: export GEMINI_API_KEY="your_key"
# Or fallback to the provided key (ensure this key is valid)
api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY") 
client = genai.Client(api_key=api_key)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask App
# template_folder="templates" tells Flask where to look for ats_interface.html
app = Flask(__name__, template_folder="templates") 

# CRITICAL: Enable Cross-Origin Resource Sharing
# This allows your frontend (which might run as a file or different port) 
# to talk to this backend server without browser security blocking it.
CORS(app) 

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# HELPER FUNCTIONS
# ==============================

def clean_json_string(json_str):
    """
    Cleans the JSON string by removing markdown code blocks (```json ... ```) 
    if the AI mistakenly includes them.
    """
    if not json_str:
        return "{}"
    # Remove markdown code fences
    cleaned = re.sub(r'```json\s*', '', json_str)
    cleaned = re.sub(r'```\s*', '', cleaned)
    return cleaned.strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

def ats_match(resume_text, jd_text):
    """
    Uses Gemini to compare Resume vs JD and returns strict JSON.
    """
    prompt = f"""
    You are an advanced Applicant Tracking System (ATS) AI.
    
    Task: Compare the provided Resume against the Job Description.
    
    Resume Text (Truncated):
    {resume_text[:15000]} 

    Job Description:
    {jd_text[:5000]}

    Output Requirement:
    You MUST return the response in strict JSON format (NO MARKDOWN) with this structure:
    {{
        "match_percentage": <integer_0_to_100>,
        "matching_skills": ["skill1", "skill2", ...],
        "missing_skills": ["skill1", "skill2", ...],
        "formatting_issues": ["issue1", "issue2", ...],
        "summary_feedback": "<short_executive_summary_string>"
    }}
    """
    
    try:
        # Using Gemini Flash for speed
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
        
        # Clean potential markdown and parse
        cleaned_text = clean_json_string(response.text)
        return json.loads(cleaned_text)

    except Exception as e:
        print(f"AI Processing Error: {e}")
        # Return a fallback JSON structure so the frontend doesn't crash
        return {
            "match_percentage": 0,
            "matching_skills": [],
            "missing_skills": [f"Error: {str(e)}"],
            "formatting_issues": ["AI Analysis Failed"],
            "summary_feedback": "The system encountered an error while processing the document."
        }

# ==============================
# API ROUTES
# ==============================

@app.route('/')
def home():
    """
    Serves the J.A.R.V.I.S. Interface.
    Ensure 'ats_interface.html' is inside a folder named 'templates'.
    """
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    print("--- Incoming Analysis Request ---")
    
    # 1. Validation
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400
    
    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    if resume_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 2. Save & Extract
    try:
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(pdf_path)
        
        resume_text = extract_text_from_pdf(pdf_path)
        
        if not resume_text:
            return jsonify({"error": "Could not extract text from PDF. It might be empty or scanned images."}), 400

        # 3. AI Analysis
        analysis_result = ats_match(resume_text, jd_text)
        
        # 4. Cleanup (Delete uploaded file after processing)
        try:
            os.remove(pdf_path)
        except:
            pass

        return jsonify(analysis_result)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    # Ensure debug is True for detailed logs
    # Port 5000 is the default flask port
    app.run(debug=True, port=5000)
