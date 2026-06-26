import json
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from docx import Document

# 1. Configuration
CANDIDATES_FILE = 'candidates.jsonl'
JD_FILE = 'job_description.docx'  # UPDATED TO EXPECT .docx
OUTPUT_ARTIFACT = 'artifacts.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2' 

def extract_text_from_docx(file_path):
    """
    Intelligently extracts text from a Word document, preserving basic structural flow
    which is critical for accurate semantic embeddings.
    """
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read {file_path}. Is it a valid .docx file?")
        print(f"Error details: {e}")
        exit(1)

print(f"Loading local embedding model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# 2. Extract JD Embedding (Using the new .docx parser)
print(f"Parsing Semantic Ground Truth from {JD_FILE}...")
jd_text = extract_text_from_docx(JD_FILE)
# We take the first 1500 words to ensure we capture the core requirements 
# without diluting the embedding with boilerplate company descriptions at the end.
jd_embedding = model.encode(jd_text[:1500], normalize_embeddings=True)

# 3. Process Candidates
candidate_ids = []
embeddings = []
behavioral_multipliers = []
honeypot_flags = []
candidate_metadata = {} 

print("Processing 100,000 Candidates (Extracting, Flagging, and Embedding)...")
with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=100000):
        if not line.strip(): continue
        cand = json.loads(line)
        
        cid = cand['candidate_id']
        prof = cand.get('profile', {})
        history = cand.get('career_history', [])
        skills = cand.get('skills', [])
        signals = cand.get('redrob_signals', {})

        # --- A. HONEYPOT DESTROYER HEURISTICS ---
        is_honeypot = False
        
        total_work_months = sum(h.get('duration_months', 0) for h in history)
        claimed_exp_months = prof.get('years_of_experience', 0) * 12
        if total_work_months > (claimed_exp_months + 24):
            is_honeypot = True
            
        for s in skills:
            if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
                is_honeypot = True
                break
                
        if signals.get('recruiter_response_rate', 1.0) < 0.05 and signals.get('applications_submitted_30d', 1) == 0:
            is_honeypot = True

        # --- B. BEHAVIORAL MULTIPLIER SCORING ---
        resp_rate = signals.get('recruiter_response_rate', 0.5)
        int_completion = signals.get('interview_completion_rate', 0.5)
        open_work = 1.0 if signals.get('open_to_work_flag', False) else 0.8
        
        multiplier = (resp_rate * 0.4) + (int_completion * 0.4) + (open_work * 0.2)
        
        # --- C. SEMANTIC TEXT EXTRACTION ---
        top_skills = [s['name'] for s in sorted(skills, key=lambda x: x.get('duration_months', 0), reverse=True)[:5]]
        text_to_embed = f"Title: {prof.get('current_title', '')}. Experience: {prof.get('years_of_experience', 0)} years. Skills: {', '.join(top_skills)}. Summary: {prof.get('summary', '')}"
        
        # --- D. APPEND TO ARRAYS ---
        candidate_ids.append(cid)
        embeddings.append(text_to_embed)
        behavioral_multipliers.append(multiplier)
        honeypot_flags.append(is_honeypot)
        candidate_metadata[cid] = {
            "title": prof.get('current_title', 'Professional'),
            "exp": prof.get('years_of_experience', 0),
            "skills": top_skills,
            "resp_rate": resp_rate
        }

# 4. Batch Embed All Candidates
print("Running Neural Network forward pass for candidate embeddings...")
cand_embeddings_matrix = model.encode(embeddings, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

# 5. Save Artifacts to Disk
print("Saving compressed artifacts for Phase 2...")
artifacts = {
    'jd_embedding': jd_embedding,
    'candidate_ids': candidate_ids,
    'candidate_embeddings': cand_embeddings_matrix,
    'multipliers': np.array(behavioral_multipliers, dtype=np.float32),
    'honeypot_flags': np.array(honeypot_flags, dtype=bool),
    'metadata': candidate_metadata
}

with open(OUTPUT_ARTIFACT, 'wb') as f:
    pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Phase 1 Complete! Artifacts saved to {OUTPUT_ARTIFACT}.")