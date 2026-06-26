# Intelligent Candidate Discovery - Asymmetric Pipeline

**Team:** Asymmetric Inference

---

# Overview

This repository contains our solution for the **Redrob AI Hackathon - Intelligent Candidate Discovery Challenge**.

The objective is to identify the **Top 10 most relevant candidates** from a dataset of 100,000 profiles for a given Engineering Job Description while satisfying strict execution constraints:

- CPU only
- No internet access during evaluation
- Less than 5 minutes runtime
- Explainable ranking
- Deterministic output

Our solution follows a **two-stage asymmetric architecture** that performs expensive semantic processing offline and executes lightweight ranking during evaluation.

---

# Architecture

## Phase 1 — Offline Pre-computation (`precompute.py`)

This phase is executed once before submission.

Tasks performed:

- Parse the Job Description (`JD.docx`)
- Parse candidate profiles (`candidates.jsonl`)
- Extract semantic embeddings using a local SentenceTransformer model
- Compute normalized candidate vectors
- Extract structured profile features
- Apply deterministic filtering heuristics for:
  - non-engineering profiles
  - keyword stuffing
  - chronological inconsistencies
  - malformed resumes
- Compute behavioral engagement score
- Store all processed information into a serialized artifact

Output:

```
artifacts.pkl
```

This preprocessing significantly reduces runtime during evaluation.

---

## Phase 2 — Online Ranking (`rank.py`)

During evaluation:

- Internet access is disabled
- CPU-only execution
- Loads `artifacts.pkl`
- Computes semantic similarity
- Applies behavioral scoring
- Applies deterministic tie-breaking
- Generates submission CSV
- Produces grounded reasoning strings

Average runtime:

```
< 1 second
```

on a modern CPU.

---

# Repository Structure

```
.
├── precompute.py
├── rank.py
├── app.py
├── requirements.txt
├── README.md
├── artifacts.pkl
├── candidates.jsonl
├── JD.docx
└── submission.csv
```

---

# Methodology

Each candidate receives a final ranking score based on multiple components.

## 1. Semantic Similarity

Sentence embeddings are generated using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Cosine similarity between the Job Description and candidate profile forms the primary ranking signal.

---

## 2. Engineering Profile Filtering

The system removes candidates that match known non-engineering patterns such as:

- HR
- Sales
- Marketing
- Recruiter
- Business Development
- Customer Support
- Finance-only profiles

This prevents keyword stuffing attacks.

---

## 3. Resume Consistency Checks

Additional deterministic heuristics detect:

- impossible timelines
- contradictory employment history
- malformed profiles
- suspicious keyword density

These profiles receive penalties or are filtered.

---

## 4. Behavioral Multiplier

Candidate engagement signals are incorporated through a deterministic multiplier derived from available structured metadata.

Examples include:

- response rate
- platform engagement
- activity indicators

---

## 5. Final Score

Final score combines:

- semantic similarity
- behavioral multiplier
- deterministic penalties

Candidates are sorted using:

```
(-score, candidate_id)
```

ensuring reproducible rankings.

---

# Explainability

Each ranked candidate includes a dynamically generated explanation describing:

- strongest skill match
- experience relevance
- technical alignment
- ranking rationale

Reasons are generated from actual profile attributes rather than static templates.

---

# Requirements

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate environment

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Pre-computation

This step is performed once before evaluation.

```bash
python precompute.py
```

Output:

```
artifacts.pkl
```

The SentenceTransformer model is downloaded only during this stage if not already cached.

---

# Evaluation

Run the ranking pipeline:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Output:

```
submission.csv
```

---

# Gradio Demo

Launch the demo application:

```bash
python app.py
```

The interface demonstrates:

- Job description loading
- Candidate ranking
- Explainable results
- Offline inference

---

# Runtime Characteristics

| Component | Runtime |
|------------|----------|
| Pre-computation | One-time offline step |
| Ranking | Typically < 1 second |
| Hardware | CPU only |
| Internet | Not required |
| Deterministic | Yes |

---

# Compliance

This implementation satisfies the challenge requirements by:

- CPU-only inference
- Offline execution
- No network dependency during ranking
- Deterministic ranking
- Explainable candidate selection
- Reproducible outputs
- Pre-computation documented
- GitHub reproducibility instructions included

---

# Assumptions

- Candidate dataset follows the provided JSONL schema.
- Job description is provided as a DOCX file.
- All preprocessing is completed before evaluation.
- The serialized artifact (`artifacts.pkl`) is included with the repository.

---

# Reproducing the Submission

## Step 1

(Optional)

Run preprocessing:

```bash
python precompute.py
```

---

## Step 2

Generate submission:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## Output

The generated CSV contains:

- candidate_id
- score
- rank
- explanation

ordered by descending relevance.

---

# Notes

- The online ranking stage performs no model downloads.
- All expensive embedding computation occurs during offline preprocessing.
- The repository is designed for deterministic, reproducible execution under the hackathon evaluation constraints.

---

# License

This repository is submitted solely for participation in the Redrob AI Hackathon and is intended for evaluation purposes.
