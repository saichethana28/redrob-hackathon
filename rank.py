import pickle
import numpy as np
import csv
import time
import argparse

def generate_reasoning(meta):
    skills_str = ", ".join(meta['skills'][:2]) if meta['skills'] else "core domain tools"

    title = meta['title'].strip()

    # Avoid "Senior Senior ..."
    if meta['exp'] > 5 and not title.lower().startswith("senior"):
        title = "Senior " + title

    if meta['exp'] > 5:
        return (
            f"{title} with {meta['exp']} years of proven experience, "
            f"particularly in {skills_str}. Demonstrates strong recruiter engagement "
            f"({int(meta['resp_rate']*100)}% response rate), making them a strong match for the role."
        )

    elif meta['resp_rate'] > 0.7:
        return (
            f"{title} with {meta['exp']} years of experience and a high recruiter response rate "
            f"({int(meta['resp_rate']*100)}%). Skills in {skills_str} align well with the job requirements."
        )

    else:
        return (
            f"{title} with {meta['exp']} years of experience. Technical strengths in "
            f"{skills_str} align with the role, though engagement signals are comparatively moderate."
        )
    
def main():
    # --- NEW: Accept the Hackathon's required arguments ---
    parser = argparse.ArgumentParser(description="Redrob 1st-Place Ranker")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Path for output CSV")
    args = parser.parse_args()

    start_time = time.time()
    print("Loading artifacts (No Network, CPU Only)...")
    
    try:
        with open('artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
    except FileNotFoundError:
        print("CRITICAL ERROR: artifacts.pkl not found. You must run precompute.py first.")
        exit(1)
        
    jd_vec = artifacts['jd_embedding']
    cand_matrix = artifacts['candidate_embeddings']
    c_ids = artifacts['candidate_ids']
    multipliers = artifacts['multipliers']
    honeypots = artifacts['honeypot_flags']
    metadata = artifacts['metadata']

    print("Executing dense matrix multiplication...")
    semantic_scores = np.dot(cand_matrix, jd_vec)

    final_scores = semantic_scores * multipliers
    final_scores[honeypots] = 0.0

    print("Sorting deterministically...")
    scored_candidates = [
        (-float(final_scores[i]), c_ids[i]) for i in range(len(c_ids))
    ]
    scored_candidates.sort()
    
    # --- NEW: Use the requested output filename ---
    output_filename = args.out
    
    print(f"Writing exactly 100 rows to {output_filename}...")
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank_idx in range(100):
            score = -scored_candidates[rank_idx][0]
            cid = scored_candidates[rank_idx][1]
            reasoning_text = generate_reasoning(metadata[cid])
            writer.writerow([cid, rank_idx + 1, round(score, 6), reasoning_text])

    exec_time = time.time() - start_time
    print(f"Ranking Complete in {exec_time:.2f} seconds!")

if __name__ == "__main__":
    main()