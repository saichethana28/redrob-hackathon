import gradio as gr
import subprocess
import os

def execute_pipeline():
    # Run the rank.py script
    result = subprocess.run(["python", "rank.py"], capture_output=True, text=True)
    log_output = result.stdout
    
    file_path = "asymmetric_inference.csv"
    if os.path.exists(file_path):
        return log_output, file_path
    else:
        return log_output + "\n\nError: CSV not generated.", None

with gr.Blocks() as demo:
    gr.Markdown("# 🏆 Asymmetric Inference Ranker")
    gr.Markdown("Executing `rank.py` using pre-computed artifacts (`artifacts.pkl`). CPU-only, 0 network.")
    
    run_btn = gr.Button("Execute rank.py", variant="primary")
    
    with gr.Row():
        logs = gr.Textbox(label="Terminal Output", lines=8)
        download = gr.File(label="Download Validated CSV")
        
    run_btn.click(fn=execute_pipeline, inputs=[], outputs=[logs, download])

demo.launch()