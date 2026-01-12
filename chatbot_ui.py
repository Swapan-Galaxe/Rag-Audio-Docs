import gradio as gr
from rag_summarizer import RAGSummarizer
from pathlib import Path
import time

summarizer = RAGSummarizer()
processed = False
file_list = []

def scan_files():
    """Scan and list available files"""
    script_dir = Path(__file__).parent
    doc_dir = script_dir / "data" / "documents"
    audio_dir = script_dir / "data" / "audio"
    
    docs = list(doc_dir.glob("*.txt")) + list(doc_dir.glob("*.pdf"))
    audios = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
    
    if not docs and not audios:
        return [], "❌ No files found in data/ directory", gr.update(visible=False)
    
    all_files = [(str(f), f.name) for f in docs + audios]
    return (
        gr.update(choices=[f[1] for f in all_files], value=[], visible=True),
        f"📁 Found {len(docs)} documents and {len(audios)} audio files",
        gr.update(visible=True)
    )

def load_files(selected_files, progress=gr.Progress()):
    """Load selected files"""
    global processed, file_list
    
    if not selected_files:
        return "⚠️ Please select files to load", "", gr.update(visible=False), gr.update(visible=False)
    
    script_dir = Path(__file__).parent
    doc_dir = script_dir / "data" / "documents"
    audio_dir = script_dir / "data" / "audio"
    
    doc_paths = []
    audio_paths = []
    
    progress(0, desc="📂 Scanning selected files...")
    time.sleep(0.5)
    
    for filename in selected_files:
        doc_path = doc_dir / filename
        audio_path = audio_dir / filename
        
        if doc_path.exists():
            doc_paths.append(str(doc_path))
        elif audio_path.exists():
            audio_paths.append(str(audio_path))
    
    progress(0.1, desc="📄 Loading documents...")
    time.sleep(0.5)
    
    progress(0.3, desc="🎵 Transcribing audio (this may take 30-60s per file)..." if audio_paths else "📝 Processing documents...")
    
    progress(0.5, desc="✂️ Splitting into chunks...")
    
    progress(0.7, desc="🔢 Creating embeddings (calling OpenAI API)...")
    
    summarizer.process_files(
        doc_paths if doc_paths else None,
        audio_paths if audio_paths else None
    )
    
    progress(0.9, desc="💾 Storing in vector database...")
    time.sleep(0.5)
    
    processed = True
    file_list = selected_files
    progress(1.0, desc="✅ Complete!")
    
    file_display = "\n".join([f"• {f}" for f in file_list])
    return (
        f"✅ Successfully loaded {len(selected_files)} files",
        file_display,
        gr.update(visible=True),
        gr.update(visible=True)
    )

def get_summary(progress=gr.Progress()):
    """Get comprehensive summary"""
    if not processed:
        return "⚠️ Please load files first"
    
    progress(0, desc="🔍 Retrieving relevant chunks...")
    time.sleep(0.5)
    progress(0.3, desc="🤖 Calling GPT-3.5 API...")
    time.sleep(0.5)
    progress(0.6, desc="✍️ Generating summary...")
    summary = summarizer.custom_summary("Provide a comprehensive summary covering all topics from all documents and audio")
    progress(1.0, desc="✅ Complete!")
    return summary

def chat(message, history):
    """Chat function with typing effect"""
    if not processed:
        return "⚠️ Please load files first using the 'Load Files' button"
    
    response = summarizer.query(message)
    return response

def clear_all():
    """Clear all data"""
    global processed, file_list
    processed = False
    file_list = []
    return "", "", "", gr.update(visible=False, value=[]), gr.update(visible=False), gr.update(visible=False)

# Create Gradio interface with custom theme
with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown(
        """
        # 🤖 RAG Audio & Document Chatbot
        ### Ask questions about your documents and audio files
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 File Management")
            scan_btn = gr.Button("🔍 Scan Files", variant="secondary", size="lg")
            
            file_selector = gr.CheckboxGroup(
                label="Select files to load",
                choices=[],
                visible=False
            )
            
            load_btn = gr.Button("📥 Load Selected Files", variant="primary", size="lg", visible=False)
            clear_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
            
            status = gr.Textbox(label="Status", interactive=False, show_label=False)
            
            with gr.Accordion("📄 Loaded Files", open=False) as files_accordion:
                files_display = gr.Textbox(label="Files", interactive=False, show_label=False, lines=5)
            
            with gr.Accordion("📊 Summary", open=False, visible=False) as summary_accordion:
                summary_btn = gr.Button("✨ Generate Summary", variant="secondary")
                summary_output = gr.Textbox(label="", lines=15, interactive=False, show_label=False)
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with Your Documents")
            chatbot_interface = gr.ChatInterface(
                chat,
                examples=[
                    "What are the main topics discussed?",
                    "Explain the key concepts in detail",
                    "What are the important findings?",
                    "Summarize the conclusions",
                    "What insights can you provide?"
                ],
                title="",
            )
    
    gr.Markdown(
        """
        ---
        💡 **Tips:** Scan files → Select files → Load → Ask questions or generate summary
        """
    )
    
    # Event handlers
    scan_btn.click(
        scan_files,
        outputs=[file_selector, status, load_btn]
    )
    
    load_btn.click(
        load_files,
        inputs=[file_selector],
        outputs=[status, files_display, files_accordion, summary_accordion]
    )
    
    summary_btn.click(
        get_summary,
        outputs=summary_output
    )
    
    clear_btn.click(
        clear_all,
        outputs=[status, files_display, summary_output, file_selector, files_accordion, summary_accordion]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), inbrowser=True, share=False)
