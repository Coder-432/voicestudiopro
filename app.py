import os
import tempfile
import logging
import streamlit as st
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Voice Studio Pro | Lavender & Pink Suite",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS (Removing dark header strip & updating theme styling)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global App Theme - Deep Midnight Plum to Soft Rose-Violet Gradient */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #2e1065 0%, #170f2c 45%, #0b0714 100%);
        color: #f3e8ff;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Completely remove default Streamlit top header / decoration bar color mismatches */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        letter-spacing: -0.03em;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(23, 15, 44, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(216, 180, 254, 0.15);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Modern Glass Container Cards with Lavender/Pink Glow */
    .app-card {
        background: linear-gradient(135deg, rgba(88, 28, 135, 0.25) 0%, rgba(157, 78, 221, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(232, 121, 249, 0.2);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 15px 35px -10px rgba(88, 28, 135, 0.4);
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }
    .app-card:hover {
        border-color: rgba(244, 114, 182, 0.4);
        box-shadow: 0 20px 40px -10px rgba(244, 114, 182, 0.2);
    }
    
    /* Lavender & Soft Pink Gradient Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
        color: #ffffff;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 0.65rem 1.5rem;
        box-shadow: 0 4px 20px rgba(236, 72, 153, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #c084fc 0%, #f472b6 100%);
        box-shadow: 0 6px 25px rgba(236, 72, 153, 0.6);
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    /* Inputs & Selectboxes */
    div[data-baseweb="select"] > div, .stTextInput input {
        background-color: rgba(35, 20, 64, 0.8) !important;
        color: #f3e8ff !important;
        border: 1px solid rgba(216, 180, 254, 0.2) !important;
        border-radius: 12px !important;
        padding: 4px 12px !important;
    }
    div[data-baseweb="select"] > div:hover, .stTextInput input:focus {
        border-color: #f472b6 !important;
        box-shadow: 0 0 0 2px rgba(244, 114, 182, 0.3) !important;
    }
    
    /* Text Areas */
    textarea {
        background-color: rgba(25, 14, 48, 0.7) !important;
        color: #f3e8ff !important;
        border: 1px solid rgba(216, 180, 254, 0.2) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        font-size: 0.95rem !important;
    }
    textarea:focus {
        border-color: #ec4899 !important;
        box-shadow: 0 0 0 2px rgba(236, 72, 153, 0.3) !important;
    }
    
    /* Gradient Accent Headers */
    .accent-header {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #f472b6 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    /* SVG Symbol Styling */
    .svg-icon {
        width: 24px;
        height: 24px;
        vertical-align: middle;
        fill: #f472b6;
        margin-right: 10px;
    }
    /* Large & Uniform Image Gallery Cards */
    .gallery-img {
        width: 100%;
        height: 300px;
        object-fit: cover;
        border-radius: 20px;
        border: 1px solid rgba(232, 121, 249, 0.3);
        box-shadow: 0 15px 35px rgba(88, 28, 135, 0.4);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_TRANSCRIPT_LENGTH = 5000
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac"}

def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@st.cache_resource(show_spinner=False)
def load_whisper_model() -> whisper.Whisper:
    logger.info("Initializing Whisper transcription engine...")
    return whisper.load_model("base")

@st.cache_resource(show_spinner=False)
def load_text_engine():
    logger.info("Initializing Qwen text engine...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

# Model Loading Initialization with Spinner
with st.spinner("Initializing local neural runtimes..."):
    try:
        whisper_model = load_whisper_model()
        text_tokenizer, text_model = load_text_engine()
    except Exception as error:
        st.error(f"Initialization Error: {error}")
        logger.exception("Failed to initialize models.")
        st.stop()

# Website Navigation Bar
selected_tab = st.sidebar.radio(
    "Navigation", 
    ["Home / Overview", "Voice Studio App"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.markdown("<p style='color: #4ade80; font-size: 0.85rem; font-weight: 600;'>&#9679; Engines Online</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #d8b4fe; font-size: 0.8rem;'>Ready for secure transcription</p>", unsafe_allow_html=True)


# ==========================================
# PAGE 1: HOME / OVERVIEW
# ==========================================
if selected_tab == "Home / Overview":
    st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>Voice Studio Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #e9d5ff; font-size: 1.2rem; margin-bottom: 2rem;'>Transform your spoken ideas, dictations, and meetings into polished professional text instantly, right from your desktop.</p>", unsafe_allow_html=True)

    img_col1, img_col2, img_col3 = st.columns(3, gap="medium")
    with img_col1:
        st.markdown('<img class="gallery-img" src="https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?q=80&w=1000&auto=format&fit=crop" alt="Audio Mixing Console">', unsafe_allow_html=True)
    with img_col2:
        st.markdown('<img class="gallery-img" src="https://images.unsplash.com/photo-1590602847861-f357a9332bbc?q=80&w=1000&auto=format&fit=crop" alt="Recording Studio">', unsafe_allow_html=True)
    with img_col3:
        st.markdown('<img class="gallery-img" src="https://images.unsplash.com/photo-1531482615713-2afd69097998?q=80&w=1000&auto=format&fit=crop" alt="Team Collaboration">', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='accent-header'>Core Studio Features</h2>", unsafe_allow_html=True)

    feat_col1, feat_col2 = st.columns(2, gap="large")
    with feat_col1:
        st.markdown("""
            <div class="app-card">
                <h3>
                    <svg class="svg-icon" viewBox="0 0 24 24"><path d="M12 3c-4.97 0-9 4.03-9 9 0 2.12.74 4.07 1.97 5.61L4.35 19.35c-.39.39-.39 1.02 0 1.41.39.39 1.02.39 1.41 0l2.35-2.35C9.65 18.96 10.79 19.2 12 19.2c4.97 0 9-4.03 9-9s-4.03-9-9-9zm0 15c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6-6 6z"/></svg>
                    Tone Conversion
                </h3>
                <p style='color: #d8b4fe; font-size: 0.95rem;'>Instantly adapt the mood and style of your voice notes. Switch between formal professional emails, polite diplomatic correspondence, concise directions, or casual team messaging with a single click.</p>
            </div>
            <div class="app-card">
                <h3>
                    <svg class="svg-icon" viewBox="0 0 24 24"><path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/></svg>
                    Translation
                </h3>
                <p style='color: #d8b4fe; font-size: 0.95rem;'>Speak naturally in your native language and dynamically translate your speech into French, German, Spanish, Italian, Japanese, and more on the fly.</p>
            </div>
            <div class="app-card">
                <h3>
                    <svg class="svg-icon" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 10h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"/></svg>
                    Meeting Summaries
                </h3>
                <p style='color: #d8b4fe; font-size: 0.95rem;'>Turn long, rambling meeting recordings or brainstorming sessions into crisp executive summaries and clear bullet-point action items.</p>
            </div>
        """, unsafe_allow_html=True)

    with feat_col2:
        st.markdown("""
            <div class="app-card">
                <h3>
                    <svg class="svg-icon" viewBox="0 0 24 24"><path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/></svg>
                    Grammar Check
                </h3>
                <p style='color: #d8b4fe; font-size: 0.95rem;'>Automatically eliminate filler words like 'um' and 'uh', correct stutters, fix awkward phrasing, and polish grammatical errors without changing your core message.</p>
            </div>
            <div class="app-card">
                <h3>
                    <svg class="svg-icon" viewBox="0 0 24 24"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z"/></svg>
                    100% Privacy
                </h3>
                <p style='color: #d8b4fe; font-size: 0.95rem;'>Your voice data and recordings never leave your local computer. There are no cloud servers, no third-party APIs, no telemetry, and zero tracking.</p>
            </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: VOICE STUDIO APP
# ==========================================
elif selected_tab == "Voice Studio App":
    st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 0.2rem;'>Studio Interface</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #e9d5ff; font-size: 1.1rem; margin-bottom: 2rem;'>Record your voice or upload an audio file to process.</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Studio Controls")
        preset_style = st.selectbox(
            "Target Style or Language",
            [
                "Professional Email",
                "Polite and Diplomatic",
                "Concise and Direct",
                "Slack or Teams Message",
                "Bullet-Point Action Items",
                "Grammar and Spelling Correction Only",
                "Translate to Italian",
                "Translate to Spanish",
                "Translate to French",
                "Translate to German",
                "Executive Summary"
            ]
        )
        custom_style = st.text_input("Custom Style Override", placeholder="e.g., Translate to Japanese")
        st.markdown("---")
        input_method = st.radio("Capture Mode", ["Microphone Recording", "Audio File Upload"])

    audio_bytes = None
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("<div class='accent-header'>Audio Stream Input</div>", unsafe_allow_html=True)

    if input_method == "Microphone Recording":
        audio_data = st.audio_input("Record audio stream from your hardware device")
        if audio_data is not None:
            audio_bytes = audio_data.getvalue()
    else:
        uploaded_file = st.file_uploader(
            "Upload audio container asset",
            type=list(ALLOWED_EXTENSIONS),
            help="Maximum file size limit: 25 MB (Supported formats: wav, mp3, m4a, ogg, flac)"
        )
        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error("The uploaded file exceeds the 25 MB security limit.")
            elif not is_allowed_file(uploaded_file.name):
                st.error("Invalid file format detected. Please upload a standard audio file.")
            else:
                audio_bytes = uploaded_file.getvalue()
    st.markdown('</div>', unsafe_allow_html=True)

    # Process audio bytes
    if audio_bytes is not None:
        selected_style = custom_style.strip() if custom_style.strip() else preset_style
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name

            with st.spinner("Processing local acoustic extraction and text synthesis..."):
                try:
                    transcription_result = whisper_model.transcribe(temp_path)
                    raw_transcript = transcription_result.get("text", "").strip()
                except Exception as decode_err:
                    logger.error(f"Acoustic decoding failed: {decode_err}")
                    st.error("Could not parse the audio file. It may be corrupted or in an unsupported format.")
                    st.stop()

            if not raw_transcript:
                st.warning("No intelligible acoustic patterns detected.")
            else:
                if len(raw_transcript) > MAX_TRANSCRIPT_LENGTH:
                    raw_transcript = raw_transcript[:MAX_TRANSCRIPT_LENGTH]

                messages = [
                    {
                        "role": "system", 
                        "content": (
                            "You are an expert writer and editor. Rewrite, translate, or format "
                            "the provided text strictly according to the requested style constraint. "
                            "Ensure it reads naturally and professionally. Return exclusively the final text."
                        )
                    },
                    {
                        "role": "user", 
                        "content": f"Requested Style or Target Language: {selected_style}\n\nSource Text:\n{raw_transcript}"
                    }
                ]

                prompt = text_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

                inputs = text_tokenizer(prompt, return_tensors="pt").to(text_model.device)
                
                with torch.no_grad():
                    outputs = text_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.4,
                        top_p=0.9,
                        do_sample=True
                    )
                
                decoded_output = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
                transformed_text = decoded_output.split("assistant\n")[-1].strip()

                st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2, gap="large")
                
                con_icon = '<svg class="svg-icon" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg>'
                
                with col1:
                    st.markdown('<div class="app-card">', unsafe_allow_html=True)
                    st.markdown(f"<div class='accent-header' style='font-size: 1.2rem;'>{con_icon} Raw Transcription</div>", unsafe_allow_html=True)
                    st.text_area("Whisper Output", value=raw_transcript, height=260, disabled=True, label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="app-card">', unsafe_allow_html=True)
                    st.markdown(f"<div class='accent-header' style='font-size: 1.2rem;'>{con_icon} Transformed Output ({selected_style})</div>", unsafe_allow_html=True)
                    st.text_area("Qwen Output", value=transformed_text, height=260, label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)

        except Exception as err:
            logger.error(f"Execution error: {err}")
            st.error(f"An error occurred during execution: {err}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
