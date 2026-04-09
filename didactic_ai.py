import os
import json
import textwrap

import streamlit as st
import PyPDF2
import google.generativeai as genai

# Read GEMINI_API_KEY from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your system environment first.")

genai.configure(api_key=GEMINI_API_KEY)


def _pick_default_model() -> str:
    """
    Automatically pick a model that supports generateContent from the available models in the current account.
    This prevents 404 errors due to specific model names (e.g., 1.5-flash-002) being unavailable in certain regions.
    """
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name
    except Exception:
        # If listing models fails, fall back to a generic alias
        return "gemini-2.0-flash"

    # Final fallback if none are found
    return "gemini-2.0-flash"


MODEL_NAME = _pick_default_model()
model = genai.GenerativeModel(MODEL_NAME)


# ---------- Utility Functions ----------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(uploaded_file)
    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        texts.append(page_text)
    full_text = "\n".join(texts)
    # Light cleanup
    return " ".join(full_text.split())


def build_prompt(document_text: str) -> str:
    """Create a prompt for the Gemini API."""
    # To control cost, truncate extremely long PDFs
    max_chars = 9000
    doc_snippet = document_text[:max_chars]

    prompt = f"""
You are an expert instructional designer at "Didactic Innovations".

You are given content from a PDF document. Read it and:

1. Distill it into exactly 5 concise, practical "Action Steps" for a motivated learner.
   - Each action step should be a single, clear sentence.
   - Focus on what the learner should DO after reading the PDF.

2. Create exactly 3 multiple-choice questions to check understanding.
   - Each question should have 4 answer options.
   - Clearly mark which option is correct.

Return your answer as valid JSON only, with this exact structure:

{{
  "action_steps": [
    "Action step 1",
    "Action step 2",
    "Action step 3",
    "Action step 4",
    "Action step 5"
  ],
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_option_index": 1
    }},
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_option_index": 2
    }},
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_option_index": 0
    }}
  ]
}}

The field names and structure must match exactly and there should be no extra fields.
Here is the document content:

\"\"\"{doc_snippet}\"\"\"
"""
    return textwrap.dedent(prompt).strip()


def get_actions_and_questions(document_text: str):
    """Call Gemini to get 5 action steps and 3 MCQs."""
    prompt = build_prompt(document_text)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,
        ),
    )

    # Gemini response contains a text attribute
    content = response.text.strip()

    # Attempt to parse JSON with defensive handling
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = content[start:end]
            data = json.loads(json_str)
        else:
            raise ValueError("Model response did not contain valid JSON.")

    action_steps = data.get("action_steps", [])
    questions = data.get("questions", [])

    # Basic sanity trimming
    action_steps = [s.strip() for s in action_steps if isinstance(s, str) and s.strip()]
    cleaned_questions = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        question_text = str(q.get("question", "")).strip()
        options = q.get("options", [])
        correct_index = q.get("correct_option_index", 0)
        if question_text and isinstance(options, list) and len(options) == 4:
            cleaned_questions.append(
                {
                    "question": question_text,
                    "options": [str(o).strip() for o in options],
                    "correct_option_index": int(correct_index),
                }
            )

    return action_steps[:5], cleaned_questions[:3]


# ---------- Streamlit Page Config & Styling ----------
st.set_page_config(
    page_title="Didactic AI - PDF Coach",
    page_icon="📘",
    layout="wide",
)

# Custom CSS for a professional blue/white theme
st.markdown(
    """
    <style>
        /* Base background */
        .stApp {
            background-color: #f5f8ff;
        }

        /* Main title */
        h1, h2, h3 {
            color: #0b2e59;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b2e59 0%, #144a86 100%);
            color: #ffffff;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p {
            color: #ffffff !important;
        }

        /* Cards / containers */
        .analysis-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(11, 46, 89, 0.08);
        }

        .action-step {
            padding: 0.6rem 0.9rem;
            margin-bottom: 0.4rem;
            border-radius: 6px;
            background-color: #e3edff;
            color: #0b2e59;
            border-left: 4px solid #0b72e6;
            font-size: 0.96rem;
        }

        .mcq-block {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 6px;
            background-color: #ffffff;
            border: 1px solid #d5e3ff;
        }

        .correct {
            color: #0b8a3a;
            font-weight: 600;
        }

        .incorrect {
            color: #c0392b;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📘 Didactic AI")
    st.markdown(
        """
        **PDF Learning Coach**

        - Upload a PDF.
        - Get 5 practical action steps.
        - Test yourself with 3 MCQs.

        Built with **Streamlit** + **Gemini**, inspired by *Didactic Innovations*.
        """
    )

    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown(
        """
        - Use focused, topic-based PDFs.\n
        - Avoid scanned images without text.\n
        - Longer PDFs may be truncated for speed & cost.
        """
    )


# ---------- Main Layout ----------
st.title("Didactic AI – PDF Action Planner")

st.markdown(
    "Transform any PDF into **practical action steps** and quick **knowledge checks**."
)

uploaded_pdf = st.file_uploader(
    "Upload a PDF to begin",
    type=["pdf"],
    help="Drop a PDF file here or browse your computer.",
)

if "action_steps" not in st.session_state:
    st.session_state.action_steps = None
if "questions" not in st.session_state:
    st.session_state.questions = None
if "answers_checked" not in st.session_state:
    st.session_state.answers_checked = False

analyze_button_disabled = uploaded_pdf is None

col_left, col_right = st.columns([2, 3])

with col_left:
    if st.button("Analyze PDF with Gemini", disabled=analyze_button_disabled):
        if uploaded_pdf is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Extracting text and calling Gemini..."):
                try:
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    if not pdf_text.strip():
                        st.error(
                            "No readable text was found in this PDF. "
                            "It might be a scanned image without embedded text."
                        )
                    else:
                        action_steps, questions = get_actions_and_questions(pdf_text)
                        if not action_steps:
                            st.error("Gemini did not return any action steps.")
                        else:
                            st.session_state.action_steps = action_steps
                            st.session_state.questions = questions
                            st.session_state.answers_checked = False
                            st.success("Analysis complete! Scroll down for results.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

with col_right:
    if uploaded_pdf is not None:
        st.info(f"**Selected file:** {uploaded_pdf.name}")
    else:
        st.info("Upload a PDF and click **Analyze PDF with Gemini**.")

st.markdown("---")

# ---------- Results Section ----------
if st.session_state.action_steps:
    st.subheader("📌 Action Steps")

    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    for idx, step in enumerate(st.session_state.action_steps, start=1):
        st.markdown(
            f'<div class="action-step"><strong>Step {idx}.</strong> {step}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.questions:
    st.subheader("🧠 Knowledge Check – 3 Questions")

    questions = st.session_state.questions
    for idx, q in enumerate(questions, start=1):
        st.markdown('<div class="mcq-block">', unsafe_allow_html=True)
        st.markdown(f"**Q{idx}. {q['question']}**")

        # Unique key for each question's radio
        answer_key = f"q_{idx}_answer"
        st.radio(
            "Select one:",
            options=list(range(len(q["options"]))),
            format_func=lambda i, opts=q["options"]: opts[i],
            key=answer_key,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Check my answers"):
        st.session_state.answers_checked = True

    if st.session_state.answers_checked:
        st.markdown("#### ✅ Feedback")
        for idx, q in enumerate(questions, start=1):
            chosen = st.session_state.get(f"q_{idx}_answer", None)
            correct = q["correct_option_index"]

            if chosen is None:
                st.markdown(
                    f"- Q{idx}: <span class='incorrect'>No answer selected.</span>",
                    unsafe_allow_html=True,
                )
            elif chosen == correct:
                st.markdown(
                    f"- Q{idx}: <span class='correct'>Correct!</span>",
                    unsafe_allow_html=True,
                )
            else:
                correct_text = q["options"][correct]
                st.markdown(
                    f"- Q{idx}: <span class='incorrect'>Not quite.</span> "
                    f"The correct answer was **{correct_text}**.",
                    unsafe_allow_html=True,
                )

elif uploaded_pdf and not st.session_state.action_steps:
    st.caption(
        "Upload a PDF and click **Analyze PDF with Gemini** to see results here."
    )
