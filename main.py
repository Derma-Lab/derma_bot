import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Azure OpenAI API key and endpoint
azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = "2024-07-01-preview"

# Set up Azure OpenAI client
client = AzureOpenAI(
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
)

# Set page config
st.set_page_config(
    page_title="Dermatology Chat Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial welcome message
    welcome_msg = """👋 Welcome to our Dermatology Assistant! 
    
I'm here to help you with your skin concerns. Please feel free to describe your condition or symptoms in detail. 
    
You'll be guided through a consultation process with:
🏥 Our Reception Team
👨‍⚕️ A Dermatologist
💊 A Pharmacist

How can we help you today?"""
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

if "current_step" not in st.session_state:
    st.session_state.current_step = "initial"
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None

# Display chat title
st.title("Dermatology Assistant 🩺")
st.markdown("by [Derma Lab](https://github.com/Derma-Lab)")

# Define agent prompts
GENERAL_MANAGER_PROMPT = """You are a friendly and caring receptionist at a dermatology clinic.
Gather patient information with empathy and understanding.
Use a warm, welcoming tone while collecting necessary details."""

DOCTOR_PROMPT = """You are an experienced dermatologist with a caring bedside manner.
Analyze the patient's symptoms and provide:
🔍 Detailed symptom analysis
📋 Potential diagnosis
⚖️ Confidence level
Keep explanations clear and reassuring."""

PHARMACIST_PROMPT = """You are a knowledgeable and helpful pharmacist.
Provide medication advice in this format:
💊 RECOMMENDED MEDICATIONS:
1. [Medication Name]
   - Dosage: [amount]
   - Frequency: [times per day]
   - Duration: [days]

💰 ESTIMATED COSTS:
- Medication 1: $[amount]
- Total: $[total]

📝 USAGE INSTRUCTIONS:
1. [Specific instructions]

⚠️ PRECAUTIONS:
1. [Key precautions]"""

def get_ai_response(messages, system_prompt):
    """Get response from Azure OpenAI"""
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            *messages
        ],
        max_tokens=500,
        temperature=0,
    )
    return chat_completion.choices[0].message.content.strip()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interaction logic
if prompt := st.chat_input("Describe your skin concern..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # General Manager initial response
    if st.session_state.current_step == "initial":
        gm_questions = """Thank you for sharing that. To help you better, please provide:

👤 1. Your age and gender
⏱️ 2. How long you've had this condition
🎯 3. Where on your body is it located and how does it look
💊 4. Any treatments you've already tried
🏥 5. Your medical history and allergies (if any)"""
        
        with st.chat_message("assistant"):
            st.markdown(gm_questions)
            st.session_state.messages.append({"role": "assistant", "content": gm_questions})
            st.session_state.current_step = "doctor"
    
    # Doctor's diagnosis
    elif st.session_state.current_step == "doctor":
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        diagnosis = get_ai_response(messages, DOCTOR_PROMPT)
        
        with st.chat_message("assistant"):
            st.markdown(diagnosis)
            st.session_state.messages.append({"role": "assistant", "content": diagnosis})
            st.session_state.diagnosis = diagnosis
            st.session_state.current_step = "pharmacist"
    
    # Pharmacist's prescription
    elif st.session_state.current_step == "pharmacist":
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        prescription = get_ai_response(messages, PHARMACIST_PROMPT)
        
        with st.chat_message("assistant"):
            st.markdown(prescription)
            st.session_state.messages.append({"role": "assistant", "content": prescription})
            st.session_state.current_step = "complete"

# Add a sidebar with additional information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a dermatology chat assistant that can help you with:
    - Initial symptom assessment
    - Basic skin care advice
    - General dermatological information
    
    Please note: This is not a replacement for professional medical advice.
    """)
    
    # Add consultation progress tracker
    st.title("Consultation Progress")
    steps = ["Initial Assessment", "Doctor's Diagnosis", "Prescription & Medicines"]
    current_step_idx = {"initial": 0, "doctor": 1, "pharmacist": 2, "complete": 3}
    current_idx = current_step_idx.get(st.session_state.current_step, 0)
    
    for idx, step in enumerate(steps):
        if idx < current_idx:
            st.success(step)
        elif idx == current_idx:
            st.info(step)
        else:
            st.empty()