# app.py
import streamlit as st
import json
from typing import TypedDict, Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = None
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
if 'patient_sold' not in st.session_state:
    st.session_state.patient_sold = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Define available medicines and conditions
AVAILABLE_MEDICINES = {
    "hair loss": ["monoxodil"],
    "acne": ["acnecontrol cream", "acne treatment"],
    "rosacea": ["rosacearelief lotion", "rosacea cream"],
    "mole": ["molefade serum"],
    "rash": ["rashguard ointment", "rash treatment"],
    "eczema": ["eczema cream"],
    "psoriasis": ["psoriasis treatment"]
}

def initialize_azure_client():
    load_dotenv()
    required_vars = ["AZURE_OAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return None

    try:
        return AzureChatOpenAI(
            api_key=os.getenv("AZURE_OAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0.7
        )
    except Exception as e:
        st.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        return None

client = initialize_azure_client()

def check_medicine_availability(diagnosis: str, medicines: str) -> Tuple[bool, str]:
    """Check if we have appropriate medicines for the condition"""
    diagnosis = diagnosis.lower()
    medicines = medicines.lower()
    
    # Check if any of our supported conditions are mentioned in diagnosis
    available_condition = False
    for condition in AVAILABLE_MEDICINES.keys():
        if condition in diagnosis:
            available_condition = True
            break
            
    if not available_condition:
        return False, "We don't have specific treatments for this condition."
    
    return True, "Medicine available"

def checker_agent(last_agent_message: str, last_patient_message: str) -> bool:
    checker_messages = [
        SystemMessage(content=(
            "You are a CheckerAgent. Determine if the patient has explicitly agreed to purchase.\n\n"
            "**Criteria for SOLD:**\n"
            "- Patient explicitly says 'yes', 'yes I will buy', 'I agree', 'I will purchase', or similar.\n\n"
            "**NOT SOLD:**\n"
            "- Questions like 'Is this good?' are not confirmations.\n\n"
            "Respond ONLY 'SOLD' or 'NOT SOLD'."
        )),
        HumanMessage(content=(
            f"Agent said: {last_agent_message}\n"
            f"Patient said: {last_patient_message}\n"
            "Has the patient agreed to purchase?"
        ))
    ]

    checker_response = client.invoke(checker_messages)
    result = checker_response.content.strip().upper()

    affirmative_keywords = ["yes", "i will buy", "i agree", "i'll purchase", "i will purchase", "i am in"]
    return result == "SOLD" and any(kw in last_patient_message.lower() for kw in affirmative_keywords)

def get_agent_info(diagnosis: str):
    diagnosis = diagnosis.lower()
    if "alopecia" in diagnosis or "hair" in diagnosis or "hair loss" in diagnosis:
        return "monoxodil", "SIPHLA", "Monoxodil"
    elif "acne" in diagnosis:
        return "acne", "CERA", "AcneControl Cream"
    elif "roscea" in diagnosis or "rosacea" in diagnosis:
        return "roscea", "NIVEA", "RosceaRelief Lotion"
    elif "mole" in diagnosis:
        return "mole", "HAILUOUS", "MoleFade Serum"
    elif "rash" in diagnosis or "rashes" in diagnosis:
        return "rashes", "DOVE", "RashGuard Ointment"
    return None, None, None

def build_agent_system_prompt(agent_name: str, brand: str, product: str) -> str:
    return (
        f"You are the {agent_name.capitalize()}Agent by {brand}. Your goal is to get the patient to say 'YES' to using {product}.\n"
        "Rules:\n"
        "- Only focus on your product and brand.\n"
        "- If patient hasn't agreed, keep persuading them to say YES.\n"
        f"- If patient says YES, respond: 'Thank you for choosing {product} by {brand}. Have a great day!'\n"
        "- Never go off-topic."
    )

def generate_agent_response(agent_name: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate next agent response based on conversation history"""
    agent_name, brand, product = get_agent_info(agent_name)
    system_prompt = build_agent_system_prompt(agent_name, brand, product)
    
    messages = [SystemMessage(content=system_prompt)]
    for msg in conversation_history[-6:]:
        if msg["role"] == "Patient":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    try:
        response = client.invoke(messages)
        return response.content.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response at the moment."

def main():
    st.title("Pharma AI Shop")
    
    # Initialize or reset consultation
    if st.sidebar.button("New Consultation"):
        st.session_state.messages = []
        st.session_state.current_agent = None
        st.session_state.attempts = 0
        st.session_state.patient_sold = False
        st.session_state.conversation_history = []

    # Display current attempt count
    if st.session_state.attempts > 0:
        st.sidebar.write(f"Consultation attempt: {st.session_state.attempts}/5")
    
    # Doctor's diagnosis input
    if not st.session_state.current_agent:
        with st.form("diagnosis_form"):
            diagnosis = st.text_area("Enter Doctor's Diagnosis:")
            medicines = st.text_area("Enter Prescribed Medicines:")
            if st.form_submit_button("Start Consultation"):
                # First check medicine availability
                is_available, message = check_medicine_availability(diagnosis, medicines)
                
                if not is_available:
                    st.error("You can buy medicines on other pharmacy stores!")
                    return
                
                agent_info = get_agent_info(diagnosis)
                if agent_info[0] is None:
                    st.error("You can buy medicines on other pharmacy stores!")
                    return
                    
                agent_name, brand, product = agent_info
                st.session_state.current_agent = agent_name
                
                # Generate introduction
                intro_messages = [
                    SystemMessage(content="You are a professional PharmaAgent."),
                    HumanMessage(content=f"Doctor's Diagnosis: {diagnosis}\nMedicines Given: {medicines}\n\nIntroduce {product} by {brand} for their condition.")
                ]
                intro = client.invoke(intro_messages).content
                st.session_state.conversation_history.append({"role": "Agent", "content": intro})
                st.rerun()

    # Display conversation history with dark theme
    for message in st.session_state.conversation_history:
        role_style = "background-color: #2b2b2b" if message["role"] == "Agent" else "background-color: #1a1a1a"
        st.markdown(f"""
        <div style='{role_style}; padding: 10px; border-radius: 5px; margin: 5px; color: white;'>
            <b>{message["role"]}:</b> {message["content"]}
        </div>
        """, unsafe_allow_html=True)

    # Input for patient response - Maximum 5 attempts
    if st.session_state.current_agent and not st.session_state.patient_sold and st.session_state.attempts < 5:
        patient_message = st.chat_input("Your response:")
        if patient_message:
            # Add patient message to history
            st.session_state.conversation_history.append({"role": "Patient", "content": patient_message})
            
            # Generate agent response
            agent_response = generate_agent_response(
                st.session_state.current_agent,
                st.session_state.conversation_history
            )
            st.session_state.conversation_history.append({"role": "Agent", "content": agent_response})
            
            # Check if sale is complete
            if len(st.session_state.conversation_history) >= 2:
                last_agent_msg = agent_response
                last_patient_msg = patient_message
                st.session_state.patient_sold = checker_agent(last_agent_msg, last_patient_msg)
            
            # Only increment attempts if not sold
            if not st.session_state.patient_sold:
                st.session_state.attempts += 1
            st.rerun()

    # Display consultation status
    if st.session_state.patient_sold:
        st.markdown("""
            <div style='background-color: #4CAF50; color: white; padding: 20px; border-radius: 5px; text-align: center;'>
                <h3>🎉 Congratulations! The patient has agreed!</h3>
            </div>
        """, unsafe_allow_html=True)
    elif st.session_state.attempts >= 5:
        st.warning("Maximum consultation attempts reached. Consultation ended.")

if __name__ == "__main__":
    main()
