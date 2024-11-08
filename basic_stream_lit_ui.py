import os
from typing import Annotated, Dict, Any, Sequence
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
import operator

# Load environment variables
load_dotenv()

# Set up Azure OpenAI
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OAI_API_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2023-05-15",
    streaming=True )

# Define state management
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Define constants
RECEPTIONIST_QUESTIONS = """1. Patient's age and gender
2. Duration of skin condition
3. Location and appearance of condition
4. Previous treatments tried
5. Any allergies or medical history"""

# Node functions
def receptionist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    st.subheader("Receptionist's Questions")
    st.write(RECEPTIONIST_QUESTIONS)
    st.markdown("---")
    
    patient_answers = st.text_area("Please answer all questions above:", key="patient_answers")
    
    if 'receptionist_submitted' not in st.session_state:
        if st.button("Submit Answers"):
            with st.spinner('Processing your answers...'):
                state["messages"].append(
                    HumanMessage(content=f"Patient's answers to questions: {patient_answers}")
                )
                state["next"] = "dermatologist"
                st.session_state['receptionist_submitted'] = True
                st.session_state['state'] = state
                st.rerun()
        else:
            st.stop()
    else:
        st.stop()
    
    return state

def dermatologist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = """You are a dermatologist. Based on the patient's information, provide ONLY:
1. Brief symptom summary
2. Most likely diagnosis
3. Confidence level (percentage)

Be concise and do NOT ask additional questions."""
    
    if 'dermatologist_response' not in st.session_state:
        # Create a placeholder for the streaming response
        diagnosis_placeholder = st.empty()
        full_response = ""
        
        with st.spinner('Dermatologist is analyzing your case...'):
            try:
                messages = [
                    SystemMessage(content=prompt),
                    *state["messages"]
                ]
                
                # Invoke the language model with streaming
                response_stream = llm.stream(messages)
                
                # Iterate over the streaming response
                for chunk in response_stream:
                    # Append the new text to the full response
                    full_response += chunk.content
                    
                    # Update the placeholder with the current full response
                    diagnosis_placeholder.markdown(full_response)
                
                st.session_state['dermatologist_response'] = full_response
                state["messages"].append(AIMessage(content=full_response))
                state["next"] = "pharmacist"
                st.session_state['state'] = state
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during dermatologist consultation: {str(e)}")
                st.stop()
    else:
        st.subheader("Dermatologist's Assessment")
        st.write(st.session_state['dermatologist_response'])
        st.markdown("---")
        st.stop()
    
    return state

def pharmacist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = """You are a pharmacist. Provide prescription in this EXACT format:

RECOMMENDED MEDICATIONS:
1. [Medication Name]
   - Dosage: [amount]
   - Frequency: [times per day]
   - Duration: [days]

ESTIMATED COSTS:
- Medication 1: $[amount]
- Total: $[total]

USAGE INSTRUCTIONS:
1. [Specific instructions]

PRECAUTIONS:
1. [Key precautions]"""
    
    if 'pharmacist_response' not in st.session_state:
        with st.spinner('Preparing your prescription...'):
            try:
                messages = [
                    SystemMessage(content=prompt),
                    *state["messages"]
                ]
                
                response = llm.invoke(messages)
                
                st.session_state['pharmacist_response'] = response.content
                state["messages"].append(AIMessage(content=response.content))
                state["next"] = "END"
                st.session_state['state'] = state
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during prescription preparation: {str(e)}")
                st.stop()
    else:
        st.subheader("Prescription Details")
        st.write(st.session_state['pharmacist_response'])
        st.markdown("---")
        st.stop()
    
    return state

# Set up workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("receptionist", receptionist_node)
workflow.add_node("dermatologist", dermatologist_node)
workflow.add_node("pharmacist", pharmacist_node)

# Add edges
workflow.add_edge(START, "receptionist")

workflow.add_conditional_edges(
    "receptionist",
    lambda x: x["next"],
    {
        "dermatologist": "dermatologist",
    }
)

workflow.add_conditional_edges(
    "dermatologist",
    lambda x: x["next"],
    {
        "pharmacist": "pharmacist",
    }
)

workflow.add_conditional_edges(
    "pharmacist",
    lambda x: x["next"],
    {
        "END": END
    }
)

# Compile workflow
graph = workflow.compile()

# Initialize session state
if 'state' not in st.session_state:
    st.session_state['state'] = {
        "messages": [],
        "next": "receptionist"
    }

def reset_consultation():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

def run_consultation():
    st.title("Dermatology Consultation System")
    st.markdown("---")
    
    # Add reset button in sidebar
    with st.sidebar:
        if st.button("Start New Consultation"):
            reset_consultation()
    
    state = st.session_state['state']
    
    if not state['messages']:
        # Initial patient input
        patient_input = st.text_input("Please describe your skin concern:", key="initial_input")
        if patient_input:
            with st.spinner('Starting consultation...'):
                state['messages'] = [HumanMessage(content=patient_input)]
                state['next'] = 'receptionist'
                st.session_state['state'] = state
                st.rerun()
        else:
            st.stop()
    else:
        # Display conversation history
        with st.expander(":green[Conversation History]", expanded=False):
            for msg in state['messages']:
                if isinstance(msg, HumanMessage):
                    st.info(f"Patient: {msg.content}", icon="👤")
                elif isinstance(msg, AIMessage):
                    st.warning(f"Doctor: {msg.content}", icon="⚕️")
        
        # Based on 'next', invoke the appropriate node
        if state['next'] == 'receptionist':
            state = receptionist_node(state)
        elif state['next'] == 'dermatologist':
            state = dermatologist_node(state)
        elif state['next'] == 'pharmacist':
            state = pharmacist_node(state)
        elif state['next'] == 'END':
            st.success("Consultation Complete. Thank you!")
            st.markdown("**Note:** This is an AI-generated consultation. Please consult with a real healthcare provider for accurate medical advice.")
            
            # Add download button for consultation summary
            consultation_summary = "\n\n".join([f"{'Patient' if isinstance(msg, HumanMessage) else 'Doctor'}: {msg.content}" 
                                              for msg in state['messages']])
            st.download_button(
                label="Download Consultation Summary",
                data=consultation_summary,
                file_name="consultation_summary.txt",
                mime="text/plain"
            )
            st.stop()

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="AI Dermatology Consultation",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add CSS for better styling
        st.markdown("""
            <style>
            .stButton>button {
                width: 100%;
                margin-top: 10px;
            }
            .success {
                padding: 20px;
                border-radius: 5px;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .info {
                padding: 10px;
                border-radius: 5px;
                background-color: #e2e3e5;
                border: 1px solid #d6d8db;
                color: #383d41;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Add app description in sidebar
        with st.sidebar:
            st.markdown("""
            ## About This App
            This is an AI-powered dermatology consultation system that provides:
            - Initial assessment
            - Preliminary diagnosis
            - Treatment recommendations
            
            **Note:** This is for educational purposes only and should not replace professional medical advice.
            
            ### How to Use
            1. Describe your skin concern
            2. Answer the receptionist's questions
            3. Receive dermatologist's assessment
            4. Get prescription details
            """)
            
            # Add version info
            st.markdown("---")
            st.markdown("v1.0.0")
            
            # Add error reporting
            with st.expander("Report an Issue"):
                issue_description = st.text_area("Describe the issue:")
                if st.button("Submit Issue"):
                    # Here you could add functionality to log issues
                    st.success("Thank you for reporting the issue!")
        
        run_consultation()
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        if st.button("Restart Application"):
            reset_consultation()
