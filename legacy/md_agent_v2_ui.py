import os
from typing import Annotated, TypedDict, List, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import operator
import dashscope
import base64
import requests

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Dermatology Consultation Agent",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OAI_API_KEY"),
    deployment_name="gpt-4o",
    api_version="2023-03-15-preview"
)

# Define state management
class DermState(TypedDict):
    """State for dermatology consultation flow"""
    messages: Annotated[List[Any], operator.add]  # Chat history
    patient_data: dict  # Patient information
    complexity: str  # Complexity level
    members: List[dict]  # Medical team members
    opinions: Dict[str, str]  # Specialist opinions
    interaction_logs: Dict[str, Any]  # Discussion logs
    final_diagnosis: str
    treatment_plan: str

def get_image_description(image_content: bytes) -> str:
    """Retrieve image description using GPT-4 Vision"""
    # Encode the image content to base64
    encoded_image = base64.b64encode(image_content).decode('utf-8')
    
    # Prepare API request
    headers = {
        'api-key': os.getenv("AZURE_OAI_API_KEY"),
        'Content-Type': 'application/json'
    }
    
    endpoint = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}"
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an experienced dermatologist. Please describe the visual symptoms in a quantifiable descriptive way and provide a potential diagnosis in one word."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error processing image: {str(e)}"


def get_image_description_v2(image_content: bytes) -> str:
    """Retrieve image description using DashScope API"""
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    print("DashScope tool is called")
    
    # Encode the image content to base64
    encoded_image = base64.b64encode(image_content).decode('utf-8')
    
    response = dashscope.MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=[{
            'role': 'user',
            'content': [
                {
                    'image': encoded_image  # Send the base64 encoded image
                },
                {
                    'text': 'Imagine you are an experienced dermatologist, please describe the visual symptom with quantifiable descriptive way and give a potential diagnose in one word'
                }
            ]
        }]
    )
    
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"Error processing image: {response.code} - {response.message}"

def assess_complexity_node(state: DermState) -> DermState:
    """Determines case complexity and required team structure"""
    patient_info = state["patient_data"]
    image_content = patient_info.get('image_content')

    # Get image description using DashScope
    image_description = get_image_description(image_content)

    # Display image and description
    st.subheader("Image Description")
    st.image(image_content, use_column_width=True)
    st.write(image_description)

    # Prepare the content for the LLM
    assessment_prompt = f"""You are a dermatology triage specialist. Based on the following image description, classify case complexity as:
    - Low: Standard conditions manageable by one dermatologist (e.g., mild eczema, acne, common rashes)
    - Moderate: Complex cases needing multi-specialist collaboration (e.g., severe psoriasis, unusual rashes)
    - High: Severe cases requiring coordinated multi-team approach (e.g., severe drug reactions, complex autoimmune conditions)
    
    Provide only complexity level and brief rationale.

    Image Description: {image_description}

    Patient Information: {patient_info}
    """

    with st.spinner('Assessing case complexity...'):
        assessment = llm.invoke([
            SystemMessage(content=assessment_prompt)
        ])
    
    complexity = assessment.content.split("\n")[0].strip()
    st.subheader("Case Complexity Assessment")
    st.write(f"**Complexity Level:** {complexity}")
    st.write(assessment.content)

    # Update state
    state["complexity"] = complexity
    state["messages"].append(AIMessage(content=assessment.content))
    return state

def single_dermatologist_node(state: DermState) -> DermState:
    """Handle low complexity cases with a single dermatologist"""
    patient_info = state["patient_data"]
    image_content = patient_info.get('image_content')

    # Get image description using DashScope
    image_description = get_image_description(image_content)
    
    st.subheader("Primary Dermatologist Assessment")
    
    assessment_prompt = f"""You are a dermatologist handling a straightforward case.
    Provide a complete assessment in the following format:
    
    DIAGNOSIS:
    [Clear diagnosis with key findings]
    
    TREATMENT PLAN:
    [Specific treatment recommendations]

    Image Description: {image_description}

    Patient Information: {patient_info}
    """

    with st.spinner('Dermatologist is assessing the case...'):
        assessment = llm.invoke([
            SystemMessage(content=assessment_prompt)
        ])
    
    # Parse the diagnosis and treatment plan
    try:
        parts = assessment.content.split("DIAGNOSIS:")
        if len(parts) > 1:
            content = parts[1]
            diagnosis_treatment = content.split("TREATMENT PLAN:")
            if len(diagnosis_treatment) == 2:
                diagnosis = diagnosis_treatment[0].strip()
                treatment = diagnosis_treatment[1].strip()
            else:
                diagnosis = content.strip()
                treatment = "Treatment plan parsing error"
        else:
            diagnosis = assessment.content
            treatment = "Format parsing error"
    except:
        diagnosis = assessment.content
        treatment = "Error parsing treatment plan"

    # Display diagnosis and treatment plan
    st.subheader("Diagnosis")
    st.write(diagnosis)
    st.subheader("Treatment Plan")
    st.write(treatment)

    # Update state
    state["final_diagnosis"] = diagnosis
    state["treatment_plan"] = treatment
    state["messages"].append(AIMessage(content=assessment.content))
    return state

def recruit_specialists_node(state: DermState) -> DermState:
    """Recruits appropriate specialists based on complexity"""
    complexity = state["complexity"]
    patient_info = state["patient_data"]
    
    recruitment_prompt = """You are a medical recruiter. Based on case complexity and patient needs:
    - For moderate cases: Recruit 3-4 relevant specialists for group discussion
    - For high complexity: Recruit 2-3 teams of specialists for multi-team consultation
    
    Provide your response in the following format:
    SPECIALIST: [Role]
    EXPERTISE: [Primary focus]
    CONTRIBUTION: [Expected contribution]
    
    List each specialist separately with the exact headers shown above."""
    
    st.subheader("Recruiting Specialist Team")
    
    with st.spinner('Recruiting specialists...'):
        recruitment = llm.invoke([
            SystemMessage(content=recruitment_prompt),
            HumanMessage(content=f"""
            Complexity Level: {complexity}
            Patient Information: {patient_info}""")
        ])
    
    specialists = []
    current_specialist = {}
    
    for line in recruitment.content.split("\n"):
        line = line.strip()
        if line.startswith("SPECIALIST:"):
            if current_specialist:
                specialists.append(current_specialist)
            current_specialist = {
                "role": line.split("SPECIALIST:")[1].strip(),
                "status": "active"
            }
        elif line.startswith("EXPERTISE:"):
            if current_specialist:
                current_specialist["expertise"] = line.split("EXPERTISE:")[1].strip()
        elif line.startswith("CONTRIBUTION:"):
            if current_specialist:
                current_specialist["contribution"] = line.split("CONTRIBUTION:")[1].strip()
                
    if current_specialist:
        specialists.append(current_specialist)
        
    st.success(f"Recruited {len(specialists)} team members")
    for specialist in specialists:
        st.write(f"- **{specialist['role']}**: {specialist['expertise']}")
    
    # Update state
    state["members"] = specialists
    state["messages"].append(AIMessage(content=recruitment.content))
    return state

def facilitate_discussion_node(state: DermState) -> DermState:
    """Manages specialist discussions and opinion gathering"""
    opinions = state.get("opinions", {})
    specialists = state["members"]
    patient_info = state["patient_data"]
    complexity = state["complexity"].lower()
    interaction_logs = {}

    image_content = patient_info.get('image_content')
    # Get image description using DashScope
    image_description = get_image_description(image_content)
    
    st.subheader("Team Discussion")
    
    # Gather initial opinions
    for specialist in specialists:
        opinion_prompt = f"""You are a {specialist['role']}. 
        Provide a focused assessment including:
        1. Key observations from your specialty perspective
        2. Diagnosis considerations
        3. Treatment recommendations
        
        Format your response with clear DIAGNOSIS: and TREATMENT PLAN: sections.

        Image Description: {image_description}

        Patient Information: {patient_info}
        """

        with st.spinner(f"{specialist['role']} is providing assessment..."):
            opinion = llm.invoke([
                SystemMessage(content=opinion_prompt)
            ])
        opinions[specialist["role"]] = opinion.content
        st.write(f"**{specialist['role']}** assessment completed.")
            
    # Facilitate inter-specialist discussion if needed
    if "high" in complexity:
        st.subheader("Multi-team Consultation")
        for i in range(3):  # Maximum 3 discussion rounds
            round_log = {}
            for specialist in specialists:
                other_opinions = {k:v for k,v in opinions.items() if k != specialist["role"]}
                
                response_prompt = f"""You are a {specialist['role']}.
                Review other specialists' opinions and provide:
                1. Points of agreement/disagreement
                2. Questions for specific specialists
                3. Updated assessment based on discussion

                Image Description: {image_description}

                Patient Information: {patient_info}

                Other Opinions: {other_opinions}
                """

                with st.spinner(f"{specialist['role']} is participating in discussion..."):
                    response = llm.invoke([
                        SystemMessage(content=response_prompt)
                    ])
                round_log[specialist["role"]] = response.content
                opinions[specialist["role"]] = response.content
            
            interaction_logs[f"Round {i+1}"] = round_log
            st.write(f"Consultation round {i+1} completed.")
                
    # Update state
    state["opinions"] = opinions
    state["interaction_logs"] = interaction_logs
    state["messages"].append(AIMessage(content=str(opinions)))
    return state

def synthesize_decision_node(state: DermState) -> DermState:
    """Synthesizes specialist inputs into final decision"""
    opinions = state["opinions"]
    complexity = state["complexity"]
    patient_info = state["patient_data"]

    image_content = patient_info.get('image_content')
    # Get image description using DashScope
    image_description = get_image_description(image_content)
    
    st.subheader("Synthesizing Final Decision")
    
    final_decision_prompt = f"""You are the lead dermatologist.
    Synthesize all specialist inputs to provide:
    
    DIAGNOSIS:
    [Detailed diagnosis with key findings]
    
    TREATMENT PLAN:
    [Comprehensive treatment approach]

    Image Description: {image_description}

    Case Complexity: {complexity}
    Specialist Opinions: {opinions}
    """

    with st.spinner('Synthesizing final decision...'):
        final_decision = llm.invoke([
            SystemMessage(content=final_decision_prompt)
        ])
    
    # Parse the diagnosis and treatment plan
    try:
        parts = final_decision.content.split("DIAGNOSIS:")
        if len(parts) > 1:
            content = parts[1]
            diagnosis_treatment = content.split("TREATMENT PLAN:")
            if len(diagnosis_treatment) == 2:
                diagnosis = diagnosis_treatment[0].strip()
                treatment = diagnosis_treatment[1].strip()
            else:
                diagnosis = content.strip()
                treatment = "Treatment plan parsing error"
        else:
            diagnosis = final_decision.content
            treatment = "Format parsing error"
    except:
        diagnosis = final_decision.content
        treatment = "Error parsing treatment plan"
    
    # Display final diagnosis and treatment plan
    st.subheader("Final Diagnosis")
    st.write(diagnosis)
    st.subheader("Final Treatment Plan")
    st.write(treatment)
        
    # Update state
    state["final_diagnosis"] = diagnosis
    state["treatment_plan"] = treatment
    state["messages"].append(AIMessage(content=final_decision.content))
    return state

def display_final_report(state: DermState):
    st.title("Final Dermatology Report")
    st.subheader("Case Complexity")
    st.write(state["complexity"])

    if state["members"]:
        st.subheader("Consulting Team")
        for member in state["members"]:
            st.write(f"- **{member['role']}**: {member.get('expertise', '')}")

    st.subheader("Diagnosis")
    st.write(state["final_diagnosis"])

    st.subheader("Treatment Plan")
    st.write(state["treatment_plan"])

    if state["interaction_logs"]:
        st.subheader("Key Discussion Points")
        for round_name, round_log in state["interaction_logs"].items():
            st.write(f"**{round_name}:**")
            for specialist, comments in round_log.items():
                st.write(f"**{specialist}:** {comments}")

    # Add download button for the report
    report_content = f"""
    Case Complexity: {state["complexity"]}

    Diagnosis:
    {state["final_diagnosis"]}

    Treatment Plan:
    {state["treatment_plan"]}
    """
    st.download_button(
        label="Download Report",
        data=report_content,
        file_name="dermatology_report.txt",
        mime="text/plain"
    )

    # Optionally, provide a button to restart
    if st.button("Start New Consultation"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

def main():
    st.title("Dermatology Consultation Agent")
    st.markdown("---")

    # Add app description in sidebar
    with st.sidebar:
        st.markdown("""
        ## About This App
        This is an AI-powered dermatology consultation system that provides:
        - Case complexity assessment
        - Diagnosis and treatment plan
        - Specialist team consultation (if needed)
        
        **Note:** This is for educational purposes only and should not replace professional medical advice.
        
        ### How to Use
        1. Upload an image of your skin condition or provide an image URL.
        2. Fill in the patient information form.
        3. Receive the consultation results.
        """)
        st.markdown("---")
        st.markdown("v1.0.0")

    if "state" not in st.session_state:
        # Collect image URL or upload image
        st.subheader("Step 1: Provide Skin Condition Image")
        image_option = st.radio("How would you like to provide the image?", ("Upload Image", "Enter Image URL", "Use Default Image"))

        if image_option == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image of the skin condition", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                image_content = uploaded_file.read()
                st.success("Image uploaded successfully.")
            else:
                st.stop()
        elif image_option == "Enter Image URL":
            image_url = st.text_input("Enter the URL of the image")
            if image_url:
                try:
                    # Fetch the image content from the URL
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        image_content = response.content
                        st.success("Image fetched successfully.")
                    else:
                        st.error("Failed to fetch image from the URL.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error fetching image: {str(e)}")
                    st.stop()
            else:
                st.stop()
        else:
            # Use default image
            default_image_url = "https://derma-image.oss-rg-china-mainland.aliyuncs.com/shutterstock_1892383180.webp"
            try:
                # Fetch the default image content
                response = requests.get(default_image_url)
                if response.status_code == 200:
                    image_content = response.content
                    st.info("Using default image.")
                else:
                    st.error("Failed to fetch default image.")
                    st.stop()
            except Exception as e:
                st.error(f"Error fetching default image: {str(e)}")
                st.stop()

        st.subheader("Step 2: Fill in Patient Information")
        with st.form("patient_info_form"):
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
            symptoms = st.text_input("Symptoms")
            duration = st.text_input("Duration")
            history = st.text_input("Medical History")
            allergies = st.text_input("Allergies")
            medications = st.text_input("Current Medications")
            recent_changes = st.text_input("Recent Changes")
            additional_symptoms = st.text_input("Additional Symptoms")
            previous_treatments = st.text_input("Previous Treatments")
            impact = st.text_input("Impact on Daily Life")
            family_history = st.text_input("Family Medical History")
            submitted = st.form_submit_button("Submit")

        if submitted:
            # Build the patient_info dictionary
            patient_info = {
                "age": age,
                "symptoms": symptoms,
                "duration": duration,
                "history": history,
                "allergies": allergies,
                "medications": medications,
                "recent_changes": recent_changes,
                "additional_symptoms": additional_symptoms,
                "previous_treatments": previous_treatments,
                "impact": impact,
                "family_history": family_history,
                "image_content": image_content  # Store the image content
            }

            # Initialize state
            st.session_state["state"] = {
                "messages": [],
                "patient_data": patient_info,
                "complexity": "",
                "members": [],
                "opinions": {},
                "interaction_logs": {},
                "final_diagnosis": "",
                "treatment_plan": ""
            }
            st.session_state["node"] = "assess_complexity"
            st.rerun()
        else:
            st.stop()
    else:
        state = st.session_state["state"]
        node = st.session_state["node"]

        if node == "assess_complexity":
            state = assess_complexity_node(state)
            complexity = state["complexity"].lower()
            if "low" in complexity:
                st.session_state["node"] = "single_dermatologist"
            else:
                st.session_state["node"] = "recruit_specialists"
            st.session_state["state"] = state
            st.rerun()
        elif node == "single_dermatologist":
            state = single_dermatologist_node(state)
            st.session_state["node"] = "END"
            st.session_state["state"] = state
            st.rerun()
        elif node == "recruit_specialists":
            state = recruit_specialists_node(state)
            st.session_state["node"] = "facilitate_discussion"
            st.session_state["state"] = state
            st.rerun()
        elif node == "facilitate_discussion":
            state = facilitate_discussion_node(state)
            st.session_state["node"] = "synthesize_decision"
            st.session_state["state"] = state
            st.rerun()
        elif node == "synthesize_decision":
            state = synthesize_decision_node(state)
            st.session_state["node"] = "END"
            st.session_state["state"] = state
            st.rerun()
        elif node == "END":
            display_final_report(state)
        else:
            st.error("Unknown node in the workflow.")
            st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        if st.button("Restart Application"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
