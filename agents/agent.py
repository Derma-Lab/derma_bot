import os
import asyncio
import base64
import tempfile
import requests
import dashscope
import streamlit as st
from PIL import Image
from typing import Optional, List, Dict, TypedDict
from dataclasses import dataclass
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from pdf2image import convert_from_path
from langgraph.graph import StateGraph
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AZURE_OAI_API_KEY = os.getenv("AZURE_OAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://derma-lab-test.openai.azure.com")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not AZURE_OAI_API_KEY:
    raise ValueError("AZURE_OAI_API_KEY not found in environment variables")

if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY not found in environment variables")

dashscope.api_key = DASHSCOPE_API_KEY

def initialize_azure_client(deployment_name="gpt-4o-mini"):
    return AzureChatOpenAI(
        api_key=AZURE_OAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=deployment_name,
        api_version="2024-02-15-preview",
        temperature=0.7
    )

client = initialize_azure_client()

### Qwen-VL API call ###
async def call_vlm(image: Image.Image, prompt: str = "") -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        image.save(tmp.name, format="PNG")
        with open(tmp.name, "rb") as f:
            img_bytes = f.read()
    image_b64 = base64.b64encode(img_bytes).decode("utf-8")

    messages = [{
        'role': 'user',
        'content': [
            {
                'image': f"data:image/png;base64,{image_b64}"
            },
            {
                'text': prompt
            }
        ]
    }]

    response = dashscope.MultiModalConversation.call(
        model='qwen-vl-max-0809',
        messages=messages
    )

    if response and response.output and 'text' in response.output:
        return response.output['text']
    return ""

async def extract_pdf_summary(pdf_path: str) -> str:
    if not pdf_path:
        return ""
        
    images = convert_from_path(pdf_path)
    if not images:
        return "No images extracted from PDF."

    first_page = images[0]
    prompt = (
        "You are a medical data summarization assistant. Below is an image of a patient intake form. "
        "Provide a single sentence summary including the patient's name, age, gender, relevant medical history, "
        "medications, and allergies if visible. If not specified, say 'None known'. "
        "Do not guess any details not clearly visible. Do not assume family history."
    )

    # Try full image
    try:
        analysis = await call_vlm(first_page, prompt)
        if analysis:
            return analysis.strip()
    except Exception:
        pass

    # If fails, try half image
    width, height = first_page.size
    half_box = (0, 0, width, height // 2)
    top_half = first_page.crop(half_box)
    try:
        analysis = await call_vlm(top_half, prompt)
        if analysis:
            return analysis.strip()
        else:
            return "No analysis from Qwen-VL."
    except Exception as e:
        return f"Qwen-VL Error: {e}"

### Data Structures ###
@dataclass
class PatientInfo:
    basic_info: dict
    medical_history: dict
    current_symptoms: str
    images: Optional[List] = None  # Can hold URLs, file paths, or UploadedFile objects

class MedicalState(TypedDict):
    patient_info: PatientInfo
    complexity: str
    case_digest: str
    current_diagnosis: str
    specialist_analyses: Dict[str, str]
    final_diagnosis: str
    prescription: str
    consultation_path: str
    human_feedback: str
    final_assessment: Dict[str, str]
    pharma_medication: str

### Workflow Functions ###
async def process_initial_data(state: MedicalState, pdf_path: str) -> MedicalState:
    # Process Images
    if state['patient_info'].images:
        for img_obj in state['patient_info'].images:
            # If this is an UploadedFile object:
            if hasattr(img_obj, "getvalue"):
                try:
                    img = Image.open(BytesIO(img_obj.getvalue())).convert("RGB")
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
                    continue
            else:
                # If it's a string, could be a URL or local file
                try:
                    if str(img_obj).startswith("http"):
                        r = requests.get(img_obj)
                        img = Image.open(BytesIO(r.content)).convert("RGB")
                    else:
                        img = Image.open(img_obj).convert("RGB")
                except Exception as e:
                    st.error(f"Error loading image from path: {img_obj}, error: {e}")
                    continue

            analysis = await call_vlm(img, "Describe any visible skin conditions or symptoms. Do not guess details not visible.")
            if analysis:
                state['current_diagnosis'] = analysis
                state['patient_info'].basic_info['visual_findings'] = analysis

    # Process PDF
    pdf_summary = await extract_pdf_summary(pdf_path)
    state['patient_info'].basic_info['record_summary'] = pdf_summary

    # Create case digest
    patient_name = state['patient_info'].basic_info.get('name', 'Unknown')
    patient_age = state['patient_info'].basic_info.get('age', 'Unknown')
    mh = state['patient_info'].medical_history.get('conditions', [])
    mh_str = ', '.join(mh) if mh else 'None known'

    state['case_digest'] = (
        f"Patient {patient_name}, "
        f"{patient_age} years old, "
        f"presenting with {state['patient_info'].current_symptoms}. "
        f"Medical history: {mh_str}. "
        f"From intake form: {pdf_summary}"
    )
    state['patient_info'].basic_info['case_digest'] = state['case_digest']

    return state

async def general_dermatologist_analysis(state: MedicalState) -> MedicalState:
    messages = [
        HumanMessage(content=f"""
        As a general dermatologist, analyze this case focusing on common skin conditions:
        Patient Information: {state['case_digest']}
        Current Symptoms: {state['patient_info'].current_symptoms}
        Visual Analysis: {state.get('current_diagnosis', '')}

        Consider:
        1. Visible skin changes and patterns
        2. Common dermatological conditions
        3. Initial treatment recommendations

        Do not guess family history if not provided. Provide a detailed dermatological assessment.
        """)
    ]
    response = await client.ainvoke(messages)
    state['specialist_analyses']['General_Dermatologist'] = response.content
    return state

async def endocrine_dermatologist_analysis(state: MedicalState) -> MedicalState:
    messages = [
        HumanMessage(content=f"""
        As a dermatologist specializing in endocrine-related skin conditions:
        Patient Information: {state['case_digest']}
        Current Symptoms: {state['patient_info'].current_symptoms}
        Previous Analysis: {state['specialist_analyses'].get('General_Dermatologist', '')}

        Do not assume family history. Provide analysis focusing on endocrine-related aspects.
        """)
    ]
    response = await client.ainvoke(messages)
    state['specialist_analyses']['Endocrine_Dermatologist'] = response.content
    return state

async def immune_dermatologist_analysis(state: MedicalState) -> MedicalState:
    messages = [
        HumanMessage(content=f"""
        As a dermatologist specializing in immune-related skin conditions:
        Patient Information: {state['case_digest']}
        Current Symptoms: {state['patient_info'].current_symptoms}
        Previous Analyses:
        General Assessment: {state['specialist_analyses'].get('General_Dermatologist', '')}
        Endocrine Assessment: {state['specialist_analyses'].get('Endocrine_Dermatologist', '')}

        Do not guess family history. Focus on immune-related conditions.
        """)
    ]
    response = await client.ainvoke(messages)
    state['specialist_analyses']['Immune_Dermatologist'] = response.content
    return state

async def determine_consultation_path(state: MedicalState) -> str:
    messages = [
        HumanMessage(content=f"""
        You are a medical complexity assessment assistant. Based on the information given, determine the complexity of the case:
        - simple
        - moderate
        - complicated

        Patient Information:
        - Basic Info: {state['patient_info'].basic_info}
        - Current Symptoms: {state['patient_info'].current_symptoms}
        - Medical History: {state['patient_info'].medical_history}
        - Initial Analysis: {state.get('current_diagnosis', '')}

        Do not guess family history. Return exactly one word: 'simple', 'moderate', or 'complicated'.
        """)
    ]
    response = await client.ainvoke(messages)
    complexity = response.content.strip().lower()
    return complexity

async def synthesize_diagnosis(state: MedicalState) -> Dict:
    all_analyses = "\n".join([f"{k}: {v}" for k, v in state['specialist_analyses'].items()])
    if not all_analyses.strip():
        return {
            "disease_name": "Unknown",
            "treatment_plan": "Unknown",
            "items_to_note": "Unknown"
        }

    llm_prompt = f"""
    You are a medical synthesis assistant. Based on the specialist analyses and patient info, generate a structured final diagnosis and treatment plan.
    The output should include:
    Disease Name: [Primary Diagnosis]
    Treatment Plan: [Specific Treatment Recommendations]
    Items to Note: [Additional Considerations]

    Do not guess family history if not provided.
    Specialist Analyses:
    {all_analyses}

    Patient Information:
    - Name: {state['patient_info'].basic_info.get('name', 'Unknown')}
    - Age: {state['patient_info'].basic_info.get('age', 'Unknown')}
    - Gender: {state['patient_info'].basic_info.get('gender', 'Unknown')}
    - Medical History: {', '.join(state['patient_info'].medical_history.get('conditions', []) or ['None known'])}
    - Current Symptoms: {state['patient_info'].current_symptoms}
    - Initial Analysis: {state.get('current_diagnosis', '')}
    """
    messages = [HumanMessage(content=llm_prompt)]
    response = await client.ainvoke(messages)

    lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
    disease_name = "Unknown"
    treatment_plan = "Unknown" 
    items_to_note = "Unknown"
    for line in lines:
        if line.lower().startswith("disease name:"):
            disease_name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("treatment plan:"):
            treatment_plan = line.split(":", 1)[1].strip()
        elif line.lower().startswith("items to note:"):
            items_to_note = line.split(":", 1)[1].strip()

    return {
        "disease_name": disease_name,
        "treatment_plan": treatment_plan,
        "items_to_note": items_to_note
    }

async def pharmaagent_analysis(state: MedicalState) -> MedicalState:
    # Use the final_assessment, patient info, and specialist analyses to get medications
    final_assessment = state["final_assessment"]
    all_analyses = "\n".join([f"{k}: {v}" for k,v in state['specialist_analyses'].items()])

    llm_prompt = f"""
    You are a pharma agent. Given the following information, return strictly only the medications required:

    Patient Information:
    - Name: {state['patient_info'].basic_info.get('name', 'Unknown')}
    - Age: {state['patient_info'].basic_info.get('age', 'Unknown')}
    - Gender: {state['patient_info'].basic_info.get('gender', 'Unknown')}
    - Medical History: {', '.join(state['patient_info'].medical_history.get('conditions', []) or ['None known'])}
    - Current Symptoms: {state['patient_info'].current_symptoms}

    Specialist Analyses:
    {all_analyses}

    Final Assessment:
    Disease Name: {final_assessment['disease_name']}
    Treatment Plan: {final_assessment['treatment_plan']}
    Items to Note: {final_assessment['items_to_note']}

    Return strictly only the medication required, in a short bullet list format.
    """
    messages = [HumanMessage(content=llm_prompt)]
    response = await client.ainvoke(messages)
    # Assume response is medication list
    state["pharma_medication"] = response.content.strip()
    return state

async def final_assessment_node(state: MedicalState) -> MedicalState:
    final_assessment = await synthesize_diagnosis(state)
    state["final_assessment"] = final_assessment
    return state

async def create_dermatology_workflow() -> StateGraph:
    workflow = StateGraph(MedicalState)
    workflow.add_node("general_derm", general_dermatologist_analysis)
    workflow.add_node("endocrine_derm", endocrine_dermatologist_analysis)
    workflow.add_node("immune_derm", immune_dermatologist_analysis)
    workflow.add_node("final_assessment_node", final_assessment_node)
    workflow.add_node("pharmaagent", pharmaagent_analysis)
    workflow.add_node("end", lambda x: x)

    workflow.set_entry_point("general_derm")

    def route_by_complexity(state: MedicalState) -> str:
        return state.get("consultation_path", "simple")

    # After general derm
    workflow.add_conditional_edges(
        "general_derm",
        route_by_complexity,
        {
            "simple": "final_assessment_node",
            "moderate": "endocrine_derm",
            "complicated": "endocrine_derm"
        }
    )

    # After endocrine derm
    workflow.add_conditional_edges(
        "endocrine_derm",
        route_by_complexity,
        {
            "simple": "final_assessment_node",
            "moderate": "final_assessment_node",
            "complicated": "immune_derm"
        }
    )

    # After immune derm always go to final_assessment_node
    workflow.add_edge("immune_derm", "final_assessment_node")

    # After final_assessment_node always go to pharmaagent
    workflow.add_edge("final_assessment_node", "pharmaagent")

    # After pharmaagent go to end
    workflow.add_edge("pharmaagent", "end")

    workflow.set_finish_point("end")
    return workflow.compile()

def run():
    st.set_page_config(layout="wide")
    st.title("Dermatology Consultation Assistant")
    st.markdown("An integrated demonstration of PDF, image analysis, multi-step workflow, and Azure OpenAI.")

    # Sidebar for patient info input
    with st.sidebar:
        st.header("Patient Information")
        name = st.text_input("Name", "Yan Pan")
        age = st.text_input("Age", "20")
        gender = st.selectbox("Gender", ["female", "male", "other"], index=0)

        conditions = st.text_area("Medical Conditions (comma-separated)", "Type 2 Diabetes")
        medications = st.text_area("Medications (comma-separated)", "Metformin")
        allergies = st.text_area("Allergies (comma-separated)", "")

        st.header("Patient Records")
        uploaded_pdf = st.file_uploader("Upload Patient Intake PDF (Optional)", type=["pdf"])
        pdf_path = ""
        if uploaded_pdf:
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_pdf.read())
            pdf_path = "temp_uploaded.pdf"
            st.success("PDF uploaded successfully!")

        # Handle image upload with checks
        uploaded_image = st.file_uploader("Upload Patient Skin Image (Optional)", type=["png", "jpg", "jpeg"])
        patient_images = []
        if uploaded_image is not None:
            try:
                img = Image.open(BytesIO(uploaded_image.getvalue()))
                width, height = img.size
                if width > 4000 or height > 4000:
                    st.error("Image resolution too high. Please upload an image smaller than 4000x4000.")
                elif width < 100 or height < 100:
                    st.error("Image resolution too small. Please upload an image larger than 100x100.")
                else:
                    patient_images = [uploaded_image]
                    st.success("Image uploaded successfully!")
            except Exception as e:
                st.error(f"Invalid image or format. Please upload a valid PNG/JPG/JPEG. Error: {e}")

    # Additional Symptoms Input Before Run
    additional_symptoms = st.text_input("Add any additional symptoms (optional):", "")
    if additional_symptoms.strip():
        current_symptoms = f"Red, scaly patches on face and neck, worsening with sun exposure; {additional_symptoms.strip()}"
    else:
        current_symptoms = "Red, scaly patches on face and neck, worsening with sun exposure"

    patient_info = PatientInfo(
        basic_info={
            "name": name,
            "age": age,
            "gender": gender,
        },
        medical_history={
            "conditions": [c.strip() for c in conditions.split(",") if c.strip()],
            "medications": [m.strip() for m in medications.split(",") if m.strip()],
            "allergies": [a.strip() for a in allergies.split(",") if a.strip()]
        },
        current_symptoms=current_symptoms,
        images=patient_images
    )

    human_feedback = "Patient reports stress triggers flare-ups"

    if "show_pharma_details" not in st.session_state:
        st.session_state["show_pharma_details"] = False
    if "final_assessment" not in st.session_state:
        st.session_state["final_assessment"] = None
    if "pharma_medication" not in st.session_state:
        st.session_state["pharma_medication"] = None

    run_button = st.button("Run Consultation")

    if run_button:
        async def run_consultation():
            workflow = await create_dermatology_workflow()
            initial_state = MedicalState(
                patient_info=patient_info,
                complexity="",
                case_digest="",
                current_diagnosis="",
                specialist_analyses={},
                final_diagnosis="",
                prescription="",
                consultation_path="general",
                human_feedback=human_feedback,
                final_assessment={},
                pharma_medication=""
            )

            with st.spinner("Processing initial data..."):
                state = await process_initial_data(initial_state, pdf_path)

            complexity_placeholder = st.empty()
            with st.spinner("Deciding complexity..."):
                consultation_path = await determine_consultation_path(state)
            state["consultation_path"] = consultation_path
            complexity_placeholder.write(f"**Case complexity determined:** {consultation_path}")

            if consultation_path == "moderate":
                st.info("Additional consultation with endocrine dermatologist...")
            elif consultation_path == "complicated":
                st.warning("Case is complicated. Consulting immunologist...")

            with st.spinner("Consulting specialists and finalizing..."):
                final_state = await workflow.ainvoke(state)

            final_assessment = final_state["final_assessment"]
            st.session_state["final_assessment"] = final_assessment
            st.session_state["pharma_medication"] = final_state["pharma_medication"]

            with st.expander("Show Specialist Analyses"):
                for specialist, analysis in final_state["specialist_analyses"].items():
                    if specialist == "General_Dermatologist":
                        st.success(f"### {specialist}\n{analysis}")
                    elif specialist == "Endocrine_Dermatologist":
                        st.info(f"### {specialist}\n{analysis}")
                    elif specialist == "Immune_Dermatologist":
                        st.warning(f"### {specialist}\n{analysis}")
                    else:
                        st.write(f"### {specialist}\n{analysis}")

            st.markdown("### Conclusion:")
            st.markdown(f"**Disease Name:** {final_assessment['disease_name']}")
            st.markdown(f"**Treatment Plan:** {final_assessment['treatment_plan']}")
            st.markdown(f"**Items to Note:** {final_assessment['items_to_note']}")

            final_text = (
                f"Disease Name: {final_assessment['disease_name']}\n\n"
                f"Treatment Plan: {final_assessment['treatment_plan']}\n\n"
                f"Items to Note: {final_assessment['items_to_note']}\n"
            )
            st.download_button(
                label="📥 Download Final Assessment",
                data=final_text.encode("utf-8"),
                file_name="final_assessment.txt",
                mime="text/plain"
            )

        asyncio.run(run_consultation())

    # Show button to copy details for pharma shop after consultation is done
    if st.session_state["final_assessment"] is not None and not st.session_state["show_pharma_details"]:
        if st.button("Copy details for pharma shop"):
            st.session_state["show_pharma_details"] = True

    # If user wants pharma details, show text boxes
    if st.session_state["show_pharma_details"]:
        st.markdown("### Pharma Shop Details")
        st.info("Below are details you can copy for doctor's diagnosis and medicines:")
        # Doctor's diagnosis details come from the final_assessment
        diag = f"Disease: {st.session_state['final_assessment']['disease_name']}\nRecommendations: {st.session_state['final_assessment']['treatment_plan']}"
        meds = st.session_state["pharma_medication"] if st.session_state["pharma_medication"] else "Medication A: ...\nMedication B: ..."
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Doctor's Diagnosis Details", diag, height=200)
        with col2:
            st.text_area("Medicines Required", meds, height=200)

if __name__ == "__main__":
    run()
