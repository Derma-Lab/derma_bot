import os 
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, Graph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import operator
from dataclasses import dataclass
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define state schema
class DermatologyState(TypedDict):
    """Graph state definition"""
    messages: Annotated[list, operator.add]  # chat history
    patient_info: dict  # patient info
    skin_condition: str  # skin condition
    difficulty_level: str  # Basic, Intermediate, or Advanced
    medical_dermatologist_consult: dict  # medical dermatologist view
    surgical_dermatologist_consult: dict  # surgical dermatologist view
    dermatopathologist_consult: dict  # dermatopathologist view
    diagnosis: str  # diagnosis result
    treatment_plan: str  # treatment plan
    prescription: str  # prescription

def determine_difficulty(state: DermatologyState, llm) -> str:
    """Determine the difficulty level of the case based on patient information and symptoms"""
    patient_info = state["patient_info"]["summary"] if state.get("patient_info") else ""
    skin_condition = state["skin_condition"]
    
    assessment = llm.invoke([
        SystemMessage(content="""You are an experienced dermatology triage specialist. Analyze the patient's condition and determine the appropriate difficulty level:

        1) Basic: Can be handled by a single dermatologist
           - Common conditions like acne, eczema, or simple rashes
           - Clear symptoms and typical presentation
           - Standard treatment protocols available
           
        2) Intermediate: Requires consultation between multiple dermatology specialists
           - Complex conditions requiring multiple specialist perspectives
           - Unclear diagnosis requiring additional tests
           - Multiple treatment options to consider
           
        3) Advanced: Requires collaboration between multiple dermatology teams
           - Rare or severe conditions
           - Multiple comorbidities or complications
           - High-risk cases requiring coordinated care
           - Surgical intervention likely needed
           
        Respond only with "Basic", "Intermediate", or "Advanced" followed by a brief justification."""),
        HumanMessage(content=f"""
        Patient Information:
        {patient_info}
        
        Skin Condition Description:
        {skin_condition}
        """)
    ])
    
    difficulty = assessment.content.split("\n")[0].strip()
    return difficulty

def create_dermatology_graph():
    # Initialize graph
    workflow = StateGraph(DermatologyState)
    
    # Initialize LLM using OpenAI directly
    llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OAI_API_KEY"),
    deployment_name="gpt-4o",
    api_version="2023-03-15-preview"
)
    

    # Define agent nodes
    def patient_intake_node(state: DermatologyState):
        """Node for collecting patient info"""
        messages = state.get("messages", [])
        current_msg = messages[-1].content if messages else "Start consultation"

        if "Start consultation" in current_msg:
            # Initial patient interaction
            cprint("\n👩‍⚕️ Nurse: Hello, I'll be gathering some information about your skin condition.", "green")
            skin_condition = input("\nPlease describe your main skin concerns: ")
            
            intake_questions = llm.invoke([
                SystemMessage(content="You are a nurse gathering patient information. Ask 3-4 key follow-up questions about the patient's skin condition."),
                HumanMessage(content=f"Patient's skin concerns: {skin_condition}")
            ])
            
            print("\nNurse: I need to ask you some additional questions:")
            print(intake_questions.content)
            answers = input("\nPlease provide your answers: ")
            
            # Compile information
            summary = llm.invoke([
                SystemMessage(content="You are a nurse. Create a comprehensive patient summary focused on the skin condition."),
                HumanMessage(content=f"""
                    Skin Condition: {skin_condition}
                    Additional Information: {answers}
                    
                    Please provide a well-structured summary for the dermatology team.
                """)
            ])
            
            state_update = {
                "messages": [AIMessage(content=summary.content)],
                "patient_info": {
                    "skin_concerns": skin_condition,
                    "details": answers,
                    "summary": summary.content
                },
                "skin_condition": skin_condition
            }
            
            # Determine difficulty level
            difficulty = determine_difficulty(state_update, llm)
            state_update["difficulty_level"] = difficulty
            cprint(f"\n📊 Case Difficulty Assessment: {difficulty}", "blue")
            
            return state_update
        return {}

    def medical_dermatologist_node(state: DermatologyState):
        """Medical dermatologist node focused on dermatological assessment"""
        patient_info = state["patient_info"]["summary"]
        skin_condition = state["skin_condition"]
        
        medical_opinion = llm.invoke([
            SystemMessage(content="""You are a medical dermatologist. Based on the patient's skin condition, provide:
            1. Detailed clinical assessment
            2. Differential diagnoses
            3. Recommended diagnostic tests if needed
            4. Initial treatment considerations"""),
            HumanMessage(content=f"""
            Patient Skin Condition:
            {skin_condition}
            
            Patient Information:
            {patient_info}
            """)
        ])
        
        cprint("\n👨‍⚕️ Medical Dermatologist Assessment:", "cyan")
        print(medical_opinion.content)
        
        return {
            "medical_dermatologist_consult": {
                "opinion": medical_opinion.content,
                "status": "completed"
            },
            "messages": [AIMessage(content=medical_opinion.content)]
        }

    def surgical_dermatologist_node(state: DermatologyState):
        """Surgical dermatologist node focused on surgical assessment"""
        patient_info = state["patient_info"]["summary"]
        skin_condition = state["skin_condition"]
        medical_opinion = state["medical_dermatologist_consult"]["opinion"]
        
        surgical_opinion = llm.invoke([
            SystemMessage(content="""You are a surgical dermatologist. Based on the patient's condition and medical assessment, provide:
            1. Surgical intervention assessment
            2. Procedural options and recommendations
            3. Risk-benefit analysis
            4. Surgical planning considerations"""),
            HumanMessage(content=f"""
            Patient Skin Condition: 
            {skin_condition}
            
            Medical Dermatologist's Assessment:
            {medical_opinion}
            
            Patient Information:
            {patient_info}
            """)
        ])
        
        cprint("\n👩‍⚕️ Surgical Dermatologist Assessment:", "cyan")
        print(surgical_opinion.content)
        
        return {
            "surgical_dermatologist_consult": {
                "opinion": surgical_opinion.content,
                "status": "completed"
            },
            "messages": [AIMessage(content=surgical_opinion.content)]
        }

    def dermatopathologist_node(state: DermatologyState):
        """Dermatopathologist node focused on tissue analysis and diagnosis"""
        patient_info = state["patient_info"]["summary"]
        skin_condition = state["skin_condition"]
        medical_opinion = state["medical_dermatologist_consult"]["opinion"]
        surgical_opinion = state["surgical_dermatologist_consult"]["opinion"]
        
        pathology_review = llm.invoke([
            SystemMessage(content="""You are a dermatopathologist. Based on the case information and specialist assessments, provide:
            1. Histopathological analysis
            2. Definitive diagnosis
            3. Disease staging if applicable
            4. Prognostic considerations
            5. Treatment recommendations based on pathological findings"""),
            HumanMessage(content=f"""
            Patient Skin Condition:
            {skin_condition}
            
            Medical Dermatologist's Assessment:
            {medical_opinion}
            
            Surgical Dermatologist's Assessment:
            {surgical_opinion}
            
            Patient Information:
            {patient_info}
            """)
        ])
        
        cprint("\n🔬 Dermatopathologist Assessment:", "cyan")
        print(pathology_review.content)
        
        return {
            "dermatopathologist_consult": {
                "opinion": pathology_review.content,
                "status": "completed"
            },
            "diagnosis": pathology_review.content.split("\n")[0].strip(),
            "treatment_plan": "\n".join(pathology_review.content.split("\n")[1:]).strip(),
            "messages": [AIMessage(content=pathology_review.content)]
        }

    def pharmacist_node(state: DermatologyState):
        """Pharmacist node focused on medication management"""
        patient_info = state["patient_info"]["summary"]
        skin_condition = state["skin_condition"]
        
        # Gather available specialist opinions
        specialist_opinions = f"""
        Medical Assessment: {state['medical_dermatologist_consult']['opinion']}
        """
        if state.get("surgical_dermatologist_consult", {}).get("opinion"):
            specialist_opinions += f"\nSurgical Assessment: {state['surgical_dermatologist_consult']['opinion']}"
        if state.get("dermatopathologist_consult", {}).get("opinion"):
            specialist_opinions += f"\nPathology Assessment: {state['dermatopathologist_consult']['opinion']}"
        
        prescription_review = llm.invoke([
            SystemMessage(content="""You are a pharmacist. Based on the specialist assessments, provide:
            1. Comprehensive medication plan
            2. Detailed usage instructions
            3. Potential drug interactions
            4. Side effect monitoring
            5. Important precautions
            6. Lifestyle recommendations"""),
            HumanMessage(content=f"""
            Patient Information:
            {patient_info}
            
            Skin Condition:
            {skin_condition}
            
            Specialist Assessments:
            {specialist_opinions}
            """)
        ])
        
        cprint("\n💊 Pharmacist's Recommendations:", "cyan")
        print(prescription_review.content)
        
        return {
            "prescription": prescription_review.content,
            "messages": [AIMessage(content=prescription_review.content)]
        }

    def route_by_medical_consult(state: DermatologyState) -> str:
        """Route based on medical consult and difficulty level"""
        difficulty = state["difficulty_level"]
        medical_status = state["medical_dermatologist_consult"]["status"]
        
        # If medical consult is completed, route based on difficulty
        if medical_status == "completed":
            if difficulty == "Basic":
                return "pharmacist"
            else:  # Intermediate or Advanced
                return "surgical_dermatologist"
        
        # If state doesn't have any updates, end the loop
        if not state.get("messages", []):
            return "medical_dermatologist"
        
        return END
            
    def route_by_completeness(state: DermatologyState) -> str:
        """Route by completeness and difficulty level"""
        difficulty = state["difficulty_level"]
        medical_status = state["medical_dermatologist_consult"]["status"]
        surgical_status = state["surgical_dermatologist_consult"]["status"]
        
        # Only proceed if both consultations are completed
        if medical_status == "completed" and surgical_status == "completed":
            if difficulty == "Advanced":
                return "dermatopathologist"
            elif difficulty == "Intermediate":
                return "pharmacist"
                
        # If state doesn't have any updates, continue with surgical
        if not state.get("messages", []):
            return "surgical_dermatologist"
            
        return END
    
    # Add nodes
    workflow.add_node("patient_intake", patient_intake_node)
    workflow.add_node("medical_dermatologist", medical_dermatologist_node)
    workflow.add_node("surgical_dermatologist", surgical_dermatologist_node)
    workflow.add_node("dermatopathologist", dermatopathologist_node)
    workflow.add_node("pharmacist", pharmacist_node)

    # Add edges with proper routing
    workflow.add_edge(START, "patient_intake")
    workflow.add_edge("patient_intake", "medical_dermatologist")
    
    # Medical dermatologist routing
    workflow.add_conditional_edges(
        "medical_dermatologist",
        route_by_medical_consult,
        {
            "medical_dermatologist": "medical_dermatologist",
            "surgical_dermatologist": "surgical_dermatologist",
            "pharmacist": "pharmacist",
            END: END
        }
    )
    
    # Surgical dermatologist routing
    workflow.add_conditional_edges(
        "surgical_dermatologist",
        route_by_completeness,
        {
            "surgical_dermatologist": "surgical_dermatologist",
            "dermatopathologist": "dermatopathologist",
            "pharmacist": "pharmacist",
            END: END
        }
    )
    
    workflow.add_edge("dermatopathologist", "pharmacist")
    workflow.add_edge("pharmacist", END)

    return workflow.compile()

def run_dermatology_consultation():
    """Run the dermatology consultation process"""
    graph = create_dermatology_graph()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content="Start consultation")],
        "patient_info": {},
        "skin_condition": "",
        "difficulty_level": "",
        "medical_dermatologist_consult": {"status": "pending"},
        "surgical_dermatologist_consult": {"status": "pending"},
        "dermatopathologist_consult": {"status": "pending"},
        "diagnosis": "",
        "treatment_plan": "",
        "prescription": ""
    }
    
    # Run workflow
    cprint("🏥 Starting dermatology consultation...", "blue")
    final_state = graph.invoke(initial_state)
    
    # Print results
    print("\n=== Final Dermatology Report ===")
    print(f"\nCase Difficulty Level: {final_state['difficulty_level']}")
    print("\nDiagnosis:")
    print(final_state["diagnosis"])
    print("\nTreatment Plan:")
    print(final_state["treatment_plan"])
    print("\nMedication and Care Instructions:")
    print(final_state["prescription"])
    
    return final_state

if __name__ == "__main__":
    run_dermatology_consultation()