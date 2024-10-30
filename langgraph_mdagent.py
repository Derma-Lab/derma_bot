import os 
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, Graph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import operator
from dataclasses import dataclass
from termcolor import cprint

# Define state schema
class DermatologyState(TypedDict):
    """graph state definition"""
    messages: Annotated[list, operator.add]  # chat history
    patient_info: dict  # patient info
    skin_condition: str  # skin condition
    medical_dermatologist_consult: dict  # medical dermatologist view
    surgical_dermatologist_consult: dict  # surgical dermatologist view
    dermatopathologist_consult: dict  # dermatopathologist view
    diagnosis: str  # diagnosis result
    treatment_plan: str  # treatment plan
    prescription: str  # prescription

def create_dermatology_graph():
    # Initialize graph
    workflow = StateGraph(DermatologyState)
    
    # Initialize LLM
    llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OAI_API_KEY"),
            azure_endpoint='https://derma-lab-test.openai.azure.com',
            api_version="2023-03-15-preview"
        )

    # Define agent nodes
    def patient_intake_node(state: DermatologyState):
        """node of collecting patient info"""
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
            
            return {
                "messages": [AIMessage(content=summary.content)],
                "patient_info": {
                    "skin_concerns": skin_condition,
                    "details": answers,
                    "summary": summary.content
                },
                "skin_condition": skin_condition
            }
        return {}

    def medical_dermatologist_node(state: DermatologyState):
        """node of medical dermatologist"""
        patient_info = state["patient_info"]["summary"]
        skin_condition = state["skin_condition"]
        
        medical_opinion = llm.invoke([
            SystemMessage(content="""You are a medical dermatologist. Based on the patient's skin condition, provide:
            1. Initial assessment
            2. Potential diagnoses
            3. Recommended next steps
            """),
            HumanMessage(content=f"Patient Skin Condition:\n{skin_condition}\n\nPatient Information:\n{patient_info}")
        ])
        
        return {
            "medical_dermatologist_consult": {
                "opinion": medical_opinion.content,
                "status": "completed"
            },
            "messages": [AIMessage(content=medical_opinion.content)]
        }

    def surgical_dermatologist_node(state: DermatologyState):
        """surgical dermatologist node"""
        patient_info = state["patient_info"]["summary"]
        skin_condition = state["skin_condition"]
        
        if state["medical_dermatologist_consult"]["status"] == "completed":
            surgical_opinion = llm.invoke([
                SystemMessage(content="""You are a surgical dermatologist. Based on the patient's skin condition and the medical dermatologist's assessment, provide:
                1. Evaluation of any surgical or procedural options
                2. Potential risks and benefits
                3. Recommended treatment plan
                """),
                HumanMessage(content=f"""
                Patient Skin Condition: 
                {skin_condition}
                
                Medical Dermatologist's Assessment:
                {state["medical_dermatologist_consult"]["opinion"]}
                
                Patient Information:
                {patient_info}
                """)
            ])
            
            return {
                "surgical_dermatologist_consult": {
                    "opinion": surgical_opinion.content,
                    "status": "completed"
                },
                "messages": [AIMessage(content=surgical_opinion.content)]
            }
        else:
            return {}

    def dermatopathologist_node(state: DermatologyState):
        """dermatopathologist node"""
        patient_info = state["patient_info"]["summary"]
        skin_condition = state["skin_condition"]
        
        if state["medical_dermatologist_consult"]["status"] == "completed" and \
           state["surgical_dermatologist_consult"]["status"] == "completed":
            
            pathology_review = llm.invoke([
                SystemMessage(content="""You are a dermatopathologist. Based on the patient's skin condition and the assessments from the medical and surgical dermatologists, provide:
                1. Pathological analysis
                2. Definitive diagnosis
                3. Recommended treatment plan
                """),
                HumanMessage(content=f"""
                Patient Skin Condition:
                {skin_condition}
                
                Medical Dermatologist's Assessment:
                {state["medical_dermatologist_consult"]["opinion"]}
                
                Surgical Dermatologist's Assessment:
                {state["surgical_dermatologist_consult"]["opinion"]}
                
                Patient Information:
                {patient_info}
                """)
            ])
            
            return {
                "dermatopathologist_consult": {
                    "opinion": pathology_review.content,
                    "status": "completed"
                },
                "diagnosis": pathology_review.content.split("\n")[0].strip(),
                "treatment_plan": "\n".join(pathology_review.content.split("\n")[1:]).strip(),
                "messages": [AIMessage(content=pathology_review.content)]
            }
        else:
            return {}

    def pharmacist_node(state: DermatologyState):
        """pharmacist node"""
        patient_info = state["patient_info"]["summary"]
        treatment_plan = state["treatment_plan"]
        
        prescription_review = llm.invoke([
            SystemMessage(content="""You are a pharmacist. Review the dermatology treatment plan and provide:
            1. Medication instructions
            2. Potential drug interactions
            3. Side effect warnings
            4. Lifestyle recommendations"""),
            HumanMessage(content=f"""
            Patient Information:
            {patient_info}
            
            Treatment Plan:
            {treatment_plan}
            """)
        ])
        
        return {
            "prescription": prescription_review.content,
            "messages": [AIMessage(content=prescription_review.content)]
        }

    # Add nodes
    workflow.add_node("patient_intake", patient_intake_node)
    workflow.add_node("medical_dermatologist", medical_dermatologist_node)
    workflow.add_node("surgical_dermatologist", surgical_dermatologist_node)
    workflow.add_node("dermatopathologist", dermatopathologist_node)
    workflow.add_node("pharmacist", pharmacist_node)

    # Define routing logic
    def route_by_medical_consult(state: DermatologyState) -> str:
        """route"""
        if state["medical_dermatologist_consult"]["status"] == "completed":
            return "surgical_dermatologist"
        return "medical_dermatologist"

    def route_by_completeness(state: DermatologyState) -> str:
        """route by completeness"""
        if state["medical_dermatologist_consult"]["status"] == "completed" and \
           state["surgical_dermatologist_consult"]["status"] == "completed":
            return "dermatopathologist"
        return "surgical_dermatologist"

    # Add edges
    workflow.add_edge("patient_intake", "medical_dermatologist")
    workflow.add_conditional_edges(
        "medical_dermatologist",
        route_by_medical_consult,
        {
            "medical_dermatologist": "medical_dermatologist",
            "surgical_dermatologist": "surgical_dermatologist"
        }
    )
    workflow.add_conditional_edges(
        "surgical_dermatologist",
        route_by_completeness,
        {
            "surgical_dermatologist": "surgical_dermatologist",
            "dermatopathologist": "dermatopathologist"
        }
    )
    workflow.add_edge("dermatopathologist", "pharmacist")

    return workflow.compile()

def run_dermatology_consultation():
    """run process"""
    graph = create_dermatology_graph()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content="Start consultation")],
        "patient_info": {},
        "skin_condition": "",
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
    print("\nDiagnosis:")
    print(final_state["diagnosis"])
    print("\nTreatment Plan:")
    print(final_state["treatment_plan"])
    print("\nMedication and Care Instructions:")
    print(final_state["prescription"])
    
    return final_state

if __name__ == "__main__":
    run_dermatology_consultation()