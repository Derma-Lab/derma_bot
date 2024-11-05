import os
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, Graph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import operator
from termcolor import cprint
from dataclasses import dataclass
from pptree import Node, print_tree
import random

class DermState(TypedDict):
    """State for dermatology consultation flow"""
    messages: Annotated[List, operator.add]  # Chat history 
    patient_data: dict  # Patient information
    complexity: str  # Complexity level
    members: List  # Medical team members
    opinions: Dict  # Specialist opinions 
    interaction_logs: Dict  # Discussion logs
    final_diagnosis: str
    treatment_plan: str

def create_dermatology_mdagents():
    # Initialize workflow graph
    workflow = StateGraph(DermState)
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OAI_API_KEY"),
        deployment_name="gpt-4o",
        api_version="2023-03-15-preview"
    )

    def assess_complexity(state: DermState):
        """Determines case complexity and required team structure"""
        patient_info = state["patient_data"]
        assessment = llm.invoke([
            SystemMessage(content="""You are a dermatology triage specialist. Classify case complexity as:
            - Low: Standard conditions manageable by one dermatologist (e.g., mild eczema, acne, common rashes)
            - Moderate: Complex cases needing multi-specialist collaboration (e.g., severe psoriasis, unusual rashes)
            - High: Severe cases requiring coordinated multi-team approach (e.g., severe drug reactions, complex autoimmune conditions)
            
            Provide only complexity level and brief rationale."""),
            HumanMessage(content=f"Patient Information: {patient_info}")
        ])
        
        complexity = assessment.content.split("\n")[0].strip()
        cprint(f"\n📊 Case Complexity Assessment: {complexity}", "blue")
        
        return {
            "complexity": complexity,
            "messages": [AIMessage(content=assessment.content)]
        }

    def single_dermatologist(state: DermState):
        """Handle low complexity cases with a single dermatologist"""
        patient_info = state["patient_data"]
        
        cprint("\n👨‍⚕️ Primary dermatologist assessment...", "yellow")
        
        assessment = llm.invoke([
            SystemMessage(content="""You are a dermatologist handling a straightforward case.
            Provide a complete assessment in the following format:
            
            DIAGNOSIS:
            [Clear diagnosis with key findings]
            
            TREATMENT PLAN:
            [Specific treatment recommendations]"""),
            HumanMessage(content=f"Patient Information: {patient_info}")
        ])
        
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
            
        return {
            "final_diagnosis": diagnosis,
            "treatment_plan": treatment,
            "messages": [AIMessage(content=assessment.content)]
        }

    def recruit_specialists(state: DermState):
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
        
        cprint("\n🔄 Recruiting specialist team...", "blue")
        
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
            
        cprint(f"\n✅ Recruited {len(specialists)} team members", "green")
        for specialist in specialists:
            cprint(f"  • {specialist['role']}", "green")
        
        return {
            "members": specialists,
            "messages": [AIMessage(content=recruitment.content)]
        }

    def facilitate_discussion(state: DermState):
        """Manages specialist discussions and opinion gathering"""
        opinions = state.get("opinions", {})
        specialists = state["members"]
        patient_info = state["patient_data"]
        complexity = state["complexity"].lower()
        interaction_logs = {}
        
        cprint("\n💬 Starting team discussion...", "yellow")
        
        # Gather initial opinions
        for specialist in specialists:
            opinion = llm.invoke([
                SystemMessage(content=f"""You are a {specialist['role']}. 
                Provide a focused assessment including:
                1. Key observations from your specialty perspective
                2. Diagnosis considerations
                3. Treatment recommendations
                
                Format your response with clear DIAGNOSIS: and TREATMENT PLAN: sections."""),
                HumanMessage(content=f"Patient Information: {patient_info}")
            ])
            opinions[specialist["role"]] = opinion.content
            cprint(f"  • {specialist['role']} assessment completed", "cyan")
            
        # Facilitate inter-specialist discussion if needed
        if "high" in complexity:
            cprint("\n🔄 Starting multi-team consultation...", "yellow")
            for i in range(3):  # Maximum 3 discussion rounds
                round_log = {}
                for specialist in specialists:
                    other_opinions = {k:v for k,v in opinions.items() if k != specialist["role"]}
                    
                    response = llm.invoke([
                        SystemMessage(content=f"""You are a {specialist['role']}.
                        Review other specialists' opinions and provide:
                        1. Points of agreement/disagreement
                        2. Questions for specific specialists
                        3. Updated assessment based on discussion"""),
                        HumanMessage(content=f"""
                        Other Opinions: {other_opinions}
                        Patient Information: {patient_info}""")
                    ])
                    round_log[specialist["role"]] = response.content
                    opinions[specialist["role"]] = response.content
                
                interaction_logs[f"round_{i+1}"] = round_log
                cprint(f"  • Consultation round {i+1} completed", "cyan")
                
        return {
            "opinions": opinions,
            "interaction_logs": interaction_logs,
            "messages": [AIMessage(content=str(opinions))]
        }

    def synthesize_decision(state: DermState):
        """Synthesizes specialist inputs into final decision"""
        opinions = state["opinions"]
        complexity = state["complexity"]
        
        cprint("\n📋 Synthesizing final decision...", "yellow")
        
        final_decision = llm.invoke([
            SystemMessage(content="""You are the lead dermatologist.
            Synthesize all specialist inputs to provide:
            
            DIAGNOSIS:
            [Detailed diagnosis with key findings]
            
            TREATMENT PLAN:
            [Comprehensive treatment approach]"""),
            HumanMessage(content=f"""
            Case Complexity: {complexity}
            Specialist Opinions: {opinions}""")
        ])
        
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
            
        return {
            "final_diagnosis": diagnosis,
            "treatment_plan": treatment,
            "messages": [AIMessage(content=final_decision.content)]
        }

    # Add nodes
    workflow.add_node("assess_complexity", assess_complexity)
    workflow.add_node("single_dermatologist", single_dermatologist)
    workflow.add_node("recruit_specialists", recruit_specialists)
    workflow.add_node("facilitate_discussion", facilitate_discussion)
    workflow.add_node("synthesize_decision", synthesize_decision)

    def route_by_complexity(state: DermState) -> str:
        """Routes to appropriate path based on complexity level"""
        complexity = state["complexity"].lower()
        
        if "low" in complexity:
            return "single_dermatologist"
        
        # For moderate/high complexity cases
        if not state.get("members"):
            return "recruit_specialists"
        elif not state.get("opinions"):
            return "facilitate_discussion"
        elif not state.get("final_diagnosis"):
            return "synthesize_decision"
            
        return END

    # Basic linear paths
    workflow.add_edge(START, "assess_complexity")
    workflow.add_edge("single_dermatologist", END)

    # Conditional edges based on complexity
    workflow.add_conditional_edges(
        "assess_complexity",
        route_by_complexity,
        {
            "single_dermatologist": "single_dermatologist",
            "recruit_specialists": "recruit_specialists",
            END: END
        }
    )

    # Paths for moderate/high complexity cases
    workflow.add_edge("recruit_specialists", "facilitate_discussion")
    workflow.add_edge("facilitate_discussion", "synthesize_decision")
    workflow.add_edge("synthesize_decision", END)

    return workflow.compile()

def run_dermatology_consultation(patient_info: dict):
    """Run complete dermatology consultation"""
    workflow = create_dermatology_mdagents()
    
    initial_state = {
        "messages": [],
        "patient_data": patient_info,
        "complexity": "",
        "members": [],
        "opinions": {},
        "interaction_logs": {},
        "final_diagnosis": "",
        "treatment_plan": ""
    }
    
    cprint("\n🏥 Starting dermatology consultation...", "blue")
    final_state = workflow.invoke(initial_state)
    
    # Print comprehensive final report
    print("\n=== Final Dermatology Report ===")
    print(f"\nCase Complexity: {final_state['complexity']}")
    
    if final_state['members']:
        print("\nConsulting Team:")
        for member in final_state['members']:
            print(f"- {member['role']}")
            
    print("\nDiagnosis:")
    print(final_state['final_diagnosis'])
    
    print("\nTreatment Plan:")
    print(final_state['treatment_plan'])
    
    if final_state['interaction_logs']:
        print("\nKey Discussion Points:")
        for round_name, round_log in final_state['interaction_logs'].items():
            print(f"\n{round_name.upper()}:")
            for specialist, comments in round_log.items():
                print(f"{specialist}: {comments[:200]}...")
    
    return final_state

if __name__ == "__main__":
    # Example usage
    moderate_case = {
    "symptoms": "Multiple round, scaly patches with raised borders on trunk and limbs, some with central clearing",
    "duration": "2 months, gradually spreading",
    "history": "Previously diagnosed with mild psoriasis 5 years ago, type 2 diabetes",
    "age": 55,
    "allergies": "Penicillin",
    "medications": "Metformin for diabetes",
    "recent_changes": "Started immunosuppressant therapy for rheumatoid arthritis 3 months ago",
    "additional_symptoms": "Mild joint pain, some nail changes",
    "previous_treatments": "Over-the-counter antifungal cream with no improvement",
    "impact": "Affecting sleep and daily activities",
    "family_history": "Mother had autoimmune condition"
    }
    
    result = run_dermatology_consultation(moderate_case)