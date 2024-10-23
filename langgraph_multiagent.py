import os
from typing import Annotated, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from typing import Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
import operator

def print_separator():
    print("\n" + "="*50 + "\n")

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OAI_API_KEY"),
    deployment_name="gpt-4o",
    api_version="2023-03-15-preview"
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Define the questions directly to ensure they're shown
RECEPTIONIST_QUESTIONS = """1. Patient's age and gender
2. Duration of skin condition
3. Location and appearance of condition
4. Previous treatments tried
5. Any allergies or medical history"""

def receptionist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Receptionist node that displays predefined questions"""
    print("\n=== Receptionist's Questions ===")
    print(RECEPTIONIST_QUESTIONS)
    print_separator()
    
    print("Please answer all questions above:")
    patient_answers = input("Patient: ").strip()
    
    return {
        "messages": [
            HumanMessage(content=f"Patient's initial description: {state['messages'][0].content}"),
            HumanMessage(content=f"Patient's answers to questions: {patient_answers}")
        ],
        "next": "dermatologist"
    }

def dermatologist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Dermatologist node to provide diagnosis"""
    prompt = """You are a dermatologist. Based on the patient's information, provide ONLY:
1. Brief symptom summary
2. Most likely diagnosis
3. Confidence level (percentage)

Be concise and do NOT ask additional questions."""
    
    messages = [
        SystemMessage(content=prompt),
        *state["messages"]
    ]
    
    response = llm.invoke(messages)
    
    print("\n=== Dermatologist's Assessment ===")
    print(response.content)
    print_separator()
    
    return {
        "messages": [AIMessage(content=response.content)],
        "next": "pharmacist"
    }

def pharmacist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pharmacist node to provide prescription"""
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
    
    messages = [
        SystemMessage(content=prompt),
        *state["messages"]  # Include all context
    ]
    
    response = llm.invoke(messages)
    
    print("\n=== Prescription Details ===")
    print(response.content)
    print_separator()
    
    return {
        "messages": [AIMessage(content=response.content)],
        "next": "END"
    }

# Set up workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("receptionist", receptionist_node)
workflow.add_node("dermatologist", dermatologist_node)
workflow.add_node("pharmacist", pharmacist_node)

# Add edges
workflow.add_edge(START, "receptionist")

# Add conditional edges
workflow.add_conditional_edges(
    "receptionist",
    lambda x: x["next"],
    {
        "dermatologist": "dermatologist",
        "END": END
    }
)

workflow.add_conditional_edges(
    "dermatologist",
    lambda x: x["next"],
    {
        "pharmacist": "pharmacist",
        "END": END
    }
)

workflow.add_conditional_edges(
    "pharmacist",
    lambda x: x["next"],
    {
        "END": END
    }
)

# Compile graph
graph = workflow.compile()

def run_consultation():
    """Run the complete consultation process"""
    print("\n=== Dermatology Consultation System ===")
    patient_input = input("\nPlease describe your skin concern:\nPatient: ").strip()
    
    if not patient_input:
        print("No input provided.")
        return
    
    print_separator()
    print("Starting consultation process...")
    
    try:
        # Initial state
        state = {
            "messages": [HumanMessage(content=patient_input)],
            "next": "receptionist"
        }
        
        # Process through nodes
        for current_state in graph.stream(
            state,
            {"configurable": {"thread_id": "consultation-1"}}
        ):
            state = current_state
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please try again.")
    
    print("\nNOTE: This is an AI-generated consultation. Please consult with a real healthcare provider for accurate medical advice.")

if __name__ == "__main__":
    try:
        run_consultation()
    except KeyboardInterrupt:
        print("\n\nConsultation interrupted.")
    except Exception as e:
        print(f"\nSystem error: {str(e)}")
    finally:
        print_separator()
        print("System closed.")
