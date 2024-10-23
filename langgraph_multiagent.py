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

# Initialize LLM with debug print
api_key = os.getenv("AZURE_OAI_API_KEY")
print(f"DEBUG: API Key present: {bool(api_key)}")

llm = AzureChatOpenAI(
    api_key=api_key,
    deployment_name="gpt-4o",
    api_version="2023-03-15-preview"
)

# Test LLM
print("DEBUG: Testing LLM...")
test_response = llm.invoke("Test message")
print(f"DEBUG: LLM test response received: {test_response.content}")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str

def receptionist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Receptionist node with debug prints"""
    print("DEBUG: Receptionist node activated")
    messages = [
        SystemMessage(content="You are a medical receptionist. Gather initial information about the patient's skin concern."),
        *state["messages"]
    ]
    
    try:
        response = llm.invoke(messages)
        print(f"DEBUG: Receptionist response: {response.content}")
        
        return {
            "messages": [AIMessage(content=response.content)],
            "current_agent": "receptionist"
        }
    except Exception as e:
        print(f"ERROR in receptionist node: {str(e)}")
        raise

def dermatologist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Dermatologist node with debug prints"""
    print("DEBUG: Dermatologist node activated")
    messages = [
        SystemMessage(content="You are a dermatologist. Assess the patient's skin condition and provide initial recommendations."),
        *state["messages"]
    ]
    
    try:
        response = llm.invoke(messages)
        print(f"DEBUG: Dermatologist response: {response.content}")
        
        return {
            "messages": [AIMessage(content=response.content)],
            "current_agent": "dermatologist"
        }
    except Exception as e:
        print(f"ERROR in dermatologist node: {str(e)}")
        raise

def pharmacist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pharmacist node with debug prints"""
    print("DEBUG: Pharmacist node activated")
    messages = [
        SystemMessage(content="You are a pharmacist. Provide specific product recommendations and usage instructions."),
        *state["messages"]
    ]
    
    try:
        response = llm.invoke(messages)
        print(f"DEBUG: Pharmacist response: {response.content}")
        
        return {
            "messages": [AIMessage(content=response.content)],
            "current_agent": "pharmacist"
        }
    except Exception as e:
        print(f"ERROR in pharmacist node: {str(e)}")
        raise

def should_continue(state: AgentState) -> str:
    """Determine next step with debug prints"""
    current = state.get("current_agent", "")
    print(f"DEBUG: Current agent in should_continue: {current}")
    
    if current == "receptionist":
        return "dermatologist"
    elif current == "dermatologist":
        return "pharmacist"
    return "END"

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
    should_continue,
    {
        "dermatologist": "dermatologist",
        "pharmacist": "pharmacist",
        "END": END
    }
)

workflow.add_conditional_edges(
    "dermatologist",
    should_continue,
    {
        "pharmacist": "pharmacist",
        "END": END
    }
)

workflow.add_conditional_edges(
    "pharmacist",
    lambda x: "END",
    {
        "END": END
    }
)

# Compile graph
graph = workflow.compile()

def process_consultation():
    """Process consultation with extensive debug information"""
    print("\n=== Dermatology Clinic Consultation System ===")
    patient_input = input("\nPlease describe your skin concern:\nPatient: ").strip()
    
    if not patient_input:
        print("No input provided.")
        return
    
    print_separator()
    print("Starting consultation process...")
    
    # Initial state
    state = {
        "messages": [HumanMessage(content=patient_input)],
        "current_agent": "start"
    }
    print("DEBUG: Initial state created")
    
    try:
        for current_state in graph.stream(
            state,
            {"configurable": {"thread_id": "consultation-1"}}
        ):
            print("DEBUG: Processing new state")
            print(f"DEBUG: Current agent: {current_state.get('current_agent')}")
            
            if current_state.get("messages"):
                last_message = current_state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    agent = current_state.get("current_agent", "Unknown")
                    print(f"\n=== {agent.title()}'s Response ===")
                    print(last_message.content)
                    print_separator()
    
    except Exception as e:
        print(f"Error in consultation: {str(e)}")
        print(f"Error type: {type(e)}")
    
    print("Consultation complete.")

if __name__ == "__main__":
    try:
        process_consultation()
    except KeyboardInterrupt:
        print("\n\nConsultation interrupted.")
    except Exception as e:
        print(f"\nSystem error: {str(e)}")
        print(f"Error type: {type(e)}")
    finally:
        print_separator()
        print("System closed.")
