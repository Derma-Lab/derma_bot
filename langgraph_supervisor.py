import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import Annotated, Literal, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel
import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

# Load environment variables
load_dotenv()

# Azure OpenAI setup
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OAI_API_KEY"),
    deployment_name="gpt-4o",
    api_version="2023-03-15-preview"
)

def agent_node(state: Dict[str, Any], runnable: RunnableWithMessageHistory, name: str) -> Dict[str, Any]:
    messages = state['messages']
    human_message = messages[-1].content if messages else ""
    response = runnable.invoke({"input": human_message}, {"configurable": {"session_id": name}})
    return {
        "messages": state['messages'] + [AIMessage(content=response.content, name=name)],
        "consultation_state": state.get('consultation_state', 'initial_assessment')
    }

members = ["GeneralManager", "Dermatologist", "PharmaceuticalAgent", "HumanExpert"]
system_prompt = (
    "You are a supervisor tasked with managing a dermatology consultation between the"
    " following participants: {members}. Given the patient's request, ongoing"
    " consultation, and the current consultation state, respond with the participant to act next."
    " The consultation should follow this general flow:\n"
    "1. General Manager gathers initial information.\n"
    "2. Dermatologist asks questions and provides initial assessment.\n"
    "3. When the Dermatologist has enough information, they should provide a diagnosis.\n"
    "4. After diagnosis, route to Human Expert for approval.\n"
    "5. If Human Expert approves, route to Pharmaceutical Agent for prescription.\n"
    "6. If Human Expert requests changes, route back to Dermatologist.\n"
    "7. End the consultation after Pharmaceutical Agent provides prescription.\n"
    "If more information is needed from the patient at any point, respond with PATIENT_INPUT."
)

options = ["PATIENT_INPUT"] + members

class RouteResponse(BaseModel):
    next: Literal["PATIENT_INPUT", "GeneralManager", "Dermatologist", "PharmaceuticalAgent", "HumanExpert", "END"]

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Current consultation state: {consultation_state}\nLast message: {input}\nWho should act next?"),
]).partial(options=str(options), members=", ".join(members))

def format_messages_for_prompt(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    return [{"role": m.type, "content": m.content, "name": getattr(m, 'name', None)} for m in messages]


def supervisor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    
    # Limit the number of messages to the last N messages (e.g., 5)
    max_history_length = 5
    history = format_messages_for_prompt(state['messages'][-max_history_length:])  # Keep only the last N messages
    
    gm_rounds = state.get('gm_rounds', 0)
    derm_rounds = state.get('derm_rounds', 0)
    consultation_state = state.get('consultation_state', 'initial_assessment')
    
    result = supervisor_chain.invoke({
        "history": history,
        "input": state['messages'][-1].content,
        "consultation_state": consultation_state
    })
    
    next_action = result.next
    new_state = consultation_state

    if next_action == "GeneralManager":
        if gm_rounds < 3:
            gm_rounds += 1
        else:
            next_action = "Dermatologist"
            new_state = 'dermatologist_assessment'
    elif next_action == "Dermatologist":
        if consultation_state == 'initial_assessment':
            new_state = 'dermatologist_assessment'
        if derm_rounds < 3:
            derm_rounds += 1
        else:
            next_action = "HumanExpert"
            new_state = 'awaiting_expert_approval'
    elif next_action == "HumanExpert":
        new_state = 'awaiting_expert_approval'
    elif next_action == "PharmaceuticalAgent":
        new_state = 'prescription'
    elif next_action == "END":
        new_state = 'consultation_complete'

    return {
        "messages": state['messages'] + [AIMessage(content=f"Next action: {next_action}", name="Supervisor")],
        "next": next_action,
        "consultation_state": new_state,
        "gm_rounds": gm_rounds,
        "derm_rounds": derm_rounds
    }

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    consultation_state: str
    gm_rounds: int
    derm_rounds: int

# Define agent prompts
general_manager_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a general manager for a dermatology clinic. Your role is to gather initial information from patients and route them to the appropriate specialist. Ask relevant questions about the patient's skin condition, symptoms, and medical history."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

dermatologist_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a dermatologist. Analyze the patient's information and provide an initial diagnosis or assessment. If you need more information, ask specific questions. When you have enough information, provide a clear diagnosis."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

pharmaceutical_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a pharmaceutical agent specializing in dermatological treatments. Provide information on appropriate medications or treatments based on the dermatologist's diagnosis."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Create runnables with message history
def create_runnable(prompt: ChatPromptTemplate) -> RunnableWithMessageHistory:
    chain = prompt | llm
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="history",
    )

general_manager_runnable = create_runnable(general_manager_prompt)
general_manager_node = functools.partial(agent_node, runnable=general_manager_runnable, name="GeneralManager")

dermatologist_runnable = create_runnable(dermatologist_prompt)
dermatologist_node = functools.partial(agent_node, runnable=dermatologist_runnable, name="Dermatologist")

pharmaceutical_agent_runnable = create_runnable(pharmaceutical_agent_prompt)
pharmaceutical_node = functools.partial(agent_node, runnable=pharmaceutical_agent_runnable, name="PharmaceuticalAgent")

def human_expert_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\nHuman Expert Review:")
    print("Last message:", state["messages"][-1].content)
    decision = input("Approve (a) or Request Changes (r): ").lower()
    if decision == 'a':
        return {
            "messages": state['messages'] + [HumanMessage(content="Diagnosis approved.", name="HumanExpert")],
            "consultation_state": "expert_approved"
        }
    else:
        feedback = input("Enter feedback for changes: ")
        return {
            "messages": state['messages'] + [HumanMessage(content=f"Changes requested: {feedback}", name="HumanExpert")],
            "consultation_state": "expert_requested_changes"
        }

def get_patient_input(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\nPatient Input Required:")
    user_input = input("Please provide more information or ask a question: ")
    return {
        "messages": state['messages'] + [HumanMessage(content=user_input)],
        "consultation_state": state['consultation_state']
    }

workflow = StateGraph(AgentState)
workflow.add_node("GeneralManager", general_manager_node)
workflow.add_node("Dermatologist", dermatologist_node)
workflow.add_node("PharmaceuticalAgent", pharmaceutical_node)
workflow.add_node("HumanExpert", human_expert_node)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("patient_input", get_patient_input)

for member in members + ["patient_input"]:
    workflow.add_edge(member, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "GeneralManager": "GeneralManager",
        "Dermatologist": "Dermatologist",
        "PharmaceuticalAgent": "PharmaceuticalAgent",
        "HumanExpert": "HumanExpert",
        "PATIENT_INPUT": "patient_input",
        "END": END
    }
)
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

# Example usage
initial_message = "I have a red, itchy rash on my arm that appeared yesterday. What should I do?"
state = {
    "messages": [HumanMessage(content=initial_message)],
    "next": START,
    "consultation_state": "initial_assessment",
    "gm_rounds": 0,
    "derm_rounds": 0
}

while True:
    for s in graph.stream(state, {"recursion_limit": 100}):
        print(s)
        print("----")
        if 'next' in s and s['next'] == END:
            print("Consultation ended.")
            exit(0)
        state = s  # Update the state

    # Reset the 'next' key in the state for the next iteration
    state.pop("next", None)
