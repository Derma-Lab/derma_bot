import os
from dotenv import load_dotenv
from autogen.agentchat import GroupChat, AssistantAgent, UserProxyAgent, GroupChatManager
from autogen.oai.openai_utils import config_list_from_dotenv

# ---------------------------
# Configure LLM Parameters
# ---------------------------

# Load environment variables
load_dotenv()

# Get Azure OpenAI API key and endpoint
azure_api_key = os.getenv("AZURE_OAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = "2024-02-15-preview"

# Load configuration from .env file
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": azure_api_key,
        "base_url": azure_endpoint,
        "api_type": "azure",
        "api_version": azure_api_version,
        "max_tokens": 500
    }
]

# GPT configuration settings
gpt_config = {
    "cache_seed": None,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 100,
}

# ---------------------------
# Define the Task
# ---------------------------

# Task description for all agents
task = """You are part of a dermatology clinic's AI system, each assigned a specific role. Adhere strictly to your designated responsibilities and communicate only with the permitted agents to ensure smooth workflow and effective problem-solving within the clinic"""

# ---------------------------
# Define Agents
# ---------------------------

# General Manager Agent
general_manager = AssistantAgent(
    name="General_Manager",
    llm_config=gpt_config,
    system_message=task + """I am general manager. 
    I am **ONLY** allowed to speak **immediately** after 'Patient'
    I oversee clinic operations and coordinate with `Doctor`, `Pharma_Person`, 'Patient'and `Human_Expert_User`. I will write very precise messages. I will decide when to pass the mic to patient or the human expert.
    """
)

# Doctor Agent
doctor = AssistantAgent(
    name="Doctor",
    system_message=task + """ I am the Doctor. 
    I am **ONLY** allowed to speak **immediately** after `General_Manager` or `Human_Expert_User`.
    I handle patient diagnoses, treatment plans, and communicate with `Pharma_Person` for medications.
    """,
    llm_config=gpt_config,
)

# Pharma Person Agent
pharma_person = AssistantAgent(
    name="Pharma_Person",
    system_message=task + """ I am the Pharma Person. I am only called when the doctor asks for a prescription. 
    I am **ONLY** allowed to speak **immediately** after `Doctor` or `General_Manager`.
    I manage medication inventory, prescriptions, and liaise with pharmacies.
    """,
    llm_config=gpt_config,
)


# Patient Agent
# Human Expert User Proxy Agent
patient_proxy  = UserProxyAgent(
    name="Patient",
    system_message=task + """
    I represent the human input for the Patient.
    Respond to general manager's with your disease, and talk to generral manager back and forth.
    """,
    code_execution_config=False,
    human_input_mode="ALWAYS",  # Enable human input
    llm_config=False,  # Disable LLM for this agent
)


# Human Expert User Proxy Agent
human_expert_proxy = UserProxyAgent(
    name="Human_Expert_User",
    system_message=task + """
    I represent the human input for the Human Expert.
    Only respond when prompted by `Doctor` or `General_Manager`.
    """,
    code_execution_config=False,
    human_input_mode="ALWAYS",  # Enable human input
    llm_config=False,  # Disable LLM for this agent
)

# User Proxy Agent
user_proxy = UserProxyAgent(
    name="User",
    system_message=task + """
    I initiate the conversation and represent the clinic's user input.
    Never select me as a speaker.
    """,
    code_execution_config=False,
    human_input_mode="NEVER",
    llm_config=False,
)

# ---------------------------
# Define the Graph (FSM Transitions)
# ---------------------------

# Define transition rules between agents
graph_dict = {
    user_proxy: [general_manager],
    general_manager: [doctor, pharma_person, patient_proxy, human_expert_proxy],
    patient_proxy: [general_manager],
    doctor: [general_manager],
    pharma_person: [general_manager],
    human_expert_proxy: [general_manager],
}

# ---------------------------
# Define GroupChat and GroupChatManager
# ---------------------------

# List of all agents
agents = [user_proxy, general_manager, doctor, pharma_person, patient_proxy, human_expert_proxy]

# Create the GroupChat with defined agents and transition rules
group_chat = GroupChat(
    agents=agents,
    messages=[],  # Initial messages can be empty or predefined
    max_round=50,  # Maximum number of interaction rounds
    allowed_or_disallowed_speaker_transitions=graph_dict,
    allow_repeat_speaker=None,  # Allow or disallow agents to repeat
    speaker_transitions_type="allowed"  # Type of transition rules
)

# Create the GroupChatManager with termination condition
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=gpt_config,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)

# ---------------------------
# Initiate the Chat
# ---------------------------

def initiate_chat():
    while True:
        # Start the chat by having the User proxy send an initial message
        user_proxy.initiate_chat(
            manager,
            message="""Initiate clinic operations. General manager you shall be speaking with the patient who is a dermatologist as well. Ask the patient about their disease and talk to {patient_proxy}.""",
            clear_history=True
        )

        # Run the GroupChatManager
        manager.run()

        # After termination, prompt the user to start a new conversation or exit
        user_input = input("Do you want to start a new conversation? (yes/no): ").strip().lower()
        if user_input not in ["yes", "y"]:
            print("Terminating the FSM. Goodbye!")
            break
        else:
            print("Starting a new conversation...\n")

# ---------------------------
# Initiate the Conversation Loop
# ---------------------------

if __name__ == "__main__":
    initiate_chat()