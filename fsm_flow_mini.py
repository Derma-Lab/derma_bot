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
    system_message=task,
    description="""I am **ONLY** allowed to speak **immediately** after `User`.
    I oversee clinic operations and coordinate with `Doctor`, `Pharma_Person`, and `Human_Expert_User`.
    """
)

# Doctor Agent
doctor = AssistantAgent(
    name="Doctor",
    system_message=task,
    llm_config=gpt_config,
    description="""I am **ONLY** allowed to speak **immediately** after `General_Manager` or `Human_Expert_User`.
    I handle patient diagnoses, treatment plans, and communicate with `Pharma_Person` for medications.
    """
)

# Pharma Person Agent
pharma_person = AssistantAgent(
    name="Pharma_Person",
    system_message=task,
    llm_config=gpt_config,
    description="""I am **ONLY** allowed to speak **immediately** after `Doctor` or `General_Manager`.
    I manage medication inventory, prescriptions, and liaise with pharmacies.
    """
)

# Human Expert User Proxy Agent
human_expert_proxy = UserProxyAgent(
    name="Human_Expert_User",
    system_message=task,
    code_execution_config=False,
    human_input_mode="ALWAYS",  # Enable human input
    llm_config=False,  # Disable LLM for this agent
    description="""
    I represent the human input for the Human Expert.
    Only respond when prompted by `Doctor` or `General_Manager`.
    """
)

# User Proxy Agent
user_proxy = UserProxyAgent(
    name="User",
    system_message=task,
    code_execution_config=False,
    human_input_mode="NEVER",
    llm_config=False,
    description="""
    I initiate the conversation and represent the clinic's user input.
    Never select me as a speaker.
    """
)

# ---------------------------
# Define the Graph (FSM Transitions)
# ---------------------------

# Define transition rules between agents
graph_dict = {
    user_proxy: [general_manager],
    general_manager: [doctor, pharma_person, human_expert_proxy],
    doctor: [general_manager],
    pharma_person: [general_manager],
    human_expert_proxy: [general_manager],
}

# ---------------------------
# Define GroupChat and GroupChatManager
# ---------------------------

# List of all agents
agents = [user_proxy, general_manager, doctor, pharma_person, human_expert_proxy]

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

# Start the chat by having the User proxy send an initial message
user_proxy.initiate_chat(
    manager,
    message="Initiate clinic operations. With asking the human expert about the patient's concerns that you can forward to doctor right now. You are doing that as lialision rather than initiator.",
    clear_history=True
)