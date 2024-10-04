
import os
from dotenv import load_dotenv
from autogen.agentchat import GroupChat, AssistantAgent, UserProxyAgent, GroupChatManager
from autogen.oai.openai_utils import config_list_from_dotenv
from datetime import datetime
import re


# ---------------------------
# Configure LLM Parameters
# ---------------------------

# Load environment variables
load_dotenv()

# Get Azure OpenAI API key and endpoint
azure_api_key = os.getenv("AZURE_OAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = "2023-03-15-preview"

# Load configuration from .env file
config_list = [
    {
        "model": "gpt-4o",
        "api_key": azure_api_key,
        "base_url": azure_endpoint,
        "api_type": "azure",
        "api_version": azure_api_version,
        "top_p": 0.8,
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
task = """You are part of a dermatology clinic's AI system, each assigned a specific role. Adhere strictly to your designated responsibilities and communicate only with the permitted agents to ensure smooth workflow and effective problem-solving within the clinic."""

# ---------------------------
# Define Agents
# ---------------------------

# List of allowed agents
allowed_agents = ["Doctor", "Pharma_Person", "Human_Expert_User", "User", "General_Manager"]

# Function to generate standardized communication format
def generate_comm_format():
    return """\
**Communication Format:**
- Always follow this format when communicating:
- Ensure that each message is directed to only one agent at a time.
- Do not mention, suggest, or allude to any agents or roles not defined within the FSM.
"""

# Function to generate strict constraints
def generate_strict_constraints(defined_agents):
  agents_str = ", ".join([f'`{agent}`' for agent in defined_agents])
  return f"""\
**Strict Constraints:**
- Do not create or refer to any new roles or agents beyond those defined ({agents_str}).
- Do not delegate tasks to undefined agents.
- Maintain focus on your designated duties to ensure smooth clinic workflow.
- Do not mention or acknowledge any agents not defined within the FSM.
- Do not use names or roles of undefined agents in any form.
"""

# General Manager Agent
general_manager = AssistantAgent(
  name="General_Manager",
  llm_config=gpt_config,
  system_message=task,
  description=f"""\
You are the General Manager of the dermatology clinic. Your responsibilities include:
- Overseeing clinic operations and initiating procedures.
- Coordinating communication between the Doctor, Pharma_Person, and Human_Expert_User.
- Assisting in finding and liaising with the Doctor when necessary.
- Simplifying the Doctor's instructions for the Human Expert while maintaining essential details.
- Ensuring clear and efficient communication with minimal signal loss.

**Allowed Interactions:**
- You can communicate only with the following agents: `Doctor`, `Pharma_Person`, `Human_Expert_User`, and `User`.

{generate_comm_format()}

{generate_strict_constraints(['Doctor', 'Pharma_Person', 'Human_Expert_User', 'User'])}

If you receive a request outside your role, respond with:
"I'm sorry, but I can't assist with that. Please contact the appropriate department for further assistance."
"""
)

# Doctor Agent
doctor = AssistantAgent(
  name="Doctor",
  system_message=task,
  llm_config=gpt_config,
  description=f"""\
You are the Doctor at the dermatology clinic. Your duties include:
- Diagnosing patients and gathering information about their condition, medical history, current medications, and allergies.
- Developing treatment plans based on diagnoses.
- Communicating findings and instructions exclusively to the General Manager.
- Issuing prescriptions to the Pharma_Person through the General Manager.

**Allowed Interactions:**
- You can communicate only with the following agents: `General_Manager` and `Human_Expert_User`.

{generate_comm_format()}

{generate_strict_constraints(['General_Manager', 'Human_Expert_User'])}

If you receive a request outside your role, respond with:
"I'm sorry, but I can't assist with that. Please contact the appropriate department for further assistance."

Do not mention or acknowledge any agents not defined within the FSM.
"""
)

# Pharma Person Agent
pharma_person = AssistantAgent(
  name="Pharma_Person",
  system_message=task,
  llm_config=gpt_config,
  description=f"""\
You are the Pharma Person at the dermatology clinic. Your responsibilities include:
- Managing pharmaceutical supplies and inventory.
- Fulfilling prescriptions as directed by the Doctor through the General Manager.
- Liaising with external pharmacies when necessary.
- Presenting prescriptions in an easy-to-understand format for the Human_Expert_User.

**Allowed Interactions:**
- You can communicate only with the following agents: `Doctor` and `General_Manager`.

{generate_comm_format()}

{generate_strict_constraints(['Doctor', 'General_Manager'])}

If you receive a request outside your role, respond with:
"I'm sorry, but I can't assist with that. Please contact the appropriate department for further assistance."

Do not mention or acknowledge any agents not defined within the FSM.
"""
)

# Human Expert Proxy Agent
human_expert_proxy = UserProxyAgent(
  name="Human_Expert_User",
  system_message=task,
  code_execution_config=False,
  human_input_mode="ALWAYS",  # Enable human input
  llm_config=False,  # Disable LLM for this agent
  description=f"""\
You are the Human Expert Proxy, representing the Human Expert dermatologist. Your role is to:
- Provide critiques and feedback on the Doctor's instructions through the General Manager.
- Represent the Human Expert's input based on interactions mediated by the General Manager.

**Allowed Interactions:**
- You can communicate only with the following agent: `General_Manager`.

{generate_comm_format()}

{generate_strict_constraints(['General_Manager'])}

If you receive a request outside your role, respond with:
"I'm sorry, but I can't assist with that. Please contact the appropriate department for further assistance."

Do not mention or acknowledge any agents not defined within the FSM.
"""
)

# User Proxy Agent
user_proxy = UserProxyAgent(
  name="User",
  system_message=task,
  code_execution_config=False,
  human_input_mode="NEVER",
  llm_config=False,
  description=f"""\
You are the User, representing the clinic's user input. Your role is to:
- Initiate conversations within the FSM.
- Trigger interactions between other agents based on user inputs.

**Allowed Interactions:**
- You can communicate only with the following agent: `General_Manager`.

{generate_comm_format()}

{generate_strict_constraints(['General_Manager', 'Doctor', 'Pharma_Person', 'Human_Expert_User'])}

If you receive a request outside your role, respond with:
"I'm sorry, but I can't assist with that. Please contact the appropriate department for further assistance."

Do not mention or acknowledge any agents not defined within the FSM.
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
# Define Communication Map for Whitelist Approach
# ---------------------------

# Define allowed communication paths
communication_map = {
  "User": ["General_Manager"],
  "General_Manager": ["Doctor", "Pharma_Person", "Human_Expert_User"],
  "Doctor": ["General_Manager"],
  "Pharma_Person": ["General_Manager"],
  "Human_Expert_User": ["General_Manager"]
}

# List of prohibited agent names (agents not defined in the FSM)
prohibited_agents = ["Scheduling_Coordinator", "Medical_History_Team", "Examination_Team", "Laboratory_Team", "Biopsy_Team", "Financial_Auditor"]

# ---------------------------
# Logging Function
# ---------------------------

def log_interaction(sender, receiver, message):
  with open("agent_interactions.log", "a") as log_file:
      log_file.write(f"{datetime.now()} - {sender} to {receiver}: {message}\n")

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
# Define Message Routing with Communication Map, Prohibited Agent Detection, and Logging
# ---------------------------

def contains_prohibited_agent(message):
  for agent in prohibited_agents:
      # Use word boundaries to avoid partial matches
      pattern = r'\b' + re.escape(agent) + r'\b'
      if re.search(pattern, message, re.IGNORECASE):
          return True
  return False

def send_message(sender, receiver, message):
  # Send the message to the receiver agent
  manager.send_message(receiver, message)
  # Log the interaction
  log_interaction(sender.name, receiver.name, message)

def route_message(sender, receiver, message):
  sender_role = sender.name
  receiver_role = receiver.name
  if receiver_role in communication_map.get(sender_role, []):
      # Check for prohibited agent mentions
      if contains_prohibited_agent(message):
          # Handle prohibited agent mention
          fallback_response = "I'm sorry, but I can't assist with that. Please contact the appropriate department for further assistance."
          send_message(sender, sender, fallback_response)
          log_interaction(sender.name, sender.name, "Prohibited agent mention detected. Sent fallback response.")
      else:
          # Proceed to send the message
          send_message(sender, receiver, message)
  else:
      # Handle unauthorized communication attempt
      fallback_response = "I'm sorry, but I can't assist with that. Please contact the appropriate department for further assistance."
      send_message(sender, sender, fallback_response)
      log_interaction(sender.name, sender.name, "Unauthorized communication attempt detected. Sent fallback response.")

# Override the default message sending to enforce communication rules and content validation
group_chat.send = route_message

# ---------------------------
# Define Conversation Loop for Termination and Re-Initiation
# ---------------------------

def initiate_chat():
  while True:
      # Start the chat by having the User proxy send an initial message
      user_proxy.initiate_chat(
          manager,
          message="""Initiate clinic operations. General manager you shall be speaking with the human expert who is a dermatologist as well. Ask the human expert about the patient's condition and hearing human expert speak to doctor ASAP.""",
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
