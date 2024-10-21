import os
import openai
import asyncio
from dotenv import load_dotenv
import discord
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI API key and endpoint
azure_api_key = os.getenv("AZURE_OAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = "2024-07-01-preview"

# Set up Azure OpenAI API client
client = AzureOpenAI(
    api_key=azure_api_key,
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint
)

# Initialize intents for Discord bots
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Enable message content intent

class ConversationStage(Enum):
    START_CONVERSATION = "start_conversation"
    WAITING_FOR_SYMPTOMS = "waiting_for_symptoms"
    GM_TO_DOCTOR = "gm_to_doctor"
    DOCTOR_DIAGNOSIS = "doctor_diagnosis"
    COMPLETED = "completed"

@dataclass
class ConversationState:
    current_stage: ConversationStage = ConversationStage.START_CONVERSATION
    conversation_history: List[dict] = field(default_factory=list)
    diagnosis_attempts: int = 0
    medicine_attempts: int = 0
    diagnosis_approved: bool = False
    medicine_approved: bool = False
    active_channel: discord.TextChannel = None

# Global state
conversation_state = ConversationState()

# Initialize Discord clients
client_gmkatie = discord.Client(intents=intents)
client_drsasha = discord.Client(intents=intents)
client_pharmabro = discord.Client(intents=intents)

class Agent:
    def __init__(self, name, system_prompt, is_human=False):
        self.name = name
        self.system_prompt = system_prompt
        self.is_human = is_human

    async def generate_response(self, conversation_history):
        if self.is_human:
            return None
        else:
            messages = [{'role': 'system', 'content': self.system_prompt}]
            for msg in conversation_history:
                messages.append({'role': msg['role'], 'content': f"{msg['name']}: {msg['content']}"})
            
            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",  # Replace with your actual deployment name
                messages=messages,
                max_tokens=500,
                temperature=0,
            )
            return chat_completion.choices[0].message.content.strip()

# Define agent prompts
task = """You are part of a dermatology clinic's AI system, each assigned a specific role. Adhere strictly to your designated responsibilities and communicate only with the permitted agents to ensure smooth workflow and effective problem-solving within the clinic."""

general_manager_prompt = task + """
I am the General Manager.
I start the conversation by consulting with the patient about their condition and medical history.
I coordinate between the patient, doctor, human expert, and pharma person.
I ensure that the process flows smoothly and that all approvals are obtained.
Keep responses concise and professional.
"""

doctor_prompt = task + """
I am the Doctor.
I provide diagnoses based on the patient's medical information provided by the General Manager.
I only communicate with the General Manager.
Keep responses focused on medical diagnosis.
"""

pharma_person_prompt = task + """
I am the Pharma Person.
I create a list of medicines based on the final diagnosis provided by the General Manager.
I only communicate with the General Manager.
Keep responses focused on medication recommendations.
"""

# Instantiate agents
general_manager = Agent('General_Manager', general_manager_prompt)
doctor = Agent('Doctor', doctor_prompt)
pharma_person = Agent('Pharma_Person', pharma_person_prompt)

async def process_next_stage():
    if not conversation_state.active_channel:
        return

    if conversation_state.current_stage == ConversationStage.START_CONVERSATION:
        await conversation_state.active_channel.send("Conversation initiated. Please mention me (@GM Katie) and describe your symptoms.")
        conversation_state.current_stage = ConversationStage.WAITING_FOR_SYMPTOMS

    elif conversation_state.current_stage == ConversationStage.GM_TO_DOCTOR:
        # GM Katie articulates to doctor
        print("gm katie is articulating to doctor")
        response = await general_manager.generate_response(conversation_state.conversation_history)
        print("gm katie has responded")
        await conversation_state.active_channel.send(f"**General Manager Katie**: {response}")
        conversation_state.conversation_history.append({
            'role': 'assistant',
            'name': 'General_Manager',
            'content': response
        })
        # Add delay before doctor responds
        await asyncio.sleep(1)
        conversation_state.current_stage = ConversationStage.DOCTOR_DIAGNOSIS
        
        # Doctor's diagnosis
        response = await doctor.generate_response(conversation_state.conversation_history)
        await client_drsasha.get_channel(conversation_state.active_channel.id).send(f"**Dr. Sasha**: {response}")
        conversation_state.conversation_history.append({
            'role': 'assistant',
            'name': 'Doctor',
            'content': response
        })
        conversation_state.current_stage = ConversationStage.COMPLETED
        await conversation_state.active_channel.send("Consultation completed! Type !start to begin a new consultation.")

@client_gmkatie.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.startswith('!start'):
        conversation_state.active_channel = message.channel
        conversation_state.current_stage = ConversationStage.START_CONVERSATION
        conversation_state.conversation_history = []
        await process_next_stage()
    
    elif (conversation_state.active_channel and 
          message.channel == conversation_state.active_channel and 
          conversation_state.current_stage == ConversationStage.WAITING_FOR_SYMPTOMS):

        print("This sort of if condition is passed")
        
        # Check if GM Katie is mentioned
        if isinstance(message.channel, discord.channel.DMChannel) or (client_gmkatie.user and client_gmkatie.user.mentioned_in(message)):
            print("This is also passed")
            # Record patient's symptoms
            conversation_state.conversation_history.append({
                'role': 'user',
                'name': 'Patient',
                'content': message.content
            })
            # Move to next stage
            conversation_state.current_stage = ConversationStage.GM_TO_DOCTOR
            await process_next_stage()

@client_drsasha.event
async def on_ready():
    print(f'{client_drsasha.user} has connected to Discord!')

@client_pharmabro.event
async def on_ready():
    print(f'{client_pharmabro.user} has connected to Discord!')

@client_gmkatie.event
async def on_ready():
    print(f'{client_gmkatie.user} has connected to Discord!')

# Run all bots concurrently
async def run_bots():
    await asyncio.gather(
        client_gmkatie.start(os.getenv("DISCORD_TOKEN_GMKATIE")),
        client_drsasha.start(os.getenv("DISCORD_TOKEN_DRSASHA")),
        client_pharmabro.start(os.getenv("DISCORD_TOKEN_PHARMABRO")),
    )

# Entry point
if __name__ == "__main__":
    try:
        asyncio.run(run_bots())
    except KeyboardInterrupt:
        print("Bots are shutting down.")