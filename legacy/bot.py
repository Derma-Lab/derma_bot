import os
import asyncio
from dotenv import load_dotenv
from autogen import ConversableAgent
import discord

# Load environment variables from .env file
load_dotenv()

# Fetch Cerebras API keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# Fetch Discord tokens
DISCORD_TOKEN_DRSASHA = os.getenv("DISCORD_TOKEN_DRSASHA")
DISCORD_TOKEN_GMKATIE = os.getenv("DISCORD_TOKEN_GMKATIE")
DISCORD_TOKEN_PHARMABRO = os.getenv("DISCORD_TOKEN_PHARMABRO")

# Verify that all tokens are loaded
missing_vars = []
if not CEREBRAS_API_KEY:
    missing_vars.append("CEREBRAS_API_KEY")
if not DISCORD_TOKEN_DRSASHA:
    missing_vars.append("DISCORD_TOKEN_DRSASHA")
if not DISCORD_TOKEN_GMKATIE:
    missing_vars.append("DISCORD_TOKEN_GMKATIE")
if not DISCORD_TOKEN_PHARMABRO:
    missing_vars.append("DISCORD_TOKEN_PHARMABRO")

if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# Define Conversable Agents

# General Manager Katie
gmkatie = ConversableAgent(
    "gmkatie",
    system_message=(
        "You are Katie, the General Manager. You receive queries from patients and coordinate with the Doctor and Pharma Person to address their needs."
    ),
    llm_config={
        "config_list": [{
            "model": "llama3.1-70b",
            "api_key": CEREBRAS_API_KEY,
            "base_url": "https://api.cerebras.ai/v1",
            "api_type": "openai",
            "temperature": 0.7
        }]
    },
    human_input_mode="NEVER",
)

# Doctor DrSasha
drsasha = ConversableAgent(
    "drsasha",
    system_message=(
        "You are Dr. Sasha, a competent doctor. When instructed by the General Manager, you will assess patient information and write prescriptions."
    ),
    llm_config={
        "config_list": [{
            "model": "llama3.1-70b",
            "api_key": CEREBRAS_API_KEY,
            "base_url": "https://api.cerebras.ai/v1",
            "api_type": "openai",
            "temperature": 0.7
        }]
    },
    human_input_mode="NEVER",
)

# Pharma Person PharmaBro
pharmabro = ConversableAgent(
    "pharmabro",
    system_message=(
        "You are PharmaBro, the Pharma Specialist. Based on the doctor's prescription, you decide on the appropriate medicine and provide recommendations."
    ),
    llm_config={
        "config_list": [{
            "model": "llama3.1-70b",
            "api_key": CEREBRAS_API_KEY,
            "base_url": "https://api.cerebras.ai/v1",
            "api_type": "openai",
            "temperature": 0.7
        }]
    },
    human_input_mode="NEVER",
)

# Define Discord Clients for each bot
intents = discord.Intents.default()
intents.message_content = True

# Create separate Discord clients
client_gmkatie = discord.Client(intents=intents)
client_drsasha = discord.Client(intents=intents)
client_pharmabro = discord.Client(intents=intents)

# Define asyncio Events to ensure all clients are ready
gmkatie_ready = asyncio.Event()
drsasha_ready = asyncio.Event()
pharmabro_ready = asyncio.Event()

# Helper function to extract all messages from ChatResult
def extract_messages(chat_result):
    """
    Extracts all messages from the chat history in the ChatResult.

    Returns:
        List of tuples: [(sender_name, message_content), ...]
    """
    messages = []
    if not chat_result or not chat_result.chat_history:
        return messages

    for message in chat_result.chat_history:
        name = message.get('name')
        content = message.get('content')
        if name and content:
            messages.append((name.lower(), content))
    return messages

# Helper function to map agent name to Discord client
def get_client_by_name(name):
    """
    Returns the Discord client based on the agent's name.

    Args:
        name (str): Name of the agent (e.g., 'gmkatie').

    Returns:
        discord.Client: Corresponding Discord client.
    """
    name_to_client = {
        'gmkatie': client_gmkatie,
        'drsasha': client_drsasha,
        'pharmabro': client_pharmabro,
    }
    return name_to_client.get(name.lower())

# Helper function to send messages to Discord
async def send_messages(messages, channel_id):
    """
    Sends a list of messages to the specified Discord channel.

    Args:
        messages (list): List of tuples containing (sender_name, message_content).
        channel_id (int): Discord channel ID where messages will be sent.
    """
    for sender, content in messages:
        client = get_client_by_name(sender)
        if client is None:
            print(f"Unknown sender '{sender}'. Skipping message.")
            continue

        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                # Fetch the channel if not found in cache
                channel = await client.fetch_channel(channel_id)

            # Format the message with bold sender name
            formatted_message = f"**{sender.upper()}:** {content}"
            await channel.send(formatted_message)
            print(f"Sent message from {sender.upper()}: {content[:50]}...")  # Log first 50 chars
        except Exception as e:
            print(f"Error sending message from {sender.upper()}: {e}")

# Helper function to initiate conversation
async def initiate_conversation(channel_id):
    """
    Initiates the conversation among the agents and sends all messages to Discord.

    Args:
        channel_id (int): Discord channel ID where messages will be sent.
    """
    # Simulate a patient query
    patient_query = "Hello, I've been experiencing frequent headaches and dizziness. Can you help me?"

    # GMKATIE sends the patient query
    gm_message = f"Patient Query: {patient_query}"
    print(f"GMKATIE: {gm_message}")

    # Initiate chat between GMKATIE and DRSASHA
    dr_chat = drsasha.initiate_chat(
        gmkatie,
        message=f"Please provide a prescription based on the following patient query: {patient_query}",
        max_turns=10  # Increase max_turns as needed
    )

    # Extract all messages from DRSASHA's chat
    dr_messages = extract_messages(dr_chat)

    # Initiate chat between DRSASHA and PHARMABRO
    pharma_chat = pharmabro.initiate_chat(
        drsasha,
        message=f"Based on the following prescription, please decide on the appropriate medicine: {dr_chat.summary}",
        max_turns=10  # Increase max_turns as needed
    )

    # Extract all messages from PHARMABRO's chat
    pharma_messages = extract_messages(pharma_chat)

    # Optionally, initiate further chats if needed
    # For example, continue the conversation or conclude it

    # Combine all messages in chronological order
    all_messages = [("gmkatie", gm_message)] + dr_messages + pharma_messages

    # Send all messages to Discord
    await send_messages(all_messages, channel_id)

    # Optionally, send a final message
    final_message = (
        f"**GMKATIE:** Based on our assessment, here is your prescription and recommended medicine:\n"
        f"**Prescription:** {dr_chat.summary}\n"
        f"**Medicine Recommendation:** {pharma_chat.summary}"
    )
    await client_gmkatie.get_channel(channel_id).send(final_message)
    print("Conversation initiated and messages sent.")

# Event handler for GMKATIE bot
@client_gmkatie.event
async def on_ready():
    print(f"Logged in as {client_gmkatie.user} (GMKATIE)")
    gmkatie_ready.set()

@client_gmkatie.event
async def on_message(message):
    if message.author == client_gmkatie.user:
        return

    if message.content.strip().lower() == ".start":
        print("GMKATIE received .start command.")
        await initiate_conversation(message.channel.id)

# Event handler for DrSasha bot
@client_drsasha.event
async def on_ready():
    print(f"Logged in as {client_drsasha.user} (DrSasha)")
    drsasha_ready.set()

@client_drsasha.event
async def on_message(message):
    # Define DrSasha's message handling here if needed
    pass

# Event handler for PharmaBro bot
@client_pharmabro.event
async def on_ready():
    print(f"Logged in as {client_pharmabro.user} (PharmaBro)")
    pharmabro_ready.set()

@client_pharmabro.event
async def on_message(message):
    # Define PharmaBro's message handling here if needed
    pass

# Run all three Discord clients concurrently
async def run_bots():
    await asyncio.gather(
        client_gmkatie.start(DISCORD_TOKEN_GMKATIE),
        client_drsasha.start(DISCORD_TOKEN_DRSASHA),
        client_pharmabro.start(DISCORD_TOKEN_PHARMABRO),
    )

# Entry point
if __name__ == "__main__":
    try:
        asyncio.run(run_bots())
    except KeyboardInterrupt:
        print("Bots are shutting down.")
