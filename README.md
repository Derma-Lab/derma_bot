# DERMA Consultation Discord Bot

This Discord bot simulates a medical consultation process using three AI agents: a General Manager (Katie), a Doctor (Dr. Sasha), and a Pharmacist (PharmaBro). The bot uses the Cerebras API for natural language processing and Discord for user interaction.

## Features

- Three AI agents with distinct roles in the medical consultation process
- Integration with Discord for user interaction
- Asynchronous operation to handle multiple Discord clients
- Environment variable management for secure API key and token storage

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone this repository:


```
git clone https://github.com/Derma-Lab/derma_bot
cd derma_bot
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Set up your environment variables:
   Create a `.env` file in the project root directory with the following content:

```
CEREBRAS_API_KEY=your_cerebras_api_key
DISCORD_TOKEN_DRSASHA=your_drsasha_discord_token
DISCORD_TOKEN_GMKATIE=your_gmkatie_discord_token
DISCORD_TOKEN_PHARMABRO=your_pharmabro_discord_token
```

   Replace `your_*_*` with your actual API key and Discord bot tokens.

## Usage

1. Run the bot:
```
python bot.py
```


2. Once the bot is running, you'll see console messages indicating that each bot (GMKATIE, DrSasha, and PharmaBro) has logged in.

3. In a Discord server where the bots are invited, type `.start` to initiate a medical consultation.

4. The bots will simulate a conversation, with GMKATIE coordinating between the patient (you), DrSasha, and PharmaBro.

5. The conversation results, including the prescription and medicine recommendation, will be posted in the Discord channel.

## File Structure

- `bot.py`: The main script containing the bot logic and Discord client setup.
- `.env`: Environment variable file (not tracked by git) containing API keys and Discord tokens.
- `requirements.txt`: List of Python package dependencies.

## Customization

You can customize the behavior of the AI agents by modifying their system messages in the `bot.py` file. Look for the `ConversableAgent` initialization for each agent (gmkatie, drsasha, pharmabro) to adjust their roles and responses.
