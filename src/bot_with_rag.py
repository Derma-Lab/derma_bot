import os

from dotenv import load_dotenv

load_dotenv()

from autogen import ConversableAgent


llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "base_url": os.environ["AZURE_OPENAI_ENDPOINT"],
            "api_type": "azure",
            "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
            "temperature": 0.5,
            "max_tokens": 2048,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.2,
            "top_p": 0.2,
        }
    ]
}
doctor = ConversableAgent(
    "doctor",
    system_message="You are a doctor. You receive queries from patients and provide medical advice.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

patient = ConversableAgent(
    "patient",
    system_message="You are a patient. You have a medical issue and need advice from the doctor.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

result = patient.initiate_chat(doctor, message="Hello Doctor! I have a headache. Can you help me?", max_turns=2)

print(result)

