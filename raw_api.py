import os
import openai
from dotenv import load_dotenv

load_dotenv()


azure_api_key = os.getenv("AZURE_OAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = "2024-07-01-preview"

# Set up Azure OpenAI API client
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=azure_api_key,
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint
)

# ---------------------------
# Define Agent Class
# ---------------------------

class Agent:
    def __init__(self, name, system_prompt, is_human=False):
        self.name = name
        self.system_prompt = system_prompt
        self.is_human = is_human

    def generate_response(self, conversation_history):
        if self.is_human:
            # Prompt the human user for input
            print(f"\n{self.name}, it's your turn to speak.")
            user_input = input(f"{self.name}: ")
            return user_input
        else:
            # AI generates response
            # Prepare messages for the API call
            messages = [{'role': 'system', 'content': self.system_prompt}]
            for msg in conversation_history:
                # Include the name in the message content
                messages.append({'role': msg['role'], 'content': f"{msg['name']}: {msg['content']}"})
            # Generate response using Azure OpenAI API
            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",  # Replace with your actual deployment name
                messages=messages,
                max_tokens=500,
                temperature=0,
            )
            assistant_reply = chat_completion.choices[0].message.content.strip()
            return assistant_reply

# ---------------------------
# Define Agents and Prompts
# ---------------------------

task = """You are part of a dermatology clinic's AI system, each assigned a specific role. Adhere strictly to your designated responsibilities and communicate only with the permitted agents to ensure smooth workflow and effective problem-solving within the clinic."""

general_manager_prompt = task + """
I am the General Manager.
I start the conversation by consulting with the patient about their condition and medical history.
I coordinate between the patient, doctor, human expert, and pharma person.
I ensure that the process flows smoothly and that all approvals are obtained.
Always begin your message with '@Recipient_Name', where Recipient_Name is the agent you are addressing.
"""

doctor_prompt = task + """
I am the Doctor.
I provide diagnoses based on the patient's medical information provided by the General Manager.
I only communicate with the General Manager.
Always begin your message with '@General_Manager'.
"""

patient_prompt = task + """
I am the Patient.
I respond to the General Manager's questions about my condition and medical history.
Always begin your message with '@General_Manager'.
"""

human_expert_prompt = task + """
I am the Human Expert.
I evaluate the doctor's diagnosis and the pharma person's medicine list.
I respond with 'APPROVED' if the diagnosis or medicine list is correct, or 'NOT APPROVED' if it is not.
I only communicate with the General Manager.
"""

pharma_person_prompt = task + """
I am the Pharma Person.
I create a list of medicines based on the final diagnosis provided by the General Manager.
I only communicate with the General Manager.
Always begin your message with '@General_Manager'.
"""

# Instantiate agents
general_manager = Agent('General_Manager', general_manager_prompt)
doctor = Agent('Doctor', doctor_prompt)
patient = Agent('Patient', patient_prompt, is_human=True)  # Patient is human-controlled
human_expert = Agent('Human_Expert', human_expert_prompt, is_human=True)  # Human Expert is human-controlled
pharma_person = Agent('Pharma_Person', pharma_person_prompt)

# ---------------------------
# Conversation Manager
# ---------------------------

def conversation_manager():
    conversation_history = []
    current_speaker = 'General_Manager'

    diagnosis_attempts = 0
    max_diagnosis_attempts = 3

    medicine_attempts = 0
    max_medicine_attempts = 3

    print("Conversation started.")

    # Step 1: General Manager asks patient about the disease and medical history
    print("\n--- Step 1: General Manager asks Patient ---")
    agent = general_manager
    agent_response = agent.generate_response(conversation_history)
    conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': agent_response})
    print(f"\nGeneral_Manager: {agent_response}\n")

    current_speaker = 'Patient'

    # Step 2: Patient answers about their disease
    print("\n--- Step 2: Patient responds ---")
    agent = patient
    agent_response = agent.generate_response(conversation_history)
    conversation_history.append({'role': 'user', 'name': 'Patient', 'content': agent_response})
    print(f"\nPatient: {agent_response}\n")

    current_speaker = 'General_Manager'

    # Diagnosis Approval Loop
    diagnosis_approved = False
    while diagnosis_attempts < max_diagnosis_attempts and not diagnosis_approved:
        diagnosis_attempts += 1

        # Step 3: General Manager conveys info to Doctor
        print("\n--- Step 3: General Manager asks Doctor ---")
        agent = general_manager
        agent_response = agent.generate_response(conversation_history)
        conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': agent_response})
        print(f"\nGeneral_Manager: {agent_response}\n")

        current_speaker = 'Doctor'

        # Step 4: Doctor provides diagnosis
        print("\n--- Step 4: Doctor provides diagnosis ---")
        agent = doctor
        agent_response = agent.generate_response(conversation_history)
        conversation_history.append({'role': 'assistant', 'name': 'Doctor', 'content': agent_response})
        print(f"\nDoctor: {agent_response}\n")

        current_speaker = 'General_Manager'

        # Step 5: General Manager asks Human Expert for approval
        print("\n--- Step 5: General Manager asks Human Expert ---")
        agent = general_manager
        agent_response = agent.generate_response(conversation_history)
        conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': agent_response})
        print(f"\nGeneral_Manager: {agent_response}\n")

        current_speaker = 'Human_Expert'

        # Step 6: Human Expert evaluates diagnosis
        print("\n--- Step 6: Human Expert evaluates diagnosis ---")
        agent = human_expert
        agent_response = agent.generate_response(conversation_history)
        conversation_history.append({'role': 'user', 'name': 'Human_Expert', 'content': agent_response})
        print(f"\nHuman_Expert: {agent_response}\n")

        # Analyze approval
        approval = analyze_approval(agent_response)
        if approval == 'APPROVED':
            print("Diagnosis approved.")
            diagnosis_approved = True
        else:
            print("Diagnosis not approved.")
            if diagnosis_attempts >= max_diagnosis_attempts:
                print("Maximum diagnosis attempts reached. Please visit the clinic again.")
                return
            else:
                print("Repeating diagnosis process.")

    # Step 7: General Manager conveys diagnosis to Patient
    print("\n--- Step 7: General Manager conveys diagnosis to Patient ---")
    agent = general_manager
    agent_response = agent.generate_response(conversation_history)
    conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': agent_response})
    print(f"\nGeneral_Manager: {agent_response}\n")

    current_speaker = 'Patient'

    # Step 8: Patient gives feedback
    print("\n--- Step 8: Patient gives feedback ---")
    agent = patient
    agent_response = agent.generate_response(conversation_history)
    conversation_history.append({'role': 'user', 'name': 'Patient', 'content': agent_response})
    print(f"\nPatient: {agent_response}\n")

    # Analyze patient's response for YES or NO
    feedback = analyze_yes_no(agent_response)
    if feedback == 'NO':
        print("Patient approves the diagnosis.")
    else:
        print("Patient suggests differences in condition.")
        # Loop back to diagnosis approval, max 3 times
        diagnosis_attempts = 0
        diagnosis_approved = False
        while diagnosis_attempts < max_diagnosis_attempts and not diagnosis_approved:
            diagnosis_attempts += 1

            # Steps 3 to 6 repeated
            print("\n--- Repeating Diagnosis Process ---")

            # General Manager asks Doctor
            agent = general_manager
            agent_response = agent.generate_response(conversation_history)
            conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': agent_response})
            print(f"\nGeneral_Manager: {agent_response}\n")

            current_speaker = 'Doctor'

            # Doctor provides new diagnosis
            agent = doctor
            agent_response = agent.generate_response(conversation_history)
            conversation_history.append({'role': 'assistant', 'name': 'Doctor', 'content': agent_response})
            print(f"\nDoctor: {agent_response}\n")

            current_speaker = 'General_Manager'

            # General Manager asks Human Expert
            agent = general_manager
            agent_response = agent.generate_response(conversation_history)
            conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': agent_response})
            print(f"\nGeneral_Manager: {agent_response}\n")

            current_speaker = 'Human_Expert'

            # Human Expert evaluates
            agent = human_expert
            agent_response = agent.generate_response(conversation_history)
            conversation_history.append({'role': 'user', 'name': 'Human_Expert', 'content': agent_response})
            print(f"\nHuman_Expert: {agent_response}\n")

            # Analyze approval
            approval = analyze_approval(agent_response)
            if approval == 'APPROVED':
                print("Diagnosis approved.")
                diagnosis_approved = True
            else:
                print("Diagnosis not approved.")
                if diagnosis_attempts >= max_diagnosis_attempts:
                    print("Maximum diagnosis attempts reached. Please visit the clinic again.")
                    return
                else:
                    print("Repeating diagnosis process.")

    # Step 10: General Manager tells Pharma Person
    print("\n--- Step 10: General Manager tells Pharma Person ---")
    agent = general_manager
    agent_response = agent.generate_response(conversation_history)
    conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': agent_response})
    print(f"\nGeneral_Manager: {agent_response}\n")

    current_speaker = 'Pharma_Person'

    # Medicine Approval Loop
    medicine_approved = False
    while medicine_attempts < max_medicine_attempts and not medicine_approved:
        medicine_attempts += 1

        # Pharma Person creates medicine list
        print("\n--- Pharma Person creates medicine list ---")
        agent = pharma_person
        agent_response = agent.generate_response(conversation_history)
        conversation_history.append({'role': 'assistant', 'name': 'Pharma_Person', 'content': agent_response})
        print(f"\nPharma_Person: {agent_response}\n")

        current_speaker = 'General_Manager'

        # General Manager asks Human Expert for approval
        print("\n--- General Manager asks Human Expert for medicine approval ---")
        agent = general_manager
        agent_response = agent.generate_response(conversation_history)
        conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': agent_response})
        print(f"\nGeneral_Manager: {agent_response}\n")

        current_speaker = 'Human_Expert'

        # Human Expert evaluates medicine list
        print("\n--- Human Expert evaluates medicine list ---")
        agent = human_expert
        agent_response = agent.generate_response(conversation_history)
        conversation_history.append({'role': 'user', 'name': 'Human_Expert', 'content': agent_response})
        print(f"\nHuman_Expert: {agent_response}\n")

        # Analyze approval
        approval = analyze_approval(agent_response)
        if approval == 'APPROVED':
            print("Medicine list approved.")
            medicine_approved = True
        else:
            print("Medicine list not approved.")
            if medicine_attempts >= max_medicine_attempts:
                print("Maximum medicine attempts reached. Please visit the clinic again.")
                return
            else:
                print("Repeating medicine process.")

    # Final Step: General Manager sends list to Patient and thanks them
    print("\n--- Final Step: General Manager concludes ---")
    agent = general_manager
    final_message = "@Patient\nThank you for your patience. Here is your prescribed medication. We wish you a speedy recovery."
    conversation_history.append({'role': 'assistant', 'name': 'General_Manager', 'content': final_message})
    print(f"\nGeneral_Manager: {final_message}\n")

    print("Conversation ended.")

# Helper functions

def analyze_approval(message):
    # Analyze the Human Expert's message to determine if it is 'APPROVED' or 'NOT APPROVED'
    # Using Azure OpenAI endpoint to analyze the message and output 'APPROVED' or 'NOT APPROVED'

    messages = [
        {'role': 'system', 'content': 'You will be provided with a message. Respond with "APPROVED" if the message indicates approval. Respond with "NOT APPROVED" if it indicates disapproval. Do not include any other text.'},
        {'role': 'user', 'content': message}
    ]

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your actual deployment name
        messages=messages,
        max_tokens=5,
        temperature=0,
    )
    assistant_reply = chat_completion.choices[0].message.content.strip().upper()
    if assistant_reply in ['APPROVED', 'NOT APPROVED']:
        return assistant_reply
    else:
        return 'NOT APPROVED'

def analyze_yes_no(message):
    # Analyze the Patient's message to determine if it is 'YES' or 'NO'
    # Using gpt-4o-mini to analyze the message and output 'YES' or 'NO'

    messages = [
        {'role': 'system', 'content': 'You will be provided with a message from a patient. If the patient is suggesting differences in their condition, respond with "YES". If the patient approves the diagnosis, respond with "NO". Do not include any other text.'},
        {'role': 'user', 'content': message}
    ]

    chat_completion = openai.ChatCompletion.create(
        engine="gpt-4o-mini",
        messages=messages,
        max_tokens=3,
        temperature=0,
    )
    assistant_reply = chat_completion.choices[0].message.content.strip().upper()
    if assistant_reply in ['YES', 'NO']:
        return assistant_reply
    else:
        return 'NO'


if __name__ == "__main__":
    conversation_manager()
