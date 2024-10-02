import os
import asyncio
from dotenv import load_dotenv
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
from llama_index.core.agent import ReActAgent
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.tools import FunctionTool

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = "https://derma-lab-test.openai.azure.com/"
azure_api_version = "2024-02-15-preview"

# Tool Functions
def initiate_meeting(topic: str) -> str:
    """Initiate a meeting with a specific topic."""
    return f"Meeting initiated on topic: {topic}"

def present_patient_case(patient_name: str, condition: str) -> str:
    """Present a patient's case with name and condition."""
    return f"Patient {patient_name} is suffering from {condition}."

def discuss_medication(medication: str, benefits: str) -> str:
    """Discuss medication and its benefits."""
    return f"Medication {medication} offers the following benefits: {benefits}."

# Function Tools
gm_tool = FunctionTool.from_defaults(fn=initiate_meeting)
doctor_tool = FunctionTool.from_defaults(fn=present_patient_case)
pharma_tool = FunctionTool.from_defaults(fn=discuss_medication)

# LLM Configuration
llm = AzureOpenAI(
    model="gpt-4o-mini",
    deployment_name="gpt-4o-mini",
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)

# Agent Configuration
agent = ReActAgent.from_tools(
    [gm_tool, doctor_tool, pharma_tool],
    llm=llm,
    verbose=True
)

# Custom Events
class InitiateMeetingEvent(Event):
    topic: str

class PresentCaseEvent(Event):
    patient_name: str
    condition: str

class DiscussMedicationEvent(Event):
    medication: str
    benefits: str

class ConcludeMeetingEvent(Event):
    summary: str

# Workflow Definition
class ClinicWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> InitiateMeetingEvent:
        print("GM: Let's start our meeting on improving patient care.")
        return InitiateMeetingEvent(topic="Improving Patient Care")

    @step
    async def initiate_meeting(self, ctx: Context, ev: InitiateMeetingEvent) -> PresentCaseEvent:
        gm_response = gm_tool(topic=ev.topic)
        print(f"GM: {gm_response}")
        return PresentCaseEvent(patient_name="John Doe", condition="Diabetes")

    @step
    async def present_patient_case(self, ctx: Context, ev: PresentCaseEvent) -> DiscussMedicationEvent:
        doctor_response = doctor_tool(patient_name=ev.patient_name, condition=ev.condition)
        print(f"Doctor: {doctor_response}")
        return DiscussMedicationEvent(medication="Metformin", benefits="Controls blood sugar levels effectively.")

    @step
    async def discuss_medication(self, ctx: Context, ev: DiscussMedicationEvent) -> ConcludeMeetingEvent:
        pharma_response = pharma_tool(medication=ev.medication, benefits=ev.benefits)
        print(f"Pharma: {pharma_response}")
        summary = f"Decided to prescribe {ev.medication} for patient management."
        return ConcludeMeetingEvent(summary=summary)

    @step
    async def conclude_meeting(self, ctx: Context, ev: ConcludeMeetingEvent) -> StopEvent:
        print(f"GM: {ev.summary}")
        return StopEvent(result=ev.summary)

# Workflow Execution
async def run_workflow():
    workflow = ClinicWorkflow(timeout=30, verbose=True)
    result = await workflow.run()
    print(f"Meeting Result: {result}")

if __name__ == "__main__":
    asyncio.run(run_workflow())
