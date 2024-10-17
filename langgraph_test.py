import os
import getpass
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Ensure these environment variables are set
azure_api_key = os.getenv("AZURE_OAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = "2023-03-15-preview"

# Check if the required environment variables are set
if not azure_api_key or not azure_endpoint:
    raise ValueError("Please set the AZURE_OAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")

tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

tools = [TavilySearchResults(max_result=1)]

prompt = hub.pull("hwchase17/openai-functions-agent")

# Initialize AzureChatOpenAI with required parameters
llm = AzureChatOpenAI(
    model="gpt-3.5-turbo-1106",
    api_key=azure_api_key,
    endpoint=azure_endpoint,
    api_version=azure_api_version,
    streaming=True
)

agent_runnable = create_openai_functions_agent(llm, tools, prompt)
