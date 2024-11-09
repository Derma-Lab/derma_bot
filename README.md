# DERMA Frontend


## Features

- Three AI agents with distinct roles in the medical consultation process
- Using MD Agent Framework and our variations
## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- npm 

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
AZURE_OAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=the_endpoint_url
DASHSCOPE_API_KEY = dashscope_api_key
```

DASHSCOPE_API_KEY is optional but we're recently considering to add it. 

   Replace `your_*_*` with your actual API key and Discord bot tokens.

## Usage

1. Run the backend first:
```
uvicorn md_agent_backend:app --reload
```

2. To run the frontend
```
cd agent_ui

npm install

npm run dev
```

The frontend will be hosted at localhost:3000, but backend would be hosted at localhost:8000. Run the backend file first and try fiddling around the frontend :)

