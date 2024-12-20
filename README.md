
# Installation and Usage Guide

## Cloning the Repository
Clone the repository using:

```
    git clone https://github.com/Derma-Lab/derma_bot
```

## Setup
Navigate to the agent directory:

```
    cd agent
```


### Run QuickInstall Script
For Linux users:

```
    chmod +x quickinstall.sh
    ./quickinstall.sh
```

For Windows users:

```
    chmod +x quickinstall.bat
    ./quickinstall.bat
```

## Running the Agent
First, activate the virtual environment:

For Linux users:

```
    source .venv/bin/activate
    streamlit run agent.py
````

For Windows users:'

```
    .venv\Scripts\activate
    streamlit run agent.py
```

## Using the Agent
The application has been installed via quickinstall script. To start the application:

```
    streamlit run agent.py
```