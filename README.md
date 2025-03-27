# Marketing Media Planner Project

This repository contains a project to provide marketing strategy using the key input of website name.

## Prerequisites

- Python 3.11 or higher (required by langmem)
- Git

## Installation

### 1. Set up a Python environment (Optional)

It's recommended to use a virtual environment to avoid package conflicts.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify Python version (should be 3.11 or higher)
python --version
```

### 2. Configure the environment variables

Create a `.env` file in the root directory of the project based on the provided example:

```bash
cp .env.sample .env
```

Edit the `.env` file and fill in the required variables according to your needs.

#### API Keys Configuration (Optional)

You can configure either OpenAI or Anthropic API keys in your `.env` file:

```
# Choose one of the following API configurations
# OpenAI configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Install the package

Install the package in development mode:

```bash
python -m pip install -e .
```

This will install the package and its dependencies.

## Running the application

To start the development server:

```bash
langgraph dev
```

This will start the langgraph development server, allowing you to interact with the application.

## Accessing the Agent Interface

Once your LangGraph server is running, you can access the agent interface by visiting:

```
https://agentchat.vercel.app/
```

When you first visit the site, you'll need to provide:

1. **Deployment URL** - Your LangGraph server URL (e.g., `http://127.0.0.1:2024`)
2. **Assistant / Graph ID** - The name of your agent (typically `agent`)

No thread ID is required as the interface will handle conversation threads automatically.

## Troubleshooting

- If you encounter any issues with Python version, ensure you're using Python 3.11 or higher.
- Make sure all the environment variables in the `.env` file are correctly set.
- For dependency issues, try removing the virtual environment and creating a fresh one.
