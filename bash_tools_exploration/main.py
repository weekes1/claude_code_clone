# from dotenv import load_dotenv
# load_dotenv()
# import os

# from langchain_openai import ChatOpenAI
# from langchain_community.tools import ShellTool
# from langchain_core.tools import Tool
# from langsmith import Client
# from langchain.agents import create_openai_functions_agent

# # Allowed base commands (read-only)
# READ_ONLY_COMMANDS = {"cat", "less", "head", "tail", "grep", "find", "wc", "ls"}

# WORKSPACE = os.path.expanduser("~/agent-playground")
# shell = ShellTool(working_directory=WORKSPACE)

# def safe_execute(command: str):
#     if command.startswith("/") or ".." in command:
#         raise ValueError(f"❌ Unsafe command blocked: {command}")
#     cmd = command.strip().split()[0]
#     if cmd not in READ_ONLY_COMMANDS:
#         raise ValueError(f"❌ Modification blocked: `{cmd}` is not allowed.")
#     return shell.run(command)

# safe_shell = Tool(
#     name="safe_shell",
#     func=safe_execute,
#     description="Run read-only shell commands inside the workspace."
# )

# llm = ChatOpenAI(model="gpt-4o")  # identifies as GPT-5
# client = Client()
# prompt = client.pull_prompt("hwchase17/openai-functions-agent")   

# agent = create_openai_functions_agent(
#     llm=llm,
#     tools=[safe_shell],
#     prompt=prompt,
# )

# agent.invoke({
#     "input": "List all files in the workspace and show me their sizes using `ls -l`."
# })


import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.tools import ShellTool
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_openai_functions_agent, create_tool_calling_agent, AgentExecutor

# Allowed read-only shell commands
READ_ONLY_COMMANDS = {"cat", "less", "head", "tail", "grep", "find", "wc", "ls"}

WORKSPACE = os.path.expanduser("~/agent-playground")
shell = ShellTool(working_directory=WORKSPACE)

def safe_execute(command: str):
    if command.startswith("/") or ".." in command:
        raise ValueError(f"❌ Unsafe command blocked: {command}")
    cmd = command.strip().split()[0]
    if cmd not in READ_ONLY_COMMANDS:
        raise ValueError(f"❌ Write/modify blocked: `{cmd}` is not allowed.")
    return shell.run(command)

safe_shell = Tool(
    name="safe_shell",
    func=safe_execute,
    description="Run read-only shell commands inside workspace."
)

# Model
llm = ChatOpenAI(model="gpt-4o")  # identifies as GPT-5

# Correct prompt for OpenAI tool-calling agents
prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(
    llm=llm,
    tools=[safe_shell],
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[safe_shell],
    verbose=True,
)

# Execute task
agent_executor.invoke({
    "input": "List all files in the workspace and show me their sizes using `ls -l`."
})
