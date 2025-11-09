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
    # "input": "List all files in the workspace and show me their sizes using `ls -l`."
})


# DOCKER IMAGE

from typing import Optional, Type, Dict, Any
from pathlib import Path
import docker
from docker.errors import ImageNotFound, NotFound
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import tempfile
import os


class ShellInput(BaseModel):
    command: str = Field(description="Shell command to execute in the sandbox")
    workdir: Optional[str] = Field(
        default=None,
        description="Working directory relative to mounted path"
    )


class SandboxTool(BaseTool):
    name: str = "sandbox_shell"
    description: str = (
        "Execute shell commands in a sandboxed container with git, ast-grep, "
        "ripgrep, jq, find, and Python. Commands run in an isolated environment "
        "with the current directory mounted at /workspace."
    )
    args_schema: Type[BaseModel] = ShellInput
    
    mount_path: str = Field(default_factory=lambda: os.getcwd())
    image_name: str = "langchain-sandbox:latest"
    container_name: str = "langchain-sandbox-runtime"
    timeout: int = 300
    _client: Optional[docker.DockerClient] = None
    _image_built: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, mount_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if mount_path:
            self.mount_path = mount_path
        self._client = docker.from_env()
        self._ensure_image()

    @property
    def dockerfile(self) -> str:
        return """FROM python:3.11-alpine

RUN apk add --no-cache \
    git \
    jq \
    bash \
    curl \
    findutils \
    grep \
    sed \
    coreutils \
    build-base \
    && pip install --no-cache-dir uv \
    && uv pip install --system --no-cache \
        ast-grep-py \
        pydantic \
        requests \
    && curl -LO https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz \
    && tar -xzf ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz \
    && mv ripgrep-14.1.1-x86_64-unknown-linux-musl/rg /usr/local/bin/ \
    && rm -rf ripgrep-* \
    && apk del build-base curl

WORKDIR /workspace

CMD ["/bin/bash"]
"""

    def _ensure_image(self):
        if self._image_built:
            return
            
        try:
            self._client.images.get(self.image_name)
            self._image_built = True
        except ImageNotFound:
            self._build_image()

    def _build_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(self.dockerfile)
            
            self._client.images.build(
                path=str(tmpdir),
                tag=self.image_name,
                rm=True,
                forcerm=True
            )
            self._image_built = True

    def _get_or_create_container(self):
        try:
            container = self._client.containers.get(self.container_name)
            if container.status != "running":
                container.start()
            return container
        except NotFound:
            return self._client.containers.run(
                self.image_name,
                command="tail -f /dev/null",
                name=self.container_name,
                detach=True,
                volumes={
                    str(Path(self.mount_path).resolve()): {
                        "bind": "/workspace",
                        "mode": "rw"
                    }
                },
                working_dir="/workspace",
                remove=False,
                network_mode="none"
            )

    def _run(
        self,
        command: str,
        workdir: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        container = self._get_or_create_container()
        
        exec_workdir = "/workspace"
        if workdir:
            exec_workdir = f"/workspace/{workdir.lstrip('/')}"
        
        try:
            result = container.exec_run(
                cmd=["bash", "-c", command],
                workdir=exec_workdir,
                demux=True,
                environment={"TERM": "xterm-256color"}
            )
            
            exit_code = result.exit_code
            stdout = result.output[0].decode() if result.output[0] else ""
            stderr = result.output[1].decode() if result.output[1] else ""
            
            output = f"Exit Code: {exit_code}\n"
            if stdout:
                output += f"\nStdout:\n{stdout}"
            if stderr:
                output += f"\nStderr:\n{stderr}"
            
            return output
            
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def cleanup(self):
        try:
            container = self._client.containers.get(self.container_name)
            container.stop(timeout=5)
            container.remove()
        except NotFound:
            pass

    def __del__(self):
        if hasattr(self, '_client') and self._client:
            try:
                self.cleanup()
            except:
                pass


if __name__ == "__main__":
    tool = SandboxTool(mount_path=".")
    
    result = tool.invoke({"command": "python --version && rg --version && jq --version"})
    print(result)
    
    result = tool.invoke({"command": "ls -la"})
    print(result)
    
    result = tool.invoke({
        "command": "git status || echo 'Not a git repo'",
        "workdir": "."
    })
    print(result)
    
    tool.cleanup()