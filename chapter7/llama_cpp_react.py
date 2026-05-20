import re
import inspect

import kick_zscaler

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="unsloth/gemma-4-E4B-it-GGUF",
    filename="gemma-4-E4B-it-Q4_0.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    flash_attn=True,
    verbose=False,
)

def get_my_location() -> str:
    print(f"  [get_location]")
    return 'Chicago'

def get_weather(city: str) -> str:
    print(f"  [get_weather] {city}")
    return f'It\'s sunny in {city}, temperature is 25°C.'

TOOLS = {"get_my_location": get_my_location, "get_weather": get_weather}

def _tool_signature(name: str, fn) -> str:
    params = list(inspect.signature(fn).parameters.keys())
    return f"{name}({', '.join(params)})"

TOOL_SIGNATURES = {name: _tool_signature(name, fn) for name, fn in TOOLS.items()}

SYSTEM_PROMPT = f"""You are a helpful assistant. Answer the user's question using the available tools.

Available tools:
- get_my_location(): get my current location, returns the city name
- get_weather(city): get the weather for a given city, returns weather information

You must follow this loop until you have a final answer:

Thought: <your reasoning about what to do>
Action: <tool_name>(<argument>)
Observation: <tool result>

When you have enough information:
Thought: I now know the final answer.
Final Answer: <your answer>

Important:
- Call only one tool per step.
- Tool names must be exactly one of: {list(TOOLS.keys())}
- For tools with no parameters write: Action: get_my_location()
- For tools with parameters write positional values only, no keyword names: Action: get_weather(Chicago)
- Never fabricate an Observation. Always wait for the real result.
"""

def parse_action(text: str):
    match = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text)
    if match:
        arg = match.group(2).strip().strip("\"'")
        arg = re.sub(r"^\w+=", "", arg).strip().strip("\"'")
        return match.group(1), arg
    return None, None

def call_tool(name: str, arg: str):
    fn = TOOLS[name]
    params = list(inspect.signature(fn).parameters.keys())
    if params:
        return fn(arg)
    return fn()

def react(question: str, max_steps: int = 6):
    print(f"\nQuestion: {question}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    scratchpad = ""

    for step in range(max_steps):
        current_messages = messages.copy()
        if scratchpad:
            current_messages.append({"role": "assistant", "content": scratchpad})

        response = llm.create_chat_completion(
            messages=current_messages,
            stop=["Observation:"],
            max_tokens=256,
            temperature=0.1,
        )

        chunk = response["choices"][0]["message"]["content"].strip()
        print(f"--- step {step + 1} ---\n{chunk}")

        if "Final Answer:" in chunk:
            final = chunk.split("Final Answer:")[-1].strip()
            print(f"\n=== Final Answer: {final} ===")
            return final

        tool_name, tool_arg = parse_action(chunk)
        if tool_name and tool_name in TOOLS:
            observation = call_tool(tool_name, tool_arg)
        elif tool_name:
            observation = f"Unknown tool '{tool_name}'. Use one of: {list(TOOLS.keys())}"
        else:
            observation = "No valid Action found. Follow the format: Action: tool_name(argument)"

        print(f"Observation: {observation}\n")
        scratchpad += chunk + f"\nObservation: {observation}\n"

    return "Max steps reached without a final answer."

react("Where am I, and how about the weather there?")

llm.close()