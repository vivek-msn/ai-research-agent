import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import calculator, get_weather

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Define calculator tool
calculator_tool = types.FunctionDeclaration(
            name="calculator",
            description="Perform basic mathematical calculations.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "a": types.Schema(
                        type="NUMBER",
                        description="First number"
                    ),
                    "b": types.Schema(
                        type="NUMBER",
                        description="Second number"
                    ),
                    "operation": types.Schema(
                        type="STRING",
                        description="Operation: add,substract, multiply, divide, or percentage"
                    ),
                },
                required=["a", "b", "operation"],
            ),
        )

# Define weather tool
wheather_tool = types.FunctionDeclaration(
    name="get_weather",
    description="Get weather information for a city.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "city": types.Schema(
                type="STRING",
                description="Name of the city"
            ),
        },
        required=["city"],
    ),
)

# Tool config
tools = types.Tool(
    function_declarations=[
        calculator_tool,
        wheather_tool
    ]
)

def update_summary(old_history, current_summary):

    summary_prompt = f"""
You are a memory summerization system.

Current summary:
{current_summary}

New conversation history:
{old_history}

Create an updated concise summary of the important information
that should be remembered for future conversation.

Keep important facts, user preferences, ongoing goals, and project context.
Do not include temporary questions, calculations, or irrelevant details.

Return only the updated summary.
"""

    response = client.models.generate_content(
        model ="gemini-3.1-flash-lite",
        contents=summary_prompt,
    )

    return response.text

def load_memory():

    with open("memory.json", "r") as file:
        return json.load(file)

def save_memory(memory):

    with open("memory.json", "w") as file:
        json.dump(memory, file, indent=4)
        
def extract_memory(user_message):

    prompt = f"""
You are a memory extraction system.

Analyze the user's message and identify information that is useful
to remember for future conversations.

Only extract stable and useful information such as:
- Name
- Preferences
- Long-term goals
- Important project context

Use this JSON structure when possible:

{{
    "name": "...",
    "preferences": {{
        "...": "..."
    }},
    "long_term_goals": "...",
    "project_context": "..."
}}

Rules:
- Put programming language preferences inside "preferences".
- Do not create duplicate top-level keys for information already represented
  inside "preferences".
- Do not create keys like "preferred_programming_language" if the same
  information can be stored as "preferences": {{"programming_language": "..."}}.
- Only include keys for information actually found in the user's message.

Do not extract temporary questions, calculations, weather,
or casual conversation.

User message:
{user_message}

Return ONLY valid JSON.

If there is no useful memory, return:
{{}}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    return json.loads(response.text)

def merge_memory(existing_memory, new_memory):

    for key, value in new_memory.items():

        if isinstance(value, dict) and isinstance(existing_memory.get(key), dict):
            existing_memory[key].update(value)

        else:
            existing_memory[key] = value

    return existing_memory

# memory = load_memory()

# new_memory = extract_memory(
#     "I am working on an AI Research Agent project."
# )

# print("Extracted memory:", new_memory)

# memory = merge_memory(memory, new_memory)

# save_memory(memory)

# print("Updated memory:", memory)

# memory = load_memory()

# print("Loaded memory:", memory)

# memory["name"] = "Vivek"

# save_memory(memory)

# print("Saved memory:", memory)

# Start conversation history
contents = []

memory = load_memory()

MAX_TURNS = 5
conversation_turns = 0
turn_start_indexes = []
summary = ""

# Get user's first question
user_question = input("You:")

if user_question.lower() in ["exit", "quit"]:
    exit()

turn_start_indexes.append(len(contents))

# Start conversation with the user message
contents = [
    types.Content(
        role="user",
        parts=[
            types.Part(text=user_question)
        ]
    )
]

conversation_turns +=1

available_tools = {
    "calculator": calculator,
    "get_weather": get_weather
}

# Agent Loop
while True:

    context = contents

    if summary:
        context = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"Memory summary:\n{summary}"
                    )
                ]
            )
        ] + contents

    # if memory:
    #         context = [
    #             types.Content(
    #                 role="user",
    #                 parts=[
    #                     types.Part(
    #                         text=f"Long-term memory:\n{memory}"
    #                     )
    #                 ]
    #             )
    #         ] + context

    # print("Debug memory:", memory)
    # print("Debug context:", context)

    # Ask Gemin wants to use a tool
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=context,
        config=types.GenerateContentConfig(
            tools=[tools],
            system_instruction=f"""
        You are an AI assistant.

        Long-term memory:
        {memory}
        """

        ),
    )


    # Check if Gemini wants to use a tool
    if response.function_calls:

        # Gemini tool call response
        contents.append(response.candidates[0].content)

        # Tool response collect
        tool_response_parts = []

        # Gemini Tool calls requested
        for function_call in response.function_calls:

            print("Tool requested:", function_call.name)
            print("Arguments:", function_call.args)

            # Find the requested tool
            tool = available_tools.get(function_call.name)


            if tool is None:
                raise ValueError(
                    f"Unknown tool requested: {function_call.name}"
                )

            try:

                # Execute the selected tool
                result = tool(**function_call.args)

                print("Tool result:", result)

                tool_response = {
                    "result": result
                }

            except Exception as e:

                print("Tool error:", str(e))

                tool_response = {
                    "error": str(e)
                }

            # Add tool response
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response=tool_response,
                )
            )
            
            # Add tool result to cpnversation
            contents.append(
                types.Content(
                    role="user",
                    parts=tool_response_parts
                )
            )

    else:
        # No more tools needed 
        print("Agent:", response.text)

        # Get next user message
        user_question = input("You: ")

        if user_question.lower() in ["exit", "quit"]:
            break

        turn_start_indexes.append(len(contents))

        new_memory = extract_memory(user_question)

        if new_memory:
            memory = merge_memory(memory, new_memory)
            save_memory(memory)

            print("Updated long-term memory:", memory)

        # Add next user message to conversation history
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(text=user_question)
                ]
            )
        )

        conversation_turns += 1

        if conversation_turns > MAX_TURNS:

            trim_index = turn_start_indexes[1]

            old_history = contents[:trim_index]

            summary = update_summary(old_history, summary)

            print("Current summary:", summary)
            # print("Old history:", old_history)

            contents = contents[trim_index:]

            turn_start_indexes = [
                index - trim_index
                for index in turn_start_indexes[1:]
            ]

            conversation_turns -= 1

            print("Trimmed old turn")

        print("History length:", len(contents))
        print("Conversation turns:", conversation_turns)
        print("Turn start indexes:", turn_start_indexes)