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

# Start conversation history
contents = []

MAX_TURNS = 5
conversation_turns = 0
turn_start_indexes = []

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

    # Ask Gemin wants to use a tool
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[tools]
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