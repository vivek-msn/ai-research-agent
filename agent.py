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

def generate_plan(goal):

    plan_prompt= f"""
You are a planning system for an AI agent.

Break the user's goal into clear, actionable tasks.

Rules:
- Create only the tasks required to achieve the goal.
- Keep tasks simple and executable.
- Maintain the correct order when dependencies exists.
- Do not perform the tasks.
- Return ONLY valid JSON.

Use this structure:

{{
    "goal": "{goal}",
    "tasks": [
        {{
            "id": 1,
            "task": "..."
        }}
    ]
}}

User goal:
{goal}
"""
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=plan_prompt,
    )

    return json.loads(response.text)


def execute_task_with_tools(task_description):

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=(
            "Execute the following task using the available tools "
            "when appropriate. If a tool is not suitable, explain "
            "what cannot be executed.\n\n"
            f"Task: {task_description}"
        ),
        config=types.GenerateContentConfig(
            tools=[tools]
        ),
    )

    # Check whether Gemini returned any tool calls
    if not response.function_calls:
        return {
            "status": "failed",
            "error": "No tool call was returned for this task."
        }

    available_tools = {
        "calculator": calculator,
        "get_weather": get_weather
    }

    tool_results = []

    # Execute every tool call returned by Gemini
    for function_call in response.function_calls:

        tool = available_tools.get(function_call.name)

        # Check whether tool exists
        if tool is None:
            return {
                "status": "failed",
                "error": f"Unknown tool: {function_call.name}"
            }

        try:
            result = tool(**function_call.args)

            tool_results.append({
                "tool": function_call.name,
                "arguments": function_call.args,
                "result": result
            })

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    # Return after all tool calls are executed
    return {
        "status": "completed",
        "tool_results": tool_results
    }


def execute_plan(plan, previous_results=None):

    results = previous_results.copy() if previous_results else []

    for task in plan["tasks"]:

        already_completed = any(
            result.get("task_id") == task["id"]
            and result.get("status") == "completed"
            for result in results
        )

        if already_completed:
            print(f"Task {task['id']} already completed. Skipping.")
            continue

        task_results = {
            "task_id": task["id"],
            "task": task["task"],
            "status": "pending"
        }

        print(f"Task {task['id']}: {task['task']}")
        print("Status:", task_results["status"])

        # Task start execution
        task_results["status"] = "running"
        print("Status:", task_results["status"])

        # Simulated successful execution
        # task_results["status"] = "completed"
        # print("Status:", task_results["status"])

        max_retries = 2
        retries = 0

        while True:

            execution = execute_task_with_tools(task["task"])

            task_results["results"] = execution

            if execution["status"] == "completed":
                task_results["status"] = "completed"
                print("Status:", task_results["status"])
                break

            task_results["status"] = "failed"            
            print("Status:", task_results["status"])

            if retries >= max_retries:
                print("Maximum retries reached.")
                break

            retries +=1
            print(f"Retry attempt {retries}...")
            task_results["status"] = "running"
            print("Status:", task_results["status"])

        results.append(task_results)

    return results

MAX_REPLANS = 2

replans = 0

def replan(plan, failure_reason):

    print("Re-planning...")
    print("Failure reason:", failure_reason)

    replan_prompt = f"""
You are a re-planning system for an AI agent.

The current plan failed during execution.

Current Plan:
{json.dumps(plan, indent=4)}

Failure Reason:
{failure_reason}

Creat a modified plan that can continue toward the original goal.

Rules:
- Keep successful tasks unchanged where possible.
- Modify or replace the failed task.
- Do not unnecessarily change completed tasks.
- Return ONLY valid JSON.
- Use the same structure as the original plan.

Return:
{{
    "goal": "...",
    "tasks": [
        {{
            "id": 1,
            "task": "..."
        }}
    ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=replan_prompt
    )

    new_plan = json.loads(response.text)

    return new_plan

def orchestrate_plan(goal):

    plan = generate_plan(goal)

    previous_results = []

    if not validate_plan(plan):
        print("Invalid initial plan. Stopping.")
        return []

    replans = 0

    while True:

        print("\nExecuting current plan...")

        results = execute_plan(
            plan,
            previous_results=previous_results
        )

        previous_results = results

        # failed_tasks = [
        #     task for task in results
        #     if task.get("status") == 'failed'
        # ]

        # Keep th latest result for each task
        latest_results = {}

        for task in results:
            task_id = task.get("task_id")
            latest_results[task_id] = task

        # Identify tasks whose latest status is failed
        failed_tasks = [
            task for task in latest_results.values()
            if task.get("status") == "failed"
        ]

        # Stop if every task completed successfully
        if latest_results and all(
            task.get("status") == "completed"
            for task in latest_results.values()
        ):
            print("Plan completed successfully!")
            return results
        
         # Stop if no task failed but execution is incomplete
        if not failed_tasks:
            print("Execution incomplete. Stopping.")
            return results

        # Check re-planning limit
        if replans >= MAX_REPLANS:
            print("Maximum re-planning limit reached. Stopping.")
            return results

        failure_reason = "; ".join(
            f"Task {task.get('task_id')} failed: {task.get('task', '')}"
            for task in failed_tasks
        )

        replans += 1

        print(f"\nRe-planning attempt {replans}...")
        print("Failure reason:", failure_reason)

        new_plan = replan(plan, failure_reason)

        if not validate_plan(new_plan):
            print("New plan is invalid. Stopping.")
            return results

        plan = new_plan

def validate_plan(plan):

    if not isinstance(plan, dict):
        return False

    if not plan.get("goal"):
        return False

    tasks = plan.get("tasks")

    if not isinstance(tasks, list) or not tasks:
        return False

    for task in tasks:

        if not isinstance(task, dict):
            return False

        if "id" not in task:
            return False

        if not task.get("task"):
            return False

    return True

def load_memory():

    with open("memory.json", "r") as file:
        return json.load(file)

def save_memory(memory):

    with open("memory.json", "w") as file:
        json.dump(memory, file, indent=4)

# Testing

# test_result = execute_task_with_tools(
#     "Calculate 25 multiplied by 4 using the calculator tool."
# )

# print("\nHelper Test Result:")
# print(json.dumps(test_result, indent=4))

# exit()

# Temporary Planner Test

# test_plan = generate_plan(
#     "Research RAG vs Fine-tuning, compare them, and recommend which one is better for a document question-answering application."
# )

# print("Generated Plan:")
# print(json.dumps(test_plan, indent=4))

# is_valid = validate_plan(test_plan)

# print("Plan valid:", is_valid)

# results = execute_plan(test_plan)

# # new_plan = replan(
# #     test_plan,
# #     "Task 3 failed after retry"
# # )

# replans = 0
# current_plan = test_plan

# while replans < MAX_REPLANS:

#     replans +=1

#     print(f"Re-planning attempt {replans}...")

#     current_plan = replan(
#         current_plan,
#         "Task 3 failed after retry"
#     )

#     print("New Plan:")
#     print(json.dumps(current_plan,indent=4))

# print("Maximum re-planning limit reached.")

# print("Execution Results:")
# print(json.dumps(results, indent=4))

# print("Validating New Plan...")

# if validate_plan(current_plan):

#     print("New Plan is valid!")

#     new_results = execute_plan(current_plan)

#     print("New Execution Results:")
#     print(json.dumps(new_results, indent=4))

# else:
#     print("New Plan is invalid. Execution stopped.")

# exit()



# Temporary Planner Test

# test_plan = {
#     "goal": "Test calculator execution",
#     "tasks": [
#         {
#             "id": 1,
#             "task": "Calculate 25 multiplied by 4 using the calculator tool."
#         }
#     ]
# }

# results = execute_plan(test_plan)

# print("\nFinal Execution Results:")
# print(json.dumps(results, indent=4))

# exit()


# Temporary Retry Test

# def mock_failed_execution(task_description):
#     print("Simulating task failure...")
#     return {
#         "status": "failed",
#         "error": "Temporary test failure"
#     }

# original_executor = execute_task_with_tools
# execute_task_with_tools = mock_failed_execution

# test_plan = {
#     "goal": "Test retry handling",
#     "tasks": [
#         {
#             "id": 1,
#             "task": "Simulate a task failure"
#         }
#     ]
# }

# results = execute_plan(test_plan)

# print("\nRetry Test Results:")
# print(json.dumps(results, indent=4))

# execute_task_with_tools = original_executor
# exit()


        
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

def main():

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


    # test_plan = generate_plan(
    #     "Research RAG vs Fine-tuning and recommmend which one to use"
    # )

    # print("Generated Plan: ")
    # print(json.dumps(test_plan, indent=4))

if __name__== "__main__":
   main()