
from types import SimpleNamespace
import agent


def test_multiple_tool_calls(monkeypatch):

    # Mock calculator tool
    def mock_calculator(expression):
        return 300

    # Mock weather tool
    def mock_get_weather(city):
        return f"Weather in {city}: Sunny"

    # Replace actual tools with mocked tools
    monkeypatch.setattr(agent, "calculator", mock_calculator)
    monkeypatch.setattr(agent, "get_weather", mock_get_weather)

    # Mock Gemini response with two function calls
    mock_response = SimpleNamespace(
        function_calls=[
            SimpleNamespace(
                name="calculator",
                args={"expression": "100 + 200"}
            ),
            SimpleNamespace(
                name="get_weather",
                args={"city": "Delhi"}
            )
        ]
    )

    # Mock Gemini API call
    monkeypatch.setattr(
        agent.client.models,
        "generate_content",
        lambda **kwargs: mock_response
    )

    # Execute task
    result = agent.execute_task_with_tools(
        "Calculate 100 + 200 and get weather in Delhi"
    )

    # Assertions
    assert result["status"] == "completed"

    assert len(result["tool_results"]) == 2

    assert result["tool_results"][0]["tool"] == "calculator"
    assert result["tool_results"][0]["result"] == 300

    assert result["tool_results"][1]["tool"] == "get_weather"
    assert result["tool_results"][1]["result"] == "Weather in Delhi: Sunny"
