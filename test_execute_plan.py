import agent


def test_execute_plan_retry(monkeypatch):

    attempts = 0

    def mock_execution(task_description):

        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return {
                "status": "failed",
                "error": "Temporary failure"
            }

        return {
            "status": "completed",
            "tool_results": [
                {
                    "tool": "calculator",
                    "result": 100
                }
            ]
        }

    monkeypatch.setattr(
        agent,
        "execute_task_with_tools",
        mock_execution
    )

    test_plan = {
        "goal": "Test retry handling",
        "tasks": [
            {
                "id": 1,
                "task": "Calculate 25 multiplied by 4"
            }
        ]
    }

    results = agent.execute_plan(test_plan)

    assert attempts == 2
    assert results[0]["status"] == "completed"
    assert results[0]["results"]["tool_results"][0]["result"] == 100