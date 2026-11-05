
import agent


def test_execute_plan_previous_results(monkeypatch):

    # Track actual task execution
    executed_tasks = []

    # Mock task executor
    def mock_execution(task_description):

        executed_tasks.append(task_description)

        return {
            "status": "completed",
            "output": f"{task_description} completed"
        }

    # Replace actual executor
    monkeypatch.setattr(
        agent,
        "execute_task_with_tools",
        mock_execution
    )

    # Previous execution history
    previous_results = [
        {
            "task_id": 1,
            "task": "Task 1",
            "status": "completed",
            "results": {
                "answer": 100
            }
        }
    ]

    # New plan
    plan = {
        "goal": "Test previous results",
        "tasks": [
            {
                "id": 1,
                "task": "Task 1"
            },
            {
                "id": 2,
                "task": "Task 2"
            }
        ]
    }

    # Execute actual agent function
    results = agent.execute_plan(
        plan,
        previous_results=previous_results
    )

    # Verify previous result is preserved
    assert len(results) == 2

    assert results[0]["task_id"] == 1

    assert results[0]["results"]["answer"] == 100

    # Verify new task completed
    assert results[1]["task_id"] == 2

    assert results[1]["status"] == "completed"

    # Verify completed task was skipped
    assert executed_tasks == ["Task 2"]
