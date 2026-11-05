
import agent


def test_orchestrator_replan_success(monkeypatch):

    # Track execution and replan attempts
    execution_count = 0
    replan_count = 0

    # Mock initial planner
    def mock_generate_plan(goal):

        return {
            "goal": goal,
            "tasks": [
                {
                    "id": 1,
                    "task": "Research RAG"
                }
            ]
        }

    # Mock execution
    def mock_execute_plan(plan, previous_results=None):

        nonlocal execution_count
        execution_count += 1

        print(f"\nExecution attempt: {execution_count}")

        # First execution fails
        if execution_count == 1:

            return [
                {
                    "task_id": 1,
                    "task": "Research RAG",
                    "status": "failed",
                    "results": {
                        "status": "failed",
                        "error": "Temporary failure"
                    }
                }
            ]

        # Second execution succeeds
        results = (
            previous_results.copy()
            if previous_results
            else []
        )

        results.append({
            "task_id": 1,
            "task": "Research RAG",
            "status": "completed",
            "results": {
                "status": "completed",
                "output": "Research completed"
            }
        })

        return results

    # Mock re-planning
    def mock_replan(plan, failure_reason):

        nonlocal replan_count
        replan_count += 1

        print("\nMock re-planning triggered!")
        print("Failure reason:", failure_reason)

        return {
            "goal": plan["goal"],
            "tasks": [
                {
                    "id": 1,
                    "task": "Research RAG using an alternative approach"
                }
            ]
        }

    # Replace actual functions with mocks
    monkeypatch.setattr(
        agent,
        "generate_plan",
        mock_generate_plan
    )

    monkeypatch.setattr(
        agent,
        "execute_plan",
        mock_execute_plan
    )

    monkeypatch.setattr(
        agent,
        "replan",
        mock_replan
    )

    # Run actual orchestrator
    results = agent.orchestrate_plan("Research RAG")

    # Verify execution attempts
    assert execution_count == 2

    # Verify re-planning
    assert replan_count == 1

    # Verify final task status
    assert results[-1]["status"] == "completed"
