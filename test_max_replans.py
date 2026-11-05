
import agent


def test_max_replans(monkeypatch):

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

    # Mock execution: every task fails
    def mock_execute_plan(plan, previous_results=None):

        nonlocal execution_count
        execution_count += 1

        print(f"\nExecution attempt: {execution_count}")

        results = (
            previous_results.copy()
            if previous_results
            else []
        )

        results.append({
            "task_id": 1,
            "task": plan["tasks"][0]["task"],
            "status": "failed",
            "results": {
                "status": "failed",
                "error": "Simulated failure"
            }
        })

        return results

    # Mock re-planner
    def mock_replan(plan, failure_reason):

        nonlocal replan_count
        replan_count += 1

        print(f"Mock replan attempt: {replan_count}")

        return {
            "goal": plan["goal"],
            "tasks": [
                {
                    "id": 1,
                    "task": "Research RAG using another approach"
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

    # Execute orchestrator
    results = agent.orchestrate_plan("Research RAG")

    # Verify maximum execution attempts
    assert execution_count == 3

    # Verify maximum replans
    assert replan_count == 2

    # Verify final task failed
    assert results[-1]["status"] == "failed"
