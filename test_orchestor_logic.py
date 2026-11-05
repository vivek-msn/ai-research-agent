
def test_latest_result_overwrites_old_result():

    # Step 1: Task history
    results = [
        {
            "task_id": 1,
            "status": "completed"
        },
        {
            "task_id": 2,
            "status": "failed"
        },
        {
            "task_id": 1,
            "status": "failed"
        }
    ]

    # Step 2: Keep latest result for each task
    latest_results = {}

    for task in results:
        task_id = task.get("task_id")
        latest_results[task_id] = task

    # Step 3: Verify latest result overwrites old result
    assert latest_results[1]["status"] == "failed"

    # Step 4: Identify failed tasks
    failed_tasks = [
        task for task in latest_results.values()
        if task.get("status") == "failed"
    ]

    # Task 1 latest status = failed
    # Task 2 status = failed
    assert len(failed_tasks) == 2

    # Step 5: Verify plan is not completed
    plan_completed = (
        bool(latest_results)
        and all(
            task.get("status") == "completed"
            for task in latest_results.values()
        )
    )

    assert plan_completed is False
