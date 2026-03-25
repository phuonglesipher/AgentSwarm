from pathlib import Path

def build_gameplay_change_summary():
    return {
        "task_type": "feature",
        "implementation_status": "ready-for-review",
        "unit_tests": ["smoke"],
        "source_file": Path(__file__).name,
    }