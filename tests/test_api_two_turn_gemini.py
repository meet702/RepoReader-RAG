import json
import requests


API_URL = "http://127.0.0.1:8000"
SESSION_ID = "gemini-content-normalization-smoke"

questions = [
    "can you tell me the file structure of this project",
    "give me the code of the task controller file",
]

for index, question in enumerate(questions, start=1):
    print("\n" + "=" * 80)
    print(f"REQUEST {index}")
    print(json.dumps(
        {
            "repo_name": "To-Do-list",
            "session_id": SESSION_ID,
            "question": question,
        },
        indent=2,
    ))

    response = requests.post(
        f"{API_URL}/chat",
        json={
            "repo_name": "To-Do-list",
            "session_id": SESSION_ID,
            "question": question,
        },
        timeout=180,
    )

    print(f"STATUS: {response.status_code}")
    print("RESPONSE:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
