import requests
import uuid

API_URL = "http://localhost:8000"

def test_api():
    print("Testing /health")
    res = requests.get(f"{API_URL}/health")
    print("Health:", res.json())

    print("\nTesting /repos")
    res = requests.get(f"{API_URL}/repos")
    print("Repos:", res.json())

    print("\nTesting /chat (Question 1)")
    payload = {
        "repo_name": "sampleproject",
        "session_id": str(uuid.uuid4()),
        "question": "what encoding should the readme file use"
    }
    res = requests.post(f"{API_URL}/chat", json=payload)
    print("Chat Q1:", res.json())

    print("\nTesting /chat (Question 2)")
    payload["question"] = "what calls build_and_check_dists"
    res = requests.post(f"{API_URL}/chat", json=payload)
    print("Chat Q2:", res.json())

if __name__ == "__main__":
    test_api()
