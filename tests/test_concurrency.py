import requests
import uuid
import threading
import time

API_URL = "http://localhost:8000"

def ingest_second_repo():
    print("Ingesting pypa/build...")
    res = requests.post(f"{API_URL}/ingest", json={"repo_url": "https://github.com/pypa/build"})
    print("Ingest result:", res.json())

def test_repo(repo_name, question, results_dict):
    payload = {
        "repo_name": repo_name,
        "session_id": str(uuid.uuid4()),
        "question": question
    }
    start = time.time()
    res = requests.post(f"{API_URL}/chat", json=payload)
    results_dict[repo_name] = {
        "status": res.status_code,
        "json": res.json() if res.status_code == 200 else res.text,
        "time": time.time() - start
    }

if __name__ == "__main__":
    ingest_second_repo()
    
    print("\nSending concurrent requests...")
    results = {}
    
    # We will launch both requests at the same time
    t1 = threading.Thread(target=test_repo, args=("sampleproject", "what calls build_and_check_dists", results))
    t2 = threading.Thread(target=test_repo, args=("build", "what is ProjectBuilder?", results))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    print("\nResults:")
    for repo, data in results.items():
        print(f"[{repo}] Time: {data['time']:.2f}s")
        print(f"[{repo}] Response: {data['json']}\n")
