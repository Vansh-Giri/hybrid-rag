import requests
import time

API_BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("--- STEP 1: Triggering Indexer ---")
    try:
        res = requests.post(f"{API_BASE_URL}/index")
        res.raise_for_status()
        print(f"Index Success: {res.json()['message']}")
    except Exception as e:
        print(f"Error indexing: {e}")
        return

    # Define test cases for edge-case checking
    test_queries = [
        # Test 1: Direct Factual Extraction
        "What is the exact hardware RAM requirement for this project?",
        
        # Test 2: Conceptual Synthesis
        "Explain why existing keyword-based search engines fail for technical documentation.",
        
        # Test 3: Out-of-bounds / Hallucination Check
        "How do you configure a Kubernetes cluster for load balancing?"
    ]

    print("\n--- STEP 2: Running Automated Queries ---")
    for i, query in enumerate(test_queries):
        print(f"\n[Test {i+1}] Query: '{query}'")
        start_time = time.time()
        
        try:
            payload = {"query": query, "top_k": 3}
            # Added a 45-second timeout to prevent infinite hanging
            response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=45)
            response.raise_for_status()
            
            data = response.json()
            latency = time.time() - start_time
            
            print(f"Answer (Took {latency:.2f}s): \n{data['answer']}")
            print(f"Sources Used: {[s['source'] for s in data['sources']]}")
            
            # THE FIX: Give your GPU 5 seconds to breathe and clear VRAM before the next test
            if i < len(test_queries) - 1:
                print("Cooling down GPU for 5 seconds...")
                time.sleep(5)
                
        except requests.exceptions.Timeout:
            print("Test Failed: LLM timed out after 45 seconds.")
        except Exception as e:
            print(f"Test Failed: {e}")

if __name__ == "__main__":
    run_tests()