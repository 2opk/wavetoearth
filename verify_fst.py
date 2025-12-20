import requests
import time
import subprocess
import os

def run_verification():
    # Start Server
    server_process = subprocess.Popen(["python3.10", "cli.py", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2) # Wait for startup

    try:
        # 1. Load FST
        print("Loading FST file...")
        fst_path = os.path.abspath("tests/simple.fst")
        resp = requests.post("http://localhost:8000/load", json={"file_path": fst_path})
        print("Load Response:", resp.json())
        assert resp.status_code == 200, f"Load failed: {resp.text}"

        # 2. List Signals
        resp = requests.get("http://localhost:8000/signals")
        signals = resp.json()['signals']
        print("Signals:", signals)
        assert "top.clk" in signals
        assert "top.data" in signals

        # 3. Query Data
        resp = requests.post("http://localhost:8000/query", json={"signal": "top.clk", "start": 0, "end": 20})
        data = resp.json()
        print("Query Data:", data)
        # Check values
        changes = data['changes']
        # dummy.vcd content: #0 0, #5 1, #10 0, #15 1, #20 0
        expected_times = [0, 5, 10, 15, 20]
        times = [c[0] for c in changes]
        assert times == expected_times, f"Expected times {expected_times}, got {times}"

        print("\nSUCCESS: FST Verification Passed!")

    except Exception as e:
        print(f"\nFAILURE: {e}")
        # Kill server to read output
        server_process.kill()
        outs, errs = server_process.communicate()
        print("Server Output:", outs)
        print("Server Error:", errs)
        exit(1)
    finally:
        server_process.kill()

if __name__ == "__main__":
    run_verification()
