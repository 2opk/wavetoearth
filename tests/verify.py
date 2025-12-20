import subprocess
import time
import requests
import sys
import os
import signal

# Create dummy VCD
VCD_CONTENT = """$date today $end
$version 1.0 $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 8 " data $end
$upscope $end
$enddefinitions $end
#0
0!
b00000000 "
#5
1!
#10
0!
b11111111 "
#15
1!
#20
0!
"""

def create_vcd():
    with open("tests/dummy.vcd", "w") as f:
        f.write(VCD_CONTENT)
    print("Created tests/dummy.vcd")

def run_verification():
    create_vcd()

    # Start Server
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", "8008"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server
    time.sleep(2)

    BASE_URL = "http://localhost:8008"

    try:
        # 1. Health Check
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {resp.status_code}")
        assert resp.status_code == 200

        # 2. Load File
        print("Loading file...")
        abs_path = os.path.abspath("tests/dummy.vcd")
        resp = requests.post(f"{BASE_URL}/load", json={"file_path": abs_path})
        print(f"Load Response: {resp.json()}")
        assert resp.status_code == 200

        # 3. List Signals
        resp = requests.get(f"{BASE_URL}/signals")
        signals = resp.json()["signals"]
        print(f"Signals: {signals}")
        assert "top.clk" in signals
        assert "top.data" in signals

        # 4. Query Data
        resp = requests.post(f"{BASE_URL}/query", json={"signal": "top.clk", "start": 0, "end": 20})
        data = resp.json()
        print(f"Query Data: {data}")
        # Expected changes: (0, '0'), (5, '1'), (10, '0'), (15, '1'), (20, '0')
        assert len(data['changes']) >= 4

        # 5. Analyze Data
        resp = requests.post(f"{BASE_URL}/analyze", json={"signal": "top.clk", "start": 0, "end": 20})
        analysis = resp.json()
        print(f"Analysis: {analysis}")
        assert analysis['toggle_count'] > 0
        assert "changed" in analysis['description']

        print("\nSUCCESS: All verification steps passed!")

    except Exception as e:
        print(f"\nFAILURE: {e}")
        # Print server logs if failed
        outs, errs = server_process.communicate(timeout=1)
        print("Server Output:", outs.decode())
        print("Server Error:", errs.decode())
    finally:
        server_process.terminate()

if __name__ == "__main__":
    run_verification()
