import wavetoearth_core
import duckdb
import os

vcd_path = "tests/dummy.vcd"
parquet_path = "debug.parquet"

if os.path.exists(parquet_path):
    os.remove(parquet_path)

print(f"Converting {vcd_path}...")
parser = wavetoearth_core.WaveParser()
try:
    parser.convert_to_parquet(vcd_path, parquet_path, 1000)
    print("Conversion done.")
except Exception as e:
    print(f"Conversion failed: {e}")
    exit(1)

print("Inspecting Parquet...")
con = duckdb.connect()
con.execute(f"CREATE VIEW wave AS SELECT * FROM read_parquet('{parquet_path}')")
print("Signal Names:")
print(con.execute("SELECT DISTINCT signal_name FROM wave").fetchall())

print("All Data:")
print(con.execute("SELECT * FROM wave").fetchall())
