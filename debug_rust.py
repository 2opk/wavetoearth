import wavetoearth_core
import duckdb
import os

vcd_path = "tests/dummy.vcd"
parquet_path = "debug.parquet"
meta_path = parquet_path + ".meta.parquet"

if os.path.exists(parquet_path):
    os.remove(parquet_path)
if os.path.exists(meta_path):
    os.remove(meta_path)

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
if os.path.exists(meta_path):
    con.execute(f"CREATE VIEW signals AS SELECT * FROM read_parquet('{meta_path}')")
print("Signal Names:")
if os.path.exists(meta_path):
    print(con.execute("SELECT signal_id, signal_name FROM signals ORDER BY signal_id LIMIT 10").fetchall())
else:
    print("No meta parquet found.")

print("All Data:")
print(con.execute("SELECT * FROM wave").fetchall())
