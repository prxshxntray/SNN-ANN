from pathlib import Path
import os
from dotenv import load_dotenv
from gridstatusio import GridStatusClient

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GRIDSTATUS_API_KEY")
if not api_key:
    raise RuntimeError("Missing GRIDSTATUS_API_KEY. Add it to .env or export it in your shell.")

client = GridStatusClient(api_key)
df = client.get_dataset(
  dataset="pjm_wind_forecast_hourly",
  start="2026-02-17",
  end="2026-02-24",
  publish_time="latest",
  timezone="market",
)

out_path = Path(__file__).parent / "datasets" / "pjm_wind_forecast_hourly_3.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(out_path, index=False)
print("Saved to:", out_path)