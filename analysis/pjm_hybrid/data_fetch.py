"""
PJM Wind Forecast Data Fetcher
===============================
Fetches hourly wind-forecast data from PJM via the GridStatus API.

Usage
-----
  # Fetch 6 months of data (default):
  python data_fetch.py

  # Custom date range:
  python data_fetch.py --start 2025-06-01 --end 2025-12-01

  # Custom output path:
  python data_fetch.py --output datasets/my_data.csv

Requirements
------------
  pip install gridstatusio python-dotenv
  Create a .env file with: GRIDSTATUS_API_KEY=your_key_here
"""

import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
from gridstatusio import GridStatusClient

# Load API key from .env
load_dotenv(Path(__file__).parent / ".env")


def fetch_pjm_wind(start: str, end: str, output: str) -> None:
    """
    Fetch PJM wind forecast hourly data and save to CSV.

    Parameters
    ----------
    start : str   — start date (YYYY-MM-DD)
    end   : str   — end date (YYYY-MM-DD)
    output : str  — output CSV path
    """
    api_key = os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GRIDSTATUS_API_KEY. "
            "Add it to analysis/.env or export it in your shell."
        )

    client = GridStatusClient(api_key)

    print(f"Fetching pjm_wind_forecast_hourly: {start} → {end} ...")
    df = client.get_dataset(
        dataset="pjm_wind_forecast_hourly",
        start=start,
        end=end,
        publish_time="latest",
        timezone="market",
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Fetch PJM wind forecast data")
    parser.add_argument("--start", default="2025-09-01",
                        help="Start date YYYY-MM-DD (default: 2025-09-01)")
    parser.add_argument("--end",   default="2026-03-05",
                        help="End date YYYY-MM-DD (default: 2026-03-05)")
    parser.add_argument("--output", default="datasets/pjm_wind_full.csv",
                        help="Output CSV path (default: datasets/pjm_wind_full.csv)")
    args = parser.parse_args()
    fetch_pjm_wind(args.start, args.end, args.output)


if __name__ == "__main__":
    main()