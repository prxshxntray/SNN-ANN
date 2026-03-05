import pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
# paths to your 3 csv files (edit names)
files = [
    HERE / Path("pjm_wind_forecast_hourly.csv"),
    HERE / Path("pjm_wind_forecast_hourly_2.csv"),
    HERE / Path("pjm_wind_forecast_hourly_3.csv"),
]

dfs = [pd.read_csv(f) for f in files]
df_all = pd.concat(dfs, ignore_index=True)

# ensure datetime + sort
df_all["interval_start_local"] = pd.to_datetime(df_all["interval_start_local"])
df_all = df_all.sort_values("interval_start_local").reset_index(drop=True)

# (optional) drop exact duplicate rows
df_all = df_all.drop_duplicates()

# save merged file
out = HERE / "pjm_wind_forecast_hourly_17feb_5march.csv"
df_all.to_csv(out, index=False)
print("Saved:", out.resolve())