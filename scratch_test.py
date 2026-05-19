import pandas as pd
import datetime

file1 = '/Users/dashdondoggansaikhan/Downloads/DATA_time (1) (1).xlsx'
df = pd.read_excel(file1, sheet_name='Sheet1', engine='openpyxl')

def parse_time_to_seconds(val):
    if pd.isna(val):
        return None
    if isinstance(val, datetime.time):
        return val.hour * 60 + val.minute + (val.second / 60.0) # Actually, hour is minute, minute is second.
    if isinstance(val, str):
        val = val.strip().lower()
        if val.endswith('м') or val.endswith('m'):
            val = val[:-1]
        parts = val.split(':')
        if len(parts) >= 2:
            return int(parts[0]) * 60 + int(parts[1])
    return None

df['seconds'] = df['Цаг'].apply(parse_time_to_seconds)
# Filter valid rows
valid_df = df.dropna(subset=['seconds', 'Ярьж буй хүн']).copy()
valid_df['Ярьж буй хүн'] = valid_df['Ярьж буй хүн'].astype(str).str.strip().str.upper()

# Calculate durations
valid_df = valid_df.sort_values('seconds')
valid_df['next_seconds'] = valid_df['seconds'].shift(-1)
valid_df['duration'] = valid_df['next_seconds'] - valid_df['seconds']

s_time = valid_df[valid_df['Ярьж буй хүн'] == 'S']['duration'].sum()
t_time = valid_df[valid_df['Ярьж буй хүн'] == 'T']['duration'].sum()

print(f"S time: {s_time} seconds ({s_time/60:.2f} mins)")
print(f"T time: {t_time} seconds ({t_time/60:.2f} mins)")
