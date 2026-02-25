import os
import re
import glob
import time
import json
import requests
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime, timedelta, timezone

warnings.filterwarnings('ignore')

# --- 1. Configuration ---
TAF_SITES = ['KRDU', 'KINT', 'KGSO', 'KFAY', 'KRWI']
MODELS_VIS = ['GFS', 'NAM', 'RAP', 'HRRR', 'ARW', 'NEST']
DATA_DIR = "visibility_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Global storage for initialization times
model_run_times = {} 

# --- 2. Helper Functions ---
def calculate_total_rh(t_c, td_c):
    rh_water = mpcalc.relative_humidity_from_dewpoint(t_c, td_c).magnitude * 100
    if t_c.magnitude >= 0: return rh_water
    T, Td = t_c.magnitude, td_c.magnitude
    e = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))
    e_s_ice = 6.112 * np.exp((22.46 * T) / (T + 272.62))
    return max(rh_water, (e / e_s_ice) * 100)

def colorize_flight_rules(val):
    if val == "NA" or val == "--": return 'background-color: white; color: black;'
    try:
        f = 0.25 if val == "1/4" else 0.5 if val == "1/2" else 0.75 if val == "3/4" else float(val)
        if f > 5: return 'background-color: white;'
        elif 3 <= f <= 5: return 'background-color: #458B00; color: white;'
        elif 1 <= f < 3: return 'background-color: #CD3333; color: white;'
        else: return 'background-color: #EE82EE; color: black;'
    except: return ''

def style_ceiling_table(val):
    if val == "NA" or val == "--": return 'background-color: white;'
    try:
        h = int(val)
        if h > 3000: return 'background-color: white;'
        elif 1000 <= h <= 3000: return 'background-color: #458B00; color: white;'
        elif 500 <= h < 1000: return 'background-color: #CD3333; color: white;'
        else: return 'background-color: #EE82EE; color: black;'
    except: return ''

def style_llws_table(val):
    if "|" not in str(val): return 'background-color: white;'
    mag = float(val.split('|')[0])
    if mag < 20: return 'background-color: white;'
    elif 20 <= mag < 30: return 'background-color: #FFC125; color: black;'
    elif 30 <= mag < 40: return 'background-color: #CD5B45; color: white;'
    else: return 'background-color: #7A378B; color: white;'

def download_file(url, filepath):
    for _ in range(3):
        try:
            r = requests.get(url, timeout=15, stream=True)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            return True
        except: time.sleep(1)
    return False

# --- 3. Execution Logic ---
now = datetime.now(timezone.utc)
ymd_curr = now.strftime("%Y%m%d")
ymd_prior = (now - timedelta(days=1)).strftime("%Y%m%d")

# Simple Cycle Logic
curr_h = now.hour
gfs_start, gfs_date = ("18", ymd_prior) if curr_h < 4 else ("00", ymd_curr) if curr_h < 10 else ("06", ymd_curr) if curr_h < 16 else ("12", ymd_curr) if curr_h < 22 else ("18", ymd_curr)

print("Starting Data Downloads...")
# GRIB Visibility Fetch (Truncated for readability - apply your NOMADS loops here)
# ... [Insert your NOMADS Download Loops from Cell 2] ...

# Bufkit Fetch
bufkit_urls = {
    'nam': "http://www.meteo.psu.edu/bufkit/data/latest/nam_{site}.buf",
    'gfs': "http://www.meteo.psu.edu/bufkit/data/GFS/latest/gfs3_{site}.buf",
    'rap': "http://www.meteo.psu.edu/bufkit/data/RAP/latest/rap_{site}.buf",
    'hrrr': "https://www.meteo.psu.edu/bufkit/data/HRRR/latest/hrrr_{site}.buf",
    'nest': "https://www.meteo.psu.edu/bufkit/data/NAMNEST/latest/namnest_{site}.buf",
    'arw': "https://www.meteo.psu.edu/bufkit/data/HIRESW/latest/hiresw_{site}.buf"
}

for site in TAF_SITES:
    for mod, url_template in bufkit_urls.items():
        download_file(url_template.format(site=site.lower()), os.path.join(DATA_DIR, f"{mod}_{site.lower()}.buf"))

# --- 4. Processing ---
vis_dfs, cig_dfs, llws_dfs = {}, {}, {}

print("Processing Data...")
# [Insert Cell 3, 4, 5 logic here, ensuring you update model_run_times dict]

# --- 5. Final Dashboard Export ---
html_output = {}
# [Insert Cell 6 Dashboard logic here]

with open("index.html", "w") as f:
    f.write(dashboard_html)
print("index.html generated.")