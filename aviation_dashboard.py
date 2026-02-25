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
TAF_SITES_META = {
    'KRDU': {'lat': 35.8776, 'lon': -78.7874},
    'KINT': {'lat': 36.1336, 'lon': -80.2222},
    'KGSO': {'lat': 36.0977, 'lon': -79.9373},
    'KFAY': {'lat': 34.9911, 'lon': -78.8803},
    'KRWI': {'lat': 35.8564, 'lon': -77.8919}
}
TAF_SITES = list(TAF_SITES_META.keys())
MODELS_VIS = ['GFS', 'NAM', 'RAP', 'HRRR', 'ARW', 'NEST']
MODELS_BUFKIT = ['nam', 'gfs', 'rap', 'hrrr', 'nest', 'arw']
DATA_DIR = "visibility_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Global storage for headers
model_init_strings = {} 

# --- 2. Helper Functions ---
def calculate_total_rh(t_c, td_c):
    rh_water = mpcalc.relative_humidity_from_dewpoint(t_c, td_c).magnitude * 100
    if t_c.magnitude >= 0: return rh_water
    T, Td = t_c.magnitude, td_c.magnitude
    e = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))
    e_s_ice = 6.112 * np.exp((22.46 * T) / (T + 272.62))
    return max(rh_water, (e / e_s_ice) * 100)

def find_nearest_gridpoint(ds, target_lat, target_lon):
    lat_arr, lon_arr = ds['latitude'].values, ds['longitude'].values
    if lon_arr.max() > 180 and target_lon < 0: target_lon += 360
    dist_sq = (lat_arr - target_lat)**2 + (lon_arr - target_lon)**2
    return np.unravel_index(np.argmin(dist_sq, axis=None), dist_sq.shape)

def format_visibility(vis_meters):
    if np.isnan(vis_meters): return "NA"
    vis_sm = min(vis_meters * 0.000621371, 10.0) 
    if vis_sm >= 1: return str(int(round(vis_sm)))
    elif vis_sm >= 0.75: return "3/4"
    elif vis_sm >= 0.50: return "1/2"
    else: return "1/4"

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

# --- 3. Bufkit Parsing Logic ---
def process_bufkit(filepath, model_name, mode='cig'):
    with open(filepath, 'r') as f: lines = f.readlines()
    selv_ft = 0.0
    for line in lines:
        if "SELV" in line:
            parts = line.split()
            selv_idx = parts.index('SELV')
            val_str = parts[selv_idx + 2] if parts[selv_idx + 1] == "=" else parts[selv_idx +
