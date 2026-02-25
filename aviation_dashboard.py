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

# Global storage for initialization times for headers
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
    if val == "NA" or val == "--": return 'background-color: white;'
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

# --- 3. Parsing Logic ---
def process_bufkit(filepath, model_name, mode='cig'):
    with open(filepath, 'r') as f: lines = f.readlines()
    selv_ft = 0.0
    for line in lines:
        if "SELV" in line:
            parts = line.split()
            selv_idx = parts.index('SELV')
            val_str = parts[selv_idx + 2] if parts[selv_idx + 1] == "=" else parts[selv_idx + 1]
            selv_ft = float(val_str) * 3.28084
            break
            
    times, profile_starts = [], []
    for i, line in enumerate(lines):
        if "TIME =" in line:
            match = re.search(r'TIME =\s*(\d{6}/\d{4})', line)
            if match:
                dt = datetime.strptime(match.group(1), "%y%m%d/%H%M").replace(tzinfo=timezone.utc)
                times.append(dt)
                if model_name not in model_init_strings:
                    model_init_strings[model_name] = dt.strftime('%m/%d %HZ')
        if "PRES TMPC" in line: profile_starts.append(i)

    results = []
    is_gfs = (model_name.lower() == "gfs")
    
    for i, start_idx in enumerate(profile_starts):
        end_idx = profile_starts[i+1] if i + 1 < len(profile_starts) else len(lines)
        data_lines = [l.split() for l in lines[start_idx+2 : end_idx] if l.strip() and "STN" not in l and "STIM" not in l]
        
        if mode == 'cig':
            lowest_ft = np.nan
            for j in range(0, len(data_lines)-1, 2):
                l1, l2 = data_lines[j], data_lines[j+1]
                if len(l1) < 4 or len(l2) < 1: continue
                try:
                    t_c, td_c = float(l1[1]) * units.degC, float(l1[3]) * units.degC
                    h_agl = ((float(l2[0]) if is_gfs else float(l2[1])) * 3.28084) - selv_ft
                    if 200 <= h_agl <= 38000 and calculate_total_rh(t_c, td_c) >= 95.0:
                        lowest_ft = h_agl; break
                except: continue
            results.append(str(int(round(lowest_ft/100)*100)) if not np.isnan(lowest_ft) else "--")
        
        else: # LLWS
            heights, dirs, spds = [], [], []
            for j in range(0, len(data_lines)-1, 2):
                l1, l2 = data_lines[j], data_lines[j+1]
                if len(l1) < 7 or len(l2) < 1: continue
                try:
                    h_agl = ((float(l2[0]) if is_gfs else float(l2[1])) * 3.28084) - selv_ft
                    if h_agl < 2100: heights.append(h_agl); dirs.append(float(l1[5])); spds.append(float(l1[6]))
                except: continue
            
            max_s, t_h, t_d, t_s = 0.0, 0, 0, 0
            for l in range(len(heights)-1):
                for u in range(l+1, len(heights)):
                    dr_l, dr_u = np.radians(dirs[l]), np.radians(dirs[u])
                    s = np.sqrt(max(0, spds[l]**2 + spds[u]**2 - 2*spds[l]*spds[u]*np.cos(dr_u-dr_l)))
                    if s > max_s: max_s, t_h, t_d, t_s = s, heights[u], dirs[u], spds[u]
            
            if max_s >= 20:
                results.append(f"{int(round(max_s))}|WS{int(round(t_h))//100:03d}/{int(round(t_d/10)*10):03d}{int(round(t_s/5)*5):02d}KT")
            else: results.append("--")
            
    return pd.Series(results, index=times, name=model_name.upper())

# --- 4. Main Update Loop ---
def update():
    now = datetime.now(timezone.utc)
    ymd_curr = now.strftime("%Y%m%d")
    ymd_prior = (now - timedelta(days=1)).strftime("%Y%m%d")
    curr_h = now.hour

    # Timing Logic
    gfs_s, gfs_d = ("18", ymd_prior) if curr_h < 4 else ("00", ymd_curr) if curr_h < 10 else ("06", ymd_curr) if curr_h < 16 else ("12", ymd_curr) if curr_h < 22 else ("18", ymd_curr)
    nam_s, nam_d = gfs_s, gfs_d
    arw_s, arw_d = ("12", ymd_prior) if curr_h < 4 else ("00", ymd_curr) if curr_h < 16 else ("12", ymd_curr)
    rap_t = now - timedelta(hours=2)
    rap_s, rap_d = rap_t.strftime("%H"), rap_t.strftime("%Y%m%d")

    # Downloads (GRIB & Bufkit)
    print("Downloading...")
    for hr in [f"{i:03d}" for i in range(1, 49)]:
        download_file(f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?dir=%2Fgfs.{gfs_d}%2F{gfs_s}%2Fatmos&file=gfs.t{gfs_s}z.pgrb2.0p25.f{hr}&var_VIS=on&lev_surface=on&subregion=&toplat=40&leftlon=278&rightlon=285&bottomlat=30", os.path.join(DATA_DIR, f"gfs.f{hr}.grib2"))
    
    buf_urls = {
        'nam': "http://www.meteo.psu.edu/bufkit/data/latest/nam_{site}.buf",
        'gfs': "http://www.meteo.psu.edu/bufkit/data/GFS/latest/gfs3_{site}.buf",
        'rap': "http://www.meteo.psu.edu/bufkit/data/RAP/latest/rap_{site}.buf",
        'hrrr': "https://www.meteo.psu.edu/bufkit/data/HRRR/latest/hrrr_{site}.buf",
        'nest': "https://www.meteo.psu.edu/bufkit/data/NAMNEST/latest/namnest_{site}.buf",
        'arw': "https://www.meteo.psu.edu/bufkit/data/HIRESW/latest/hiresw_{site}.buf"
    }
    for s in TAF_SITES:
        for m, u in buf_urls.items(): download_file(u.format(site=s.lower()), os.path.join(DATA_DIR, f"{m}_{s.lower()}.buf"))

    # Processing
    print("Processing...")
    v_dfs, c_dfs, l_dfs = {s: pd.DataFrame() for s in TAF_SITES}, {s: pd.DataFrame() for s in TAF_SITES}, {s: pd.DataFrame() for s in TAF_SITES}
    
    # Process GRIB Visibility
    for m in MODELS_VIS:
        files = sorted(glob.glob(os.path.join(DATA_DIR, f"{m.lower()}*.grib2")))
        if not files: continue
        ds = xr.open_mfdataset(files, engine='cfgrib', combine='nested', concat_dim='valid_time', coords="minimal", compat="override")
        model_init_strings[m.lower()] = pd.to_datetime(ds.initial_time.values).strftime('%m/%d %HZ')
        for s, co in TAF_SITES_META.items():
            idx = find_nearest_gridpoint(ds, co['lat'], co['lon'])
            vals = [format_visibility(v) for v in ds['vis'].values[:, idx[0], idx[1]]]
            v_dfs[s] = v_dfs[s].join(pd.DataFrame({m: vals}, index=pd.to_datetime(ds.valid_time.values)), how='outer')

    # Process Bufkit CIG/LLWS
    for s in TAF_SITES:
        for m in MODELS_BUFKIT:
            path = os.path.join(DATA_DIR, f"{m}_{s.lower()}.buf")
            if os.path.exists(path):
                c_dfs[s] = c_dfs[s].join(process_bufkit(path, m, 'cig'), how='outer')
                l_dfs[s] = l_dfs[s].join(process_bufkit(path, m, 'llws'), how='outer')

    # Generate HTML
    print("Generating HTML...")
    cur_ts = pd.Timestamp.utcnow().tz_localize(None).replace(minute=0, second=0, microsecond=0)
    html_tbls = {}

    def to_st_html(df, pt):
        if df.empty: return "<p>N/A</p>"
        d = df.copy()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d = d[~d.index.duplicated()].sort_index()
        d = d[d.index >= cur_ts].head(42)
        d.columns = [f"{c} [{model_init_strings.get(c.lower(), '??')}]" for c in d.columns]
        d.index = d.index.strftime('%d/%H'); d.index.name = "Time (UTC)"
        st = d.style.map(colorize_flight_rules) if pt=='vis' else d.style.map(style_ceiling_table) if pt=='cig' else d.style.format(lambda v: str(v).split('|')[-1]).map(style_llws_table)
        return st.to_html(escape=False)

    for s in TAF_SITES:
        ss = s.lower()[1:]
        html_tbls[f"vis_{ss}"] = to_st_html(v_dfs[s], 'vis')
        html_tbls[f"cig_{ss}"] = to_st_html(c_dfs[s], 'cig')
        html_tbls[f"llws_{ss}"] = to_st_html(l_dfs[s], 'llws')

    # Final HTML string assembly... (Use your dashboard_html template here, injecting html_tbls via JSON)
    # [Insert your full Cell 6 HTML Template here]

if __name__ == "__main__":
    update()
