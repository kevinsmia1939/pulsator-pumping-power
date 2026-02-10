import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator

mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['font.family'] = 'sans'
mpl.rcParams['font.serif'] = 'Liberation San'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

SEC_PER_MIN = 60.0
ML_TO_M3 = 1e-6

def lowess_func(x, y, frac: float):
    """
    LOWESS smoother (numeric). If frac<=0, pass data through but sorted by x.
    Returns (x_sorted, y_smoothed)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if frac is None or frac <= 0:
        order = np.argsort(x)
        return x[order], y[order]
    low = sm.nonparametric.lowess(y, x, frac=frac)
    return low[:, 0], low[:, 1]

def savgol_func(x, y, window_frac: float, polyorder: int = 3):
    """
    Savitzky–Golay smoother.
    If window_frac <= 0, returns data sorted by x without smoothing.

    Parameters
    ----------
    x : array-like
        x-values
    y : array-like
        y-values
    window_frac : float
        Fraction of data points to use as the smoothing window
        (0 < window_frac <= 1). It will be converted to an odd integer window size.
    polyorder : int, default=3
        Polynomial order for the filter (must be < window_length).

    Returns
    -------
    x_sorted : ndarray
        Sorted x-values
    y_smooth : ndarray
        Smoothed y-values
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sort data by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    if window_frac is None or window_frac <= 0:
        # No smoothing, just return sorted data
        return x_sorted, y_sorted

    n = len(x_sorted)
    # Convert fraction to window length (must be odd and at least polyorder+2)
    window_length = max(int(window_frac * n), polyorder + 2)
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, n if n % 2 == 1 else n - 1)
    y_smooth = savgol_filter(y_sorted, window_length=window_length, polyorder=polyorder, mode="interp")

    return x_sorted, y_smooth

def sine_func_numeric(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def track2df(file, t_lim1, t_lim2, frac,netflow_ml_min,vol_len,marker_dis,peak_distance,peakheight=0):
    """
    Read tracking CSV → return:
      df_trim (trimmed numeric table with time_LED [s], y [px]),
      t [s], y [px], flowrate [px/s],
      t_smh [s], smh_y [px], smh_flowrate [px/s],
      roi_px_per_5cm [px per 5 cm]
    """
    df = pd.read_csv(file, header=None)
    if df.shape[1] < 5 or df.shape[0] < 3:
        raise ValueError("Unexpected track file format: need at least 5 columns and 3 header rows.")
    ROI_height_val = float(df.iloc[1, 4])  # pixels per 5 cm (value only)
    if ROI_height_val <= 0:
        raise ValueError("ROI height (pixels per 5 cm) must be positive.")
    roi_px_per_5cm = ROI_height_val

    # strip header & unused cols
    df = df.drop(df.index[0:3])
    df = df.drop(df.columns[3:5], axis=1).dropna().reset_index(drop=True)
    df.columns = ["time(s)", "x", "y"]  # time [s], x/y [px]
    t_raw = pd.to_numeric(df["time(s)"], errors="coerce").to_numpy(dtype=float)
    
    # rebuild per-second linearized time like original
    index = np.where(np.diff(t_raw) != 0)[0] + 1
    df = df.drop(df.index[0:index[0]]).reset_index(drop=True)
    t_raw = pd.to_numeric(df["time(s)"], errors='coerce').to_numpy(dtype=float)
    #Cut off the first second
    index = np.where(np.diff(t_raw) != 0)[0] + 1
    idx_group = np.split(t_raw, index)
    inter_time = []
    for i in idx_group:
        inter_time_1s = np.linspace(i[0], i[0] + 1, num=len(i) + 1)[:-1]
        inter_time.append(inter_time_1s)
        
    df["time(s)"] = np.concatenate(inter_time)  # seconds
    df = df[df["time(s)"].between(t_lim1, t_lim2)]
    df["time(s)"] = df["time(s)"]-t_lim1
    df = df.reset_index(drop=True)
    df["y"] = pd.to_numeric(df['y'], errors='coerce').to_numpy(dtype=float)
    
    # smooth on trimmed data
    _, smh_y = savgol_func(df["time(s)"], df["y"], frac)
    df["smooth y"] = smh_y 
    df["flow rate mlmin"] = (np.gradient(np.asarray(df["y"], dtype=float), np.asarray(df["time(s)"], dtype=float))/roi_px_per_5cm*marker_dis*vol_len*SEC_PER_MIN)+netflow_ml_min
    df["flow rate mlmin smh"] = (np.gradient(np.asarray(df["smooth y"], dtype=float), np.asarray(df["time(s)"], dtype=float))/roi_px_per_5cm*marker_dis*vol_len*SEC_PER_MIN)+netflow_ml_min
    
    df = df.reset_index(drop=True)
    peaks, _ = find_peaks(df["flow rate mlmin smh"], height=peakheight,distance=peak_distance)
    df["peak"] = 0
    df.loc[peaks, "peak"] = 1
    
    crest, _ = find_peaks(-df["flow rate mlmin smh"], height=peakheight,distance=peak_distance)
    df["crest"] = 0
    df.loc[crest, "crest"] = 1
    return df

def pressure2df(file, time1, time2,t_lim1, t_lim2, frac, peak_distance):
    """
    Read pressure CSV with columns: time(s), P0, P1, dP.
    Returns: df
    """
    dfP = pd.read_csv(file)
    dfP.columns = ['time(s)', 'P0', 'P1', 'dP(Pa)']
    dfP["time(s)"] = dfP["time(s)"]-(5.831/1000) #subtract measured delay
    # Baseline from [time1, time2]
    baselineP = np.average(dfP[dfP["time(s)"].between(time1, time2)]["dP(Pa)"])
    
    dfP["dP(Pa)"] = dfP["dP(Pa)"]-baselineP
    dfP = dfP[dfP["time(s)"].between(t_lim1, t_lim2)]
    dfP['time(s)'] = dfP['time(s)'] - t_lim1
    # print(dfP)
    
    _, smh_dP = savgol_func(dfP["time(s)"], dfP["dP(Pa)"], frac)
    dfP["smooth dP(Pa)"] = smh_dP 
    
    dfP = dfP.reset_index(drop=True)
    peaks, _ = find_peaks(dfP["smooth dP(Pa)"], height=0,distance=peak_distance)
    dfP["peak"] = 0
    dfP.loc[peaks, "peak"] = 1
    
    crest, _ = find_peaks(-dfP["smooth dP(Pa)"], height=0,distance=peak_distance)
    dfP["crest"] = 0
    dfP.loc[crest, "crest"] = 1
    return dfP

def calpower(file_track, file_press, frac_vel, frac_press,
             bl_t1, bl_t2, t_lim1, t_lim2, netflow_ml_min,vol_len,marker_dis,electrode_size,vel_peak_distance,press_peak_distance):

    # Trim both streams but with margin for smoothing artifact
    t_lim1 = t_lim1-0.5
    t_lim2 = t_lim2+0.5
    df_vel = track2df(file_track, t_lim1, t_lim2, frac_vel,netflow_ml_min,vol_len,marker_dis,vel_peak_distance)
    df_P = pressure2df(file_press, bl_t1, bl_t2, t_lim1, t_lim2, frac_press,press_peak_distance)

    dp_interp = np.interp(df_vel["time(s)"], df_P["time(s)"], df_P["dP(Pa)"])   # dP evaluated at each t_flow
    dp_smh_interp = np.interp(df_vel["time(s)"], df_P["time(s)"], df_P["smooth dP(Pa)"])   # dP evaluated at each t_flow
    
    power = df_vel["flow rate mlmin"]*ML_TO_M3/SEC_PER_MIN*dp_interp #W
    power_smh = df_vel["flow rate mlmin smh"]*ML_TO_M3/SEC_PER_MIN*dp_smh_interp #W
    df_power = pd.DataFrame({"time(s)":df_vel["time(s)"],'Power(W)':power})
    df_power["Power smh(W)"] = power_smh
    
    avg_power = (np.trapezoid(df_power["Power smh(W)"], df_power["time(s)"]))/max(df_power["time(s)"])
    avg_power_Wm3 = avg_power / electrode_size  # W/m^3
    return df_vel,df_P,df_power,avg_power_Wm3

def plotdata(df_vel,df_P,df_power,avg_power_Wm3,flowlim,presslim,powerlim,plot_vel_peak,plot_pressure_peak,plotgraph,t_shift,customtext):

    t_lim = max(df_vel["time(s)"])
    if plotgraph == True:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        # t_shift = 0.5 # shift plot by some amount of time for nice looking plot
        # Velocity axis
        ax1.plot(df_vel["time(s)"]-0.5,df_vel["flow rate mlmin"],linestyle="--", linewidth=1, alpha=0.7, color='red')
        ax1.plot(df_vel["time(s)"]-0.5,df_vel["flow rate mlmin smh"], linestyle="-", linewidth=1, alpha=1, color='red')
        ax1.set_xlim(0, t_lim-1)
        ax1.set_ylim(-flowlim,flowlim)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Flow rate (mL/min)", color='red')
    
        # Pressure axis
        ax2 = ax1.twinx()
        ax2.plot(df_P["time(s)"]-0.5, df_P["dP(Pa)"]/1000, linestyle="--", linewidth=1, alpha=0.7, color='blue')
        ax2.plot(df_P["time(s)"]-0.5, df_P["smooth dP(Pa)"]/1000, linestyle="-", linewidth=1, alpha=1, color='blue')
        ax2.set_xlim(0, t_lim-1)
        ax2.set_ylim(-presslim, presslim)
        ax2.set_ylabel("Pressure drop (kPa)", color='blue')
        
        # Power axis
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 80))
        ax3.set_ylabel("Pumping power (mW)", color='green')
        ax3.plot(df_power["time(s)"]-0.5, df_power["Power smh(W)"]*1000, '-', label="Power", color="limegreen")
        ax3.hlines(0, 0, t_lim, linestyle="--", color="limegreen")
        ax3.set_xlim(0, (t_lim-1)+0.005*(t_lim-1))
        ax3.set_ylim(-powerlim*0.1, powerlim)
        ax1.annotate(customtext,xy=(0.02, 0.92), xycoords='axes fraction', fontsize=13)
        ax1.annotate(f'Average volumetric pumping power: {avg_power_Wm3:.2f} W/m$^3$',xy=(0.02, 0.85), xycoords='axes fraction', fontsize=13)
    
    peak_mask_vel = df_vel["peak"] == 1
    time_peak_vel = df_vel.loc[peak_mask_vel, "time(s)"]
    vel_peak = df_vel.loc[peak_mask_vel, "flow rate mlmin smh"] 
    crest_mask_vel = df_vel["crest"] == 1
    time_crest_vel = df_vel.loc[crest_mask_vel, "time(s)"]
    vel_crest = df_vel.loc[crest_mask_vel, "flow rate mlmin smh"]
    vel_peak2peak = np.average(vel_peak)-np.average(vel_crest)
    # print("Flow amp: ",vel_peak2peak)
    
    if plot_vel_peak == True and plotgraph == True:        
        ax1.scatter(time_peak_vel, vel_peak, color="red", marker="o", s=30, label="Peaks")
        ax1.scatter(time_crest_vel, vel_crest, color="red", marker="o", s=30, label="Peaks")
        
    peak_mask_pressure = df_P["peak"] == 1
    time_peak_pressure = df_P.loc[peak_mask_pressure, "time(s)"]
    pressure_peak = df_P.loc[peak_mask_pressure, "smooth dP(Pa)"]
    crest_mask_pressure = df_P["crest"] == 1
    time_crest_pressure = df_P.loc[crest_mask_pressure, "time(s)"]
    pressure_crest = df_P.loc[crest_mask_pressure, "smooth dP(Pa)"]
    press_peak2peak = np.average(pressure_peak)-np.average(pressure_crest)
    # print("Pressure amp: ",press_peak2peak) 
    
    if plot_pressure_peak == True:        
        ax2.scatter(time_peak_pressure, pressure_peak/1000, color="blue", marker="s", s=30, label="Peaks")
        ax2.scatter(time_crest_pressure, pressure_crest/1000, color="blue", marker="s", s=30, label="Peaks")

    
    return vel_peak2peak,press_peak2peak

def polyfit_2nd(x, a, b):
    return a*x + b*x**2

def polyfit_3rd(x, a, b, c):
    return a*x + b*x**2 + c*x**3

def rpm2flowrate(rpm,calibrate_date):
    # flow_corr = pd.read_csv("/home/kevin/Desktop/UAntwerp/PhD_thesis/copper limiting current/git_folder/pressure_drop_experiment/peristaltic_calibration.csv",header=None) #flow rate in mL/min
    # flow_corr = pd.read_csv("/home/kevin/Desktop/UAntwerp/PhD_thesis/copper limiting current/git_folder/pressure_drop_experiment/peristaltic_calibration_01122025.csv",header=None) #flow rate in mL/min
    # flow_rate = (polyfit_3rd(rpm, flow_corr.iat[0,0], flow_corr.iat[1,0], flow_corr.iat[2,0])) #m3/s
    
    if calibrate_date == "20251202":
        # New calibration Dec 2, 2025
        measured_rpm = np.array([20,40,60,80,100,120,140])
        measured_flow_rate = np.array([6.902,14.341,23.341,31.986,34.270,37.768,45.949])/1000000/60 #m3/s
        
    if calibrate_date == "20260119":
        measured_rpm = np.array([0,20,40,60,80,100,120,140])
        measured_flow_rate = np.array([0,6.453,13.348,20.496,27.481,33.500,38.585,43.360])
    pchip = PchipInterpolator(measured_rpm, measured_flow_rate)

    return pchip(rpm)