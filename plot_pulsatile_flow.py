import sys
sys.path.insert(0, "/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/git_folder")  # add Folder_2 path to search list
 
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from functions_collection import calpower,plotdata,pressure2df,track2df
import matplotlib.path as mpath
 
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['font.family'] = 'sans'
mpl.rcParams['font.serif'] = 'Liberation San'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
 
 
# def polyfit_1st(x, a):
#     return a*x
 
# def polyfit_3rd(x, a, b, c):
#     return a*x + b*x**2 + c*x**3
 
# def plotfit_3rd(x,y):
#     popt, _ = curve_fit(polyfit_3rd, x, y,bounds=(0, np.inf))
#     x_fit = np.linspace(0,max(x),100)
#     y_fit = polyfit_3rd(x_fit,*popt)
#     return x_fit,y_fit,popt
 
# def polyfit_4th(x, a, b, c, d):
#     return a*x + b*x**2 + c*x**3 + d*x**4
 
# def plotfit_4th(x,y):
#     popt, _ = curve_fit(polyfit_4th, x, y,bounds=(0, np.inf))
#     x_fit = np.linspace(0,max(x),100)
#     y_fit = polyfit_4th(x_fit,*popt)
#     return x_fit,y_fit,popt
 
# def plotfit_1st(avg_vel_arr,avg_amp_arr):
#     popt, _ = curve_fit(polyfit_1st, avg_vel_arr, avg_amp_arr,bounds=(0, np.inf))
#     x = np.linspace(0,max(avg_vel_arr),100)
#     y = polyfit_1st(x,*popt)
#     return x,y,popt
 
def poly4th(x, a1, a2, a3, a4):
    return a1*x + a2*x**2 + a3*x**3 + a4*x**4
 
def convex_exp_origin(x, k, b):
    return k * (np.exp(b * x) - 1.0)
 
def fit_and_curve(func, x, y, x_min=0.0, n=200, bounds=(-np.inf, np.inf)):
    popt, _ = curve_fit(func, x, y, bounds=bounds, maxfev=200000)
    x_fit = np.linspace(x_min, np.max(x), n)
    y_fit = func(x_fit, *popt)
    return x_fit, y_fit, popt
 
color = ("tomato","skyblue","limegreen","orange","magenta")
 
################################################################################
 
electrode_vol = 5*1*0.5/1000000
vol_len = 1/7.7# mL per cm #To convert from cm --> ml to not have cm/min but ml/min --> glass tube
marker_dis = 8.0 #cm 
plunger_dia = 0.6 #cm
vel_pd = 50
press_pd = 20
plot_peak = False
 
from pathlib import Path
 
file_path = Path("/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/pulsation videos/20260210/")
 
file_BCC125_1Hz = ["BCC125-0rpm-0.703cmcrankdia-1Hz"]
 
 
from pathlib import Path
 
def plot_flow_profile(fpath, fnlist):
    fpath = Path(fpath)
 
    power_arr = np.array([])
    amp_arr = np.array([])
    stroke_arr = np.array([])
 
    for fname in fnlist:
        fname = Path(fname)  # ensure Path
        filefullpath = fpath / fname
 
        # Use Path methods, do NOT add strings with '+'
        file = filefullpath.with_name(filefullpath.name + "-track.csv")
        file_P = filefullpath.with_name(filefullpath.name + "-pressure.csv")
 
        # Convert filename to string for regex
        name = fname.name
        match_crankdia = re.search(r'([\d.]+)cmcrankdia', name)
        strokeln = float(match_crankdia.group(1))
 
        match_Hz = re.search(r'([\d.]+)Hz', name)
        frequency = float(match_Hz.group(1))
 
        if frequency == 1:
            plot_time_end = 4.0005
        elif frequency == 2:
            plot_time_end = 2.0005
        elif frequency == 4:
            plot_time_end = 1.0005
 
        text = f"Pulse frequency: {frequency} Hz, stroke volume: {np.round((plunger_dia/2)**2*np.pi*strokeln,3)} mL"

        # calpower and plotdata accept either string or Path
        df_vel, df_P, df_power, avg_power = calpower(file, file_P, 0.08, 0.08, 10, 20, 64, 64+plot_time_end, 0, vol_len, marker_dis, electrode_vol, 200, 150) # The last two numbers are for the minimum distance that needs to be to have adequate minima and maxima (to not have sudden relative maxima also) First number is for velocity and second one for P-drop change to less to see
        flow_amp, press_amp = plotdata(df_vel, df_P, df_power, avg_power, 160, 1, 1, True, True, True, 1, text)
        plt.show()
        power_arr = np.append(power_arr, avg_power)
        amp_arr = np.append(amp_arr, flow_amp)
        stroke_arr = np.append(stroke_arr, strokeln)
 
    stroke_fit, power_fit, popt_stroke = fit_and_curve(poly4th, stroke_arr, power_arr, x_min=0.0, bounds=(0, np.inf))
    amp_fit, _, popt_amp = fit_and_curve(poly4th, amp_arr, power_arr, x_min=0.0, bounds=(0, np.inf))
    return stroke_arr, amp_arr, power_arr, stroke_fit, power_fit, amp_fit

file_path = Path("/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/pulsation videos/20260210/")
stroke_arr_BCC125_1Hz, amp_arr_BCC125_1Hz, power_arr_BCC125_1Hz, stroke_fit_BCC125_1Hz, power_fit_BCC125_1Hz, amp_fit_BCC125_1Hz = plot_flow_profile(file_path,file_BCC125_1Hz)
plt.plot(stroke_arr_BCC125_1Hz*10, power_arr_BCC125_1Hz, "s",color=color[0]) #multiply by 10 to convert from cm to mm
plt.plot(stroke_fit_BCC125_1Hz*10, power_fit_BCC125_1Hz, "-",label="1 Hz",color=color[0])
 
 
plt.title("BCC 1.25 mm")
plt.xlabel("Stroke length (mm)")
plt.ylabel("Pumping power (W/m${^3}$)")
plt.xlim(0,30)
plt.ylim(0,140)
plt.grid()
plt.legend()
plt.show()