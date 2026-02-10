import sys
sys.path.insert(0, "/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/git_folder")  # add Folder_2 path to search list

import re
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from pulse_flow_power_calculation_func import calpower,plotdata,pressure2df,track2df
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
vol_len = 1/7.7# mL per cm
marker_dis = 8.0 #cm 
plunger_dia = 0.6 #cm
vel_pd = 50
press_pd = 20
plot_peak = False

file_BCC167_1Hz = ["BCC1.67-0rpm-0.231cmcrankdia-1Hz",
                   "BCC1.67-0rpm-0.526cmcrankdia-1Hz",
                   "BCC1.67-0rpm-1.070cmcrankdia-1Hz",
                   "BCC1.67-0rpm-1.442cmcrankdia-1Hz",
                   "BCC1.67-0rpm-2.235cmcrankdia-1Hz",
                   "BCC1.67-0rpm-2.962cmcrankdia-1Hz"]

file_BCC167_2Hz = ["BCC1.67-0rpm-0.231cmcrankdia-2Hz",
                   "BCC1.67-0rpm-0.526cmcrankdia-2Hz",
                   "BCC1.67-0rpm-1.098cmcrankdia-2Hz",
                   "BCC1.67-0rpm-1.319cmcrankdia-2Hz",
                   "BCC1.67-0rpm-1.874cmcrankdia-2Hz"]
             
file_BCC167_4Hz = ["BCC1.67-0rpm-0.093cmcrankdia-4Hz",
                   "BCC1.67-0rpm-0.156cmcrankdia-4Hz",
                   "BCC1.67-0rpm-0.198cmcrankdia-4Hz",
                   "BCC1.67-0rpm-0.231cmcrankdia-4Hz",
                   "BCC1.67-0rpm-0.330cmcrankdia-4Hz"]

# file_BCC1674Hz = [ "BCC1.67-0rpm-0.330cmcrankdia-4Hz",
#              "BCC1.67-0rpm-0.156cmcrankdia-4Hz"]

# file_2_5cm1Hz = ["BCC1.67-2.5cm-0rpm-2.924cmcrankdia-1Hz_2",
#             "BCC1.67-2.5cm-0rpm-1.033cmcrankdia-1Hz"]

# file_2_5cm4Hz = ["BCC1.67-2.5cm-0rpm-0.327cmcrankdia-4Hz",
#             "BCC1.67-2.5cm-0rpm-0.162cmcrankdia-4Hz",
#             "BCC1.67-2.5cm-0rpm-0.145cmcrankdia-4Hz_2"]

file_CF_1Hz = ["CF30-0rpm-0.443cmcrankdia-1Hz",
              "CF30-0rpm-0.864cmcrankdia-1Hz",
              "CF30-0rpm-1.327cmcrankdia-1Hz",
              "CF30-0rpm-1.773cmcrankdia-1Hz",
              "CF30-0rpm-2.215cmcrankdia-1Hz"]

file_CF_2Hz = ["CF30-0rpm-0.233cmcrankdia-2Hz",
              "CF30-0rpm-0.460cmcrankdia-2Hz",
              "CF30-0rpm-0.648cmcrankdia-2Hz",
              "CF30-0rpm-0.880cmcrankdia-2Hz",
              "CF30-0rpm-1.115cmcrankdia-2Hz"]

file_CF_4Hz = ["CF30-0rpm-0.051cmcrankdia-4Hz",
              "CF30-0rpm-0.093cmcrankdia-4Hz",
              "CF30-0rpm-0.138cmcrankdia-4Hz",
              "CF30-0rpm-0.191cmcrankdia-4Hz",
              "CF30-0rpm-0.226cmcrankdia-4Hz"]

def plot_flow_profile(fpath,fnlist):
    # fig, ax1 = plt.subplots(figsize=(8, 4.5))
    power_arr = np.array([])
    amp_arr = np.array([])
    stroke_arr = np.array([])
    for fname in fnlist:
        filefullpath = fpath+fname
        match_crankdia = re.search(r'([\d.]+)cmcrankdia', fname)
        strokeln = float(match_crankdia.group(1))
        match_Hz = re.search(r'([\d.]+)Hz', fname)
        frequency = float(match_Hz.group(1))

        if frequency == 1:
            plot_time_end = 4.0005
        if frequency == 2:
            plot_time_end = 2.0005
        if frequency == 4:
            plot_time_end = 1.0005

        file = filefullpath+"-track.csv"
        file_P = filefullpath+"-pressure.csv"
        text = "Pulse frequency: "+str(frequency)+" Hz, stroke volume: "+str((np.round((plunger_dia/2)**2*np.pi*strokeln,3)))+" mL"
        df_vel,df_P,df_power,avg_power = calpower(file, file_P, 0.08, 0.08, 10, 20, 62, 62+plot_time_end,0,vol_len,marker_dis,electrode_vol,50,15)
        flow_amp,press_amp = plotdata(df_vel,df_P,df_power,avg_power,160,1,1,False,False,False,1,text) 
        # plt.show()
        
        power_arr = np.append(power_arr,avg_power)
        amp_arr = np.append(amp_arr,flow_amp)
        stroke_arr = np.append(stroke_arr,strokeln)
    stroke_fit, power_fit, popt_stroke = fit_and_curve(poly4th, stroke_arr, power_arr,x_min=0.0,bounds=(0, np.inf))
    amp_fit, _, popt_amp = fit_and_curve(poly4th, amp_arr, power_arr,x_min=0.0,bounds=(0, np.inf))
    return stroke_arr, amp_arr, power_arr, stroke_fit, power_fit, amp_fit
    


file_path ="/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/git_folder/pressure_drop_experiment/pulsation_velocity_pressure_data/20251121/"        
stroke_arr_BCC167_1Hz, amp_arr_BCC167_1Hz, power_arr_BCC167_1Hz, stroke_fit_BCC167_1Hz, power_fit_BCC167_1Hz, amp_fit_BCC167_1Hz = plot_flow_profile(file_path,file_BCC167_1Hz)
plt.plot(stroke_arr_BCC167_1Hz*10, power_arr_BCC167_1Hz, "s",color=color[0])
plt.plot(stroke_fit_BCC167_1Hz*10, power_fit_BCC167_1Hz, "-",label="1 Hz",color=color[0])

file_path ="/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/git_folder/pressure_drop_experiment/pulsation_velocity_pressure_data/20251121/"        
stroke_arr_BCC167_2Hz, amp_arr_BCC167_2Hz, power_arr_BCC167_2Hz, stroke_fit_BCC167_2Hz, power_fit_BCC167_2Hz, amp_fit_BCC167_2Hz = plot_flow_profile(file_path,file_BCC167_2Hz)
plt.plot(stroke_arr_BCC167_2Hz*10, power_arr_BCC167_2Hz, "^",color=color[1])
plt.plot(stroke_fit_BCC167_2Hz*10, power_fit_BCC167_2Hz, "-",label="2 Hz",color=color[1])

file_path ="/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/git_folder/pressure_drop_experiment/pulsation_velocity_pressure_data/20251121/"        
stroke_arr_BCC167_4Hz, amp_arr_BCC167_4Hz, power_arr_BCC167_4Hz, stroke_fit_BCC167_4Hz, power_fit_BCC167_4Hz, amp_fit_BCC167_4Hz = plot_flow_profile(file_path,file_BCC167_4Hz)
plt.plot(stroke_arr_BCC167_4Hz*10, power_arr_BCC167_4Hz, "o",color=color[2])
plt.plot(stroke_fit_BCC167_4Hz*10, power_fit_BCC167_4Hz, "-",label="4 Hz",color=color[2])

plt.title("BCC 1.67 mm")
plt.xlabel("Stroke length (mm)")
plt.ylabel("Pumping power (W/m${^3}$)")
plt.xlim(0,30)
plt.ylim(0,140)
plt.grid()
plt.legend()
plt.show()

file_path ="/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/git_folder/pressure_drop_experiment/pulsation_velocity_pressure_data/20260203/"        
stroke_arr_CF_1Hz, amp_arr_CF_1Hz, power_arr_CF_1Hz, stroke_fit_CF_1Hz, power_fit_CF_1Hz, amp_fit_CF_1Hz = plot_flow_profile(file_path,file_CF_1Hz)
plt.plot(stroke_arr_CF_1Hz*10, power_arr_CF_1Hz, "s",color=color[0])
plt.plot(stroke_fit_CF_1Hz*10, power_fit_CF_1Hz, "-",label="1 Hz",color=color[0])

file_path ="/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/git_folder/pressure_drop_experiment/pulsation_velocity_pressure_data/20260203/"        
stroke_arr_CF_2Hz, amp_arr_CF_2Hz, power_arr_CF_2Hz, stroke_fit_CF_2Hz, power_fit_CF_2Hz, amp_fit_CF_2Hz = plot_flow_profile(file_path,file_CF_2Hz)
plt.plot(stroke_arr_CF_2Hz*10, power_arr_CF_2Hz, "^",color=color[1])
plt.plot(stroke_fit_CF_2Hz*10, power_fit_CF_2Hz, "-",label="2 Hz",color=color[1])

file_path ="/home/kevin/Dropbox/UAntwerp/PhD_thesis/copper limiting current/git_folder/pressure_drop_experiment/pulsation_velocity_pressure_data/20260202/"        
stroke_arr_CF_4Hz, amp_arr_CF_4Hz, power_arr_CF_4Hz, stroke_fit_CF_4Hz, power_fit_CF_4Hz, amp_fit_CF_4Hz = plot_flow_profile(file_path,file_CF_4Hz)
plt.plot(stroke_arr_CF_4Hz*10, power_arr_CF_4Hz, "o",color=color[2])
plt.plot(stroke_fit_CF_4Hz*10, power_fit_CF_4Hz, "-",label="4 Hz",color=color[2])

plt.title("Carbon felt 30% compression")
plt.xlabel("Stroke length (mm)")
plt.ylabel("Pumping power (W/m${^3}$)")
plt.xlim(0,30)
plt.ylim(0,400)
plt.grid()
plt.legend()
plt.show()


plt.plot(amp_arr_CF_1Hz*10, power_arr_CF_1Hz, "s",color=color[0])
plt.plot(amp_fit_CF_1Hz*10, power_fit_CF_1Hz, "-",label="1 Hz",color=color[0])

plt.plot(amp_arr_CF_2Hz*10, power_arr_CF_2Hz, "^",color=color[1])
plt.plot(amp_fit_CF_2Hz*10, power_fit_CF_2Hz, "-",label="2 Hz",color=color[1])

plt.plot(amp_arr_CF_4Hz*10, power_arr_CF_4Hz, "o",color=color[2])
plt.plot(amp_fit_CF_4Hz*10, power_fit_CF_4Hz, "-",label="4 Hz",color=color[2])

plt.xlabel("Flow velocity amplitde (mm)")
# plt.xlabel("Flow rate amplitude (mL/min)")
plt.ylabel("Pumping power (W/m${^3}$)")
# plt.xlim(0,4)
# plt.ylim(0,80)
# plt.grid()
plt.legend()
plt.show()