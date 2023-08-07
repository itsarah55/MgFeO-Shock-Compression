import numpy as np
import math
from scipy.constants import pi, G, hbar, m_e, m_u, k, e, N_A
from scipy import signal
import scipy.optimize as opt
import matplotlib.pyplot as plt

"""Plots reflectivity from other MgO reflectivity"""

def hill_r(us, x, b, a, n):
    return ((b-a)* (us**n))/(us **n + x **n)
    #return ((b-a)* (us**n))/(us **n + x **n) + a

def noise_filter(ydata):
    b,a = signal.butter(3, 0.05)
    zi = signal.lfilter_zi(b,a)
    z, _ = signal.lfilter(b, a, ydata, zi=zi*ydata[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    y = signal.filtfilt(b,a,ydata)
    return y

txt_38691 = np.loadtxt('38691/r_us.txt')
r_38691 = txt_38691[:,0]
us_38691 = txt_38691[:,1]
popt_38691, pcov_38691 = opt.curve_fit(hill_r, us_38691, r_38691)
ref_38691 = hill_r(us_38691, *popt_38691)
env1_up = ref_38691 + np.std(r_38691 -ref_38691)
env1_down = ref_38691 - np.std(r_38691 -ref_38691)

txt_38692 = np.loadtxt('38692/r_us.txt')
r_38692 = txt_38692[:,0]
us_38692 = txt_38692[:,1]

popt_38692, pcov_38692 = opt.curve_fit(hill_r, us_38692, r_38692)
ref_38692 = hill_r(us_38692, *popt_38692)
env2_up = ref_38692 + np.std(r_38692 -ref_38692)
env2_down = ref_38692 - np.std(r_38692 -ref_38692)

txt_38693 = np.loadtxt('38693/r_us.txt')
r_38693 = txt_38693[:,0]
us_38693 = txt_38693[:,1]

popt_38693, pcov_38693 = opt.curve_fit(hill_r, us_38693, r_38693)
ref_38693 = hill_r(us_38693, *popt_38693)
env3_up = ref_38693 + np.std(r_38693 -ref_38693)
env3_down = ref_38693 - np.std(r_38693 -ref_38693)

h1 = np.loadtxt('mgo/h1.txt')
hus_1 = h1[:,0]
hr_1 = h1[:,1]

h2 = np.loadtxt('mgo/h2.txt')
hus_2 = h2[:,0]
hr_2 = h2[:,1]

h3 = np.loadtxt('mgo/h3.txt')
hus_3 = h3[:,0]
hr_3 = h3[:,1]

h4 = np.loadtxt('mgo/h4.txt')
hus_4 = h4[:,0]
hr_4 = h4[:,1]

h5 = np.loadtxt('mgo/h5.txt')
hus_5 = h5[:,0]
hr_5 = h5[:,1]

h6 = np.loadtxt('mgo/h6.txt')
hus_6 = h6[:,0]
hr_6 = h6[:,1]

m1 = np.loadtxt('mgo/m1.txt')
mus_1 = m1[:,0]
mr_1 = m1[:,1]

b1 = np.loadtxt('mgo/b1.txt')
bus_1 = b1[:,0]
br_1 = b1[:,1]

mc1 = np.loadtxt('mgo/mc1.txt')
mcus_1 = mc1[:,0]
mcr_1 = mc1[:,1]

mc2 = np.loadtxt('mgo/mc2.txt')
mcus_2 = mc2[:,0]
mcr_2 = mc2[:,1]

mc3 = np.loadtxt('mgo/mc3.txt')
mcus_3 = mc3[:,0]
mcr_3 = mc3[:,1]

mc4 = np.loadtxt('mgo/mc4.txt')
mcus_4 = mc4[:,0]
mcr_4 = mc4[:,1]

mc5 = np.loadtxt('mgo/mc5.txt')
mcus_5 = mc5[:,0]
mcr_5 = mc5[:,1]

mc6 = np.loadtxt('mgo/mc6.txt')
mcus_6 = mc6[:,0]
mcr_6 = mc6[:,1]

mc7 = np.loadtxt('mgo/mc7.txt')
mcus_7 = mc7[:,0]
mcr_7 = mc7[:,1]

mc8 = np.loadtxt('mgo/mc8.txt')
mcus_8 = mc8[:,0]
mcr_8 = mc8[:,1]

mc9 = np.loadtxt('mgo/mc9.txt')
mcus_9 = mc9[:,0]
mcr_9 = mc9[:,1]

filtr_38691 = noise_filter(r_38691)
filtr_38692 = noise_filter(r_38692)
filtr_38693 = noise_filter(r_38693)

plt.plot(mus_1, mr_1, c = '#e56b6f', label = 'McWilliams 2012')
plt.plot(bus_1, br_1, c = '#b56576', label = 'Bolis 2016')
plt.plot(mcus_1, mcr_1, c = '#6d597a', label = 'McCoy 2019')
plt.plot(mcus_2, mcr_2, c = '#6d597a')
plt.plot(mcus_3, mcr_3, c = '#6d597a')
plt.plot(mcus_4, mcr_4, c = '#6d597a')
plt.plot(mcus_5, mcr_5, c = '#6d597a')
plt.plot(mcus_6, mcr_6, c = '#6d597a')
plt.plot(mcus_7, mcr_7, c = '#6d597a')
plt.plot(mcus_8, mcr_8, c = '#6d597a')
plt.plot(mcus_9, mcr_9, c = '#6d597a')
plt.plot(hus_1, hr_1, c = '#eaac8b')
plt.plot(hus_2, hr_2, c = '#eaac8b')
plt.plot(hus_3, hr_3, c = '#eaac8b')
plt.plot(hus_4, hr_4, c = '#eaac8b')
plt.plot(hus_5, hr_5, c = '#eaac8b')
plt.plot(hus_6, hr_6, c = '#eaac8b', label = 'Hansen 2021')
plt.plot(us_38693, filtr_38693, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O', c ='red')
plt.plot(us_38692, filtr_38692, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O', c = 'black')
plt.plot(us_38691, filtr_38691, label = 'MgO', c ='blue')
plt.ylabel('Reflectivity (%)')
plt.xlabel('Us (km/s)')
plt.legend()
plt.show()

