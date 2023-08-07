import numpy as np
import math
from scipy.constants import pi, G, hbar, m_e, m_u, k, e, N_A
import matplotlib.pyplot as plt
from matplotlib import rc


plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"}) 

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

up_mgo = np.asarray([14.9933])
up_mg98 = np.asarray([14.25]),
up_mg95=np.asarray([15.32])
us_mgo = np.asarray([25.11])
us_mg98=np.asarray([24.51]) 
us_mg95=np.asarray([26.25])
rho_mgo = np.asarray([8.89])
rho_mg98 =np.asarray([8.56])
rho_mg95 =np.asarray([8.614])
p_mgo = np.asarray([1349.31])
p_mg98 =np.asarray([1251.93])
p_mg95 =np.asarray([1442.1])

gamma_mgfeo = np.asarray([0.577, 0.576, 0.578])

hug_rho = np.asarray([0.356986, 0.892464, 1.784928, 3.569856, 7.139711, 8.924641, 10.709567, 14.279423, 16.064351, 17.849278, 21.419134])
hug_p = np.asarray([1172.1])

us_mccoy = np.asarray([19.7, 22.1, 22.9, 22.9, 24.4, 24.4, 24.4, 25.8, 26.8, 26.6, 28.8, 30.4, 32.0])
up_mccoy = np.asarray([10.0, 12.3, 12.7, 12.8, 13.9, 14.0, 14.1, 15.0, 15.6, 15.7, 17.7, 19.0, 20.1])
rho_mccoy = np.asarray([7.30, 8.08, 8.03, 8.10, 8.32, 8.40, 8.46, 8.57, 8.57, 8.80, 9.28, 9.54, 9.60])
p_mccoy = np.asarray([711, 976, 1042, 1049, 1216, 1222, 1231, 1384, 1492, 1498, 1829, 2072, 2303])
up_root = np.asarray([8.13, 8.32, 8.48, 8.7, 8.89, 9.07, 9.26, 5.44, 6.18, 6.4, 6.54, 6.75, 6.98, 7.31, 7.49 ,7.8, 8.14, 8.68, 8.73, 8.85, 9.57, 10.37, 10.54, 11.04, 11.13, 11.29, 11.47, 11.78, 12.15, 12.7, 12.51, 12.63, 13.01, 13.47, 13.65])
us_root = np.asarray([17.02, 17.29, 17.38, 17.82, 18.2, 18.38, 18.56, 13.86, 14.92, 14.99, 15.34, 15.41, 15.42, 16.05, 16.06, 16.77, 17.07, 17.48, 17.88, 18.29, 18.89, 19.72, 20.12, 20.87, 20.99, 21.11, 21.36, 21.75, 22.14, 22.5, 22.6, 22.81, 23.16, 23.93, 24.07])
rho_root = np.asarray([6.86, 6.91, 7., 7., 7.01, 7.07, 7.15, 5.90, 6.12, 6.25, 6.24, 6.37, 6.55, 6.58, 6.72, 6.7, 6.85, 7.11, 7., 6.94, 7.26, 7.56, 7.53, 7.61, 7.63, 7.7, 7.74, 7.82, 7.94, 7.96, 8.03, 8.03, 8.17, 8.20, 8.28])
p_root = np.asarray([496.2, 515.3, 528.1, 555.4, 580.1, 597.2, 616.0, 270.22, 330.6, 343.8, 359.3, 372.6, 385.9, 420.6, 431.2, 468.8, 497.9, 543.4, 559.4, 579.8, 648.0, 732.6, 760., 825.7, 837.2, 854.1, 878.3, 918.5, 964.2, 997.5, 1013.4, 1032.3, 1079.6, 1155.3, 1177.6])
us_mcwilliams = np.asarray([18.82, 18.88])
up_mcwilliams = np.asarray([9.48, 8.96])
rho_mcwilliams = np.asarray([7.22, 6.82])
p_mcwilliams = np.asarray([639.0, 606.0])
us_hansen = np.asarray([23.18, 23.72, 25.39, 28.47, 30.01])
up_hansen = np.asarray([5.12, 4.98, 5.21, 3.76, 4.0])
rho_hansen = np.asarray ([8.65, 9.13, 9.53, 9.94, 11.84])
p_hansen = np.asarray([1170, 1218, 1449, 1950, 2109])


def usup (up):
    us_up = 7.049 + 1.240*up 
    return us_up

def usupcubic(up):
    usup_cubic = 6.558 + (1.424*up) + ((-.03011) * up**2.0) + (.00123 * up**3.0)
    return usup_cubic

#us_up
up = np.linspace(min(up_root), max(up_mccoy), (len(up_root) + len(up_mccoy)))
plt.plot(up, usup(up), '--', c = '#0c72e6', label = 'linear fit')
plt.plot(up, usupcubic(up),'--', c = '#5503be', label = 'cubic fit')
plt.scatter(up_mccoy, us_mccoy, c = '#8b1e3f', label = 'McCoy 2019')
plt.scatter(up_root, us_root, c = '#db4c40', label = 'Root 2015')
plt.scatter(up_mcwilliams, us_mcwilliams, c='#ffc857', label = 'McWilliams 2012')
plt.plot(up_mgo, us_mgo, c = '#bce784', label = 'MgO', marker = '*', markersize = '15')
plt.plot(up_mg98, us_mg98, c = '#5dd39e', label = '(Mg$_{0.98}$,Fe$_{0.02}$)O', marker = '*', markersize = '15')
plt.plot(up_mg95, us_mg95, c = '#348aa7', label = '(Mg$_{0.95}$,Fe$_{0.05}$)O', marker = '*', markersize = '15')
plt.scatter(up_hansen, us_hansen, c= '#eaac8b', label = 'Hansen 2021')
plt.xlabel('Up (km/s)')
plt.ylabel('Us (km/s)')
plt.legend()
plt.show()

#density_pressure
plt.plot(rho_mgo, p_mgo, label = 'MgO', marker = '*', c = '#bce784', markersize = '15')
plt.plot(rho_mg98, p_mg98, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O', marker = '*', c = '#5dd39e', markersize = '15')
plt.plot(rho_mg95, p_mg95, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O', marker = '*', c = '#348aa7', markersize = '15')
plt.scatter(rho_mccoy, p_mccoy, label = 'McCoy 2019', c = '#8b1e3f')
plt.scatter(rho_root, p_root, label = 'Root 2015', c = '#db4c40')
plt.scatter(rho_mcwilliams, p_mcwilliams, label = 'McWilliams 2012', c = '#ffc857')
plt.scatter(rho_hansen, p_hansen, label = 'Hansen 2021', c = '#3c153b')
plt.xlabel('Density (g/cm$^3$)')
plt.ylabel('Pressure (GPa)')
plt.legend()
plt.show()
