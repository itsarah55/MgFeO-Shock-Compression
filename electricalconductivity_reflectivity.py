import numpy as np
from scipy import signal
from scipy.constants import pi, G, hbar, m_e, m_u, k, e, N_A
import scipy.optimize as opt
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


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

def usupcubic(up):
    #mgo cubic relationship based on Root2015
    usup_cubic = 6.558 + (1.424*up) + ((-.03011) * up**2.0) + (.00123 * up**3.0)
    return usup_cubic

def fit_funx(x, a, b):
    #temperature fit function
    return a + b*x 
    #return a* np.exp(b*x)

def rh(us, up):
    #rankine-hugoniot equations
    us = us *1.e3
    up = up*1.e3
    rho0 = 3584.0
    rho = (rho0 * us)/ (us - up)
    p = (rho0*us*up)/1e9 #GPa
    return p, rho

def ec_model(up, temp, Z):
    #electrical conductivity model based on Z and temperature
    rho0 = 3584.0
    n0 = 1.74 #initial refractive index MgO
    molmass = (40.3044)*1e-3 #no dissociation, kg/mol, MgO
    #molmass = (20.1522)*1e-3 #using full dissociation molar mass from Danae's powerpoint
    us = usupcubic(up)
    rho = (rho0 * us)/ (us - up)
    omega = 3e8/(532e-9) #laser frequency
    e_0 = 8.854e-12  #kg*m/s^2*A^2
    e_charge=1.602E-19 #electron charge, C, A/s
    mstar = m_e/2.0 #from McWilliams 2012 for effective mass of an electron?
    
    nb = n0 + (-0.064 * (rho-rho0)*1e-3)
    ni = (rho*N_A)/molmass
    ne = Z * ni
    
    v_f = (hbar/mstar) * (3.0*pi**2.0 *ne)**(1.0/3.0)
    v_th = np.sqrt((2.0*k*temp)/m_e)
    vel = [max(v_f), max(v_th)]
    
    omega_p = np.sqrt((ne* e_charge**2.0)/(e_0*mstar))
    tau = (2.0/max(vel))*(3/(4.0*pi**2.0 *ni))**(1.0/3.0)
    sigma0 = (ne*e_charge**2.0 * tau)/mstar #DC conductivity

    n_real = np.sqrt(nb**2.0 - (((omega_p/omega)**2.0)/(1.0 +complex(0,1)/(omega*tau)))).real
    n_imag = np.sqrt(nb**2.0 - (((omega_p/omega)**2.0)/(1.0 +complex(0,1)/(omega*tau)))).imag
    r_nb = ((n_real - n0)**2.0 + n_imag**2.0)/((n_real +n0)**2.0 + n_imag**2.0)
    
    sigmaw = ((complex(0,1)*ne*e_charge**2.0)/(mstar * omega)).imag #from Danae's powerpoint

    return sigma0, sigmaw, r_nb

def ec_ref(t, us_t, r, us_r):
    #electrical conductivity based on reflectivity and temperature data
    up_r = (us_r - 7.049)/1.240
    popt, pcov_ = opt.curve_fit(fit_funx, us_t, t)
    temp = fit_funx(us_r, *popt)

    rho0 = 3584.0
    rho = (rho0 * us_r)/ (us_r - up_r)

    n0 = 1.74
    molmass = (40.3044)*1e-3 #no dissociation, kg/mol, MgO
    omega = 3e8/(532e-9) #laser frequency
    e_0 = 8.854e-12  #kg*m/s^2*A^2
    e_charge=1.602E-19 #electron charge, C, A/s
    mstar = m_e/2.0 

    n = (2 * np.sqrt((r * n0**2.0)/((r-1)**2.))) - ((n0 * (r+1))/(r-1))
    nb = n0 - (.35 * (rho-rho0)*1e-3)
    ni = (rho*N_A)/molmass

    v_th = np.sqrt((2.0*k*t)/m_e)
    tau = (2.0/max(v_th))*(3/(4.0*pi**2.0 *ni))**(1.0/3.0)
    c_imag = omega**2.0 * (1 + complex(0,1)/(omega*tau)).imag
    c_real = omega**2.0 * (1 + complex(0,1)/(omega*tau)).real

    omega_p = np.sqrt(c_real+c_imag) * np.sqrt(n**2. - nb**2.)
    ne = (omega_p**2.0*e_0*mstar)/(e_charge**2.)

    e_f = (1.054e-34/(2.0*m_e)) * ((3*np.pi**2. *ne)**(2./3.))
    v_f = np.sqrt((2.0*e_f)/m_e)
    v_th = np.sqrt(2.0*k*t/m_e)
    vel = [max(v_f), max(v_th)]
    tau = (2.0/max(v_th))*(3/(4.0*pi**2.0 *ni))**(1.0/3.0)
    c_imag = omega**2.0 * (1 + complex(0,1)/(omega*tau)).imag
    c_real = omega**2.0 * (1 + complex(0,1)/(omega*tau)).real
    omega_p = np.sqrt(c_real+c_imag) * np.sqrt(n**2. - nb**2.)

    sigma0 = (ne*e_charge**2. * tau)/mstar
    return sigma0, temp

def hill_r(us, x, a, b, n):
    #hill function
    return a + (((b-a)* (us**n))/(us**n + x**n))

def noise_filter(ydata):
    #filters data
    b,a = signal.butter(3, 0.05)
    zi = signal.lfilter_zi(b,a)
    z, _ = signal.lfilter(b, a, ydata, zi=zi*ydata[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    y = signal.filtfilt(b,a,ydata)
    return y

def reynolds(l, sigma, v):
    #reynolds number
    mu0 = 4*pi/1e7
    nu = 1/(mu0*sigma)
    return  (v*l)/nu

def bmo_l(sigma, v):
    #basal magma ocean thickness
    rm = 45 
    mu0 = 4*pi/1e7
    nu = 1/(mu0*sigma)
    return rm*nu/v


"Temperature - Us relationship"
txt_38691 = np.loadtxt('38691/t_us.txt')
us_38691 = txt_38691[:,0]
t_38691 = txt_38691[:,1]

txt_38692 = np.loadtxt('38692/t_us.txt')
us_38692 = txt_38692[:,0]
t_38692 = txt_38692[:,1]

txt_38693 = np.loadtxt('38693/t_us.txt')
us_38693 = txt_38693[:,0]
t_38693 = txt_38693[:,1]

"Temperature - Us with erroneous points"
txt_38691og = np.loadtxt('38691/t_usog.txt')
us_38691og = txt_38691og[:,0]
t_38691og = txt_38691og[:,1]

"Reflectivity - Us relationship"
txt_38691r = np.loadtxt('38691/r_us.txt')
r_38691 = txt_38691r[:,0]/100
usr_38691 = txt_38691r[:,1]
up_38691 = (usr_38691 - 7.049)/1.240

txt_38692r = np.loadtxt('38692/r_us.txt')
r_38692 = txt_38692r[:,0]/100
usr_38692 = txt_38692r[:,1]
up_38692 = (usr_38692 - 7.049)/1.240

txt_38693r = np.loadtxt('38693/r_us.txt')
r_38693 = txt_38693r[:,0]/100
usr_38693 = txt_38693r[:,1]
up_38693 = (usr_38693 - 7.049)/1.240

"electrical conductivity measurement"

sigma0_38691, temp_38691 = ec_ref(t_38691, us_38691, r_38691, usr_38691)
sigma0_38692, temp_38692 = ec_ref(t_38692, us_38692, r_38692, usr_38692)
sigma0_38693, temp_38693 = ec_ref(t_38693, us_38693, r_38693, usr_38693)

"pressure-density relationship based on us in reflectivity"
p38691, rho38691 = rh(usr_38691, up_38691)
p38692, rho38692 = rh(usr_38692, up_38692)
p38693, rho38693 = rh(usr_38693, up_38693)

"temperature profiles"

popt_38691, pcov_38691 = opt.curve_fit(fit_funx, us_38691, t_38691)
temp_38691 = fit_funx(usr_38691, *popt_38691)
popt_38692, pcov_38692 = opt.curve_fit(fit_funx, us_38692, t_38692)
temp_38692 = fit_funx(usr_38692, *popt_38692)
popt_38693, pcov_38693 = opt.curve_fit(fit_funx, us_38693, t_38693)
temp_38693 = fit_funx(usr_38693, *popt_38693)


"reflectivity fits"

def hill_r(us, x, a, b, n):
    return (((b-a)* (us**n))/(us**n + x**n))

popr_38691, pcovr_38691 = opt.curve_fit(hill_r, usr_38691, r_38691*100)
ref_38691 = hill_r(usr_38691, *popr_38691)
popr_38692, pcovr_38692 = opt.curve_fit(hill_r, usr_38692, r_38692*100)
ref_38692 = hill_r(usr_38692, *popr_38692)
popr_38693, pcovr_38693 = opt.curve_fit(hill_r, usr_38693, r_38693*100)
ref_38693 = hill_r(usr_38693, *popr_38693)
np.sqrt(pcovr_38691[0,0])
np.sqrt(pcovr_38691[1,1])
np.sqrt(pcovr_38691[2,2])
np.sqrt(pcovr_38691[3,3])


#1sigma envelope fit

env1_up = ref_38691 + np.std(r_38691 -ref_38691)
env1_down = ref_38691 - np.std(r_38691 -ref_38691)
env2_up = ref_38692 + np.std(r_38692 -ref_38692)
env2_down = ref_38692 - np.std(r_38692 -ref_38692)
env3_up = ref_38693 + np.std(r_38693 -ref_38693)
env3_down = ref_38693 - np.std(r_38693 -ref_38693)

"Reynolds number calculations????"

l_38691s = bmo_l(sigma0_38691, .0006)/1e3
l_38691m = bmo_l(sigma0_38691, .001)/1e3
l_38691f = bmo_l(sigma0_38691, .0016)/1e3

l_38692s = bmo_l(sigma0_38692, .0006)/1e3
l_38692m = bmo_l(sigma0_38692, .001)/1e3
l_38692f = bmo_l(sigma0_38692, .0016)/1e3

l_38693s = bmo_l(sigma0_38693, .0006)/1e3
l_38693m = bmo_l(sigma0_38693, .001)/1e3
l_38693f = bmo_l(sigma0_38693, .0016)/1e3

fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(p38691, noise_filter(l_38691f), noise_filter(l_38691s), linestyle = '--' , color = '#63b3ce', alpha = 0.7, label ='MgO')
ax.fill_between(p38692, noise_filter(l_38692f), noise_filter(l_38692s), linestyle = '--' , color = '#86deb6', alpha = 0.7, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O')
ax.fill_between(p38693, noise_filter(l_38693f), noise_filter(l_38693s), linestyle = '--' , color = '#b2b5af', alpha = 0.7, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O')
ax.legend()
ax.set_xlabel ('Pressure (GPa)')
ax.set_ylabel('Minimum BMO Thickness (km)')
plt.show()

"""Plots"""

up = np.linspace(0, 19, len(temp_38692))
Z = .2 #achieving reflectivity for MgO McWilliams 2012
mgo_sigma0,mgo_sigmaw, mgo_r = ec_model(up, temp_38692, Z) #model for McWilliams 2012

print(max(mgo_sigma0)/1.e5)

"Electrical Conductivity v Pressure and Temperature "
fig = plt.figure('Electrical Conductivity v Pressure, Temperature')
ax1 = fig.add_subplot(211)
ax1.scatter(p38691, sigma0_38691, label = 'MgO $\sigma_{0}$', c = '#348aa7')
ax1.scatter(p38692, sigma0_38692, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O $\sigma_{0}$', c = '#5dd39e')
ax1.scatter(p38693, sigma0_38693, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O $\sigma_{0}$', c = '#b5b8b2')
ax1.plot(p38692, mgo_sigma0, color = 'white', label = 'Expected $\sigma$ MgO')
ax1.set_ylabel('Conductivity $\sigma$ (S $\cdot$ m$^{-1}$)')
ax1.set_xlabel('Pressure (GPa)')
ax1.set_yscale('log')
ax1.legend()
ax2 = fig.add_subplot(212)
ax2.scatter(temp_38691, sigma0_38691, label = 'MgO $\sigma_{0}$', c = '#348aa7')
ax2.scatter(temp_38692, sigma0_38692, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O $\sigma_{0}$', c = '#5dd39e')
ax2.scatter(temp_38693, sigma0_38693, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O $\sigma_{0}$', c = '#b5b8b2')
ax2.plot(temp_38692, mgo_sigma0, color = 'white', label = 'Expected $\sigma$ MgO')
ax2.set_ylabel('Conductivity $\sigma$ (S $\cdot$ m$^{-1}$)')
ax2.set_xlabel('Temperature (K)')
ax2.set_yscale('log')
ax2.legend()
plt.show()

"Temperature v U_S fits"

fig = plt.figure(2)

ax2 = fig.add_subplot(311)
ax2.scatter(us_38693, t_38693, c= '#1D267D', label = 'Raw Data')
ax2.plot(us_38693, fit_funx(us_38693, *popt_38693), c = '#D4ADFC', label = 'fit')
ax2.legend()
ax2.set_title('(Mg$_{0.95}$,Fe$_{0.05}$)O')
ax2.set_xlabel('Us (km/s)')
ax2.set_ylabel('Temperature (K)')
ax3 = fig.add_subplot(312)
#ax3.scatter(us_38692og, t_38692og, c = '#00FFCA', label = 'Removed from Fit')
ax3.scatter(us_38692, t_38692, c= '#1D267D', label = 'Raw Data')
ax3.plot(us_38692, fit_funx(us_38692, *popt_38692), c = '#D4ADFC', label = 'fit')
ax3.legend()
ax3.set_xlabel('Us (km/s)')
ax3.set_ylabel('Temperature (K)')
ax3.set_title('(Mg$_{0.98}$,Fe$_{0.02}$)O')
ax1 = fig.add_subplot(313)
ax1.scatter(us_38691og, t_38691og, c = '#00FFCA', label = 'Removed from Fit')
ax1.scatter(us_38691, t_38691, c= '#1D267D', label = 'Raw Data')
ax1.plot(us_38691, fit_funx(us_38691, *popt_38691), c = '#D4ADFC', label = 'fit')
ax1.legend()
ax1.set_xlabel('Us (km/s)')
ax1.set_ylabel('Temperature (K)')
ax1.set_title('MgO')
plt.show()

"Pressure, Us, Reflectivity"

fig = plt.figure('Pressure-U_S')
ax1 = fig.add_subplot(111)
ax1.scatter(usr_38691, temp_38691)
ax1.scatter(usr_38692, temp_38692)
ax1.scatter(usr_38693, temp_38693)
ax2 = ax1.twinx()
ax2.plot(usr_38691, p38691, label = 'MgO')
ax2.plot(usr_38692, p38692, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O')
ax2.plot(usr_38693, p38693, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O')
ax2.set_xlabel('U$_S$ (km/s)')
ax2.set_ylabel('Pressure (GPa)')
ax2.legend()
plt.show()

fig = plt.figure('Pressure-U_S-Reflectivity')
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Reflectivity ($\%$)')
ax1.scatter(p38691, r_38691, color = 'black',label = 'MgO')
ax1.scatter(p38692, r_38692, color = 'black',label = '(Mg$_{0.98}$,Fe$_{0.02}$)O')
ax1.scatter(p38693, r_38693, color = 'black',label = '(Mg$_{0.95}$,Fe$_{0.05}$)O')
ax1.set_xlabel('Pressure (GPa)')
ax2 = ax1.twiny()
ax2.scatter(usr_38691, r_38691*100,c = '#348aa7', label = 'MgO')
ax2.scatter(usr_38692, r_38692*100,c = '#5dd39e', label = '(Mg$_{0.98}$,Fe$_{0.02}$)O')
ax2.scatter(usr_38693, r_38693*100,c = '#b5b8b2', label = '(Mg$_{0.95}$,Fe$_{0.05}$)O')
ax2.plot(usr_38691, ref_38691,c = '#348aa7', linewidth = '2')
ax2.plot(usr_38692, ref_38692,c = '#5dd39e', linewidth = '2')
ax2.plot(usr_38693, ref_38693,c = '#b5b8b2', linewidth = '2')
ax2.set_xlabel('U$_S$ Shock Velocity (km/s)')
ax2.fill_between(usr_38691, env1_up, env1_down, color = '#63b3ce', alpha = 0.4)
ax2.fill_between(usr_38692, env2_up, env2_down, color = '#86deb6', alpha = 0.4)
ax2.fill_between(usr_38693, env3_up, env3_down,color = '#b2b5af', alpha = 0.4)
ax2.legend()
plt.show()
