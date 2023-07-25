import numpy as np
from scipy import signal
from scipy.constants import pi, G, hbar, m_e, m_u, k, e, N_A
import scipy.optimize as opt
import matplotlib.pyplot as plt

def usupcubic(up):
    usup_cubic = 6.558 + (1.424*up) + ((-.03011) * up**2.0) + (.00123 * up**3.0)
    return usup_cubic

def fit_funx(x, a, b,n):
    return a + x**n 
    #return a* np.exp(b*x)

def rh(us, up):
    us = us *1.e3
    up = up*1.e3
    rho0 = 3584.0
    rho = (rho0 * us)/ (us - up)
    p = (rho0*us*up)/1e9 #GPa
    return p, rho

def ec_ref(t, us_t, r, us_r):
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

def ec_fit(x, sig00, ea):
    kb = 8.617333262e-5
    return (sig00 * np.exp(- ea/ (kb*x)))

def ec_predict(p, t):
    sigma0 = 1.99e5 #S/m
    del_e = 75940 #J/mol
    del_v = -6.1e-8 #m^3/mol
    r = 8.314 #J/mol*K  
    return sigma0 * np.exp(-(del_e + p*del_v)/(r*t))

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

def reynolds_mag(mu, L, sigma, v):
    return mu*v*L*sigma


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

#txt_38692og = np.loadtxt('38692/t_usog.txt')
#us_38692og = txt_38692og[:,0]
#t_38692og = txt_38692og[:,1]

#txt_38693og = np.loadtxt('38693/t_usog.txt')
#us_38693og = txt_38693og[:,0]
#t_38693og = txt_38693og[:,1]

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
popr_38691, pcovr_38691 = opt.curve_fit(hill_r, usr_38691, r_38691)
ref_38691 = hill_r(usr_38691, *popr_38691)
popr_38692, pcovr_38692 = opt.curve_fit(hill_r, usr_38692, r_38692)
ref_38692 = hill_r(usr_38692, *popr_38692)
popr_38693, pcovr_38693 = opt.curve_fit(hill_r, usr_38693, r_38693)
ref_38693 = hill_r(usr_38693, *popr_38693)

env1_up = ref_38691 + np.std(r_38691 -ref_38691)
env1_down = ref_38691 - np.std(r_38691 -ref_38691)
env2_up = ref_38692 + np.std(r_38692 -ref_38692)
env2_down = ref_38692 - np.std(r_38692 -ref_38692)
env3_up = ref_38693 + np.std(r_38693 -ref_38693)
env3_down = ref_38693 - np.std(r_38693 -ref_38693)


"Estimating Conductivity at high-pressures and temperatures?"

mg75fe25 = ec_predict(p38692, temp_38692)

mg95fe5 = mg75fe25/10e4
mg98fe2 = mg75fe25/(10**3.36)


plt.plot(temp_38692, mg95fe5)
plt.plot(temp_38692, mg98fe2)
plt.plot(temp_38692, mg75fe25)
plt.yscale('log')
plt.show()






"Reynolds number calculations????"
reynolds_mag(40, 1e6, sigma0_38691, .0001)


"""Plots"""

"Electrical Conductivity v Pressure and Temperature "
fig = plt.figure('Electrical Conductivity v Pressure, Temperature')
ax1 = fig.add_subplot(211)
ax1.scatter(p38691, sigma0_38691, label = 'MgO $\sigma_{0}$', c = '#5C2E7E')
ax1.scatter(p38692, sigma0_38692, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O $\sigma_{0}$', c = '#e5b3fe')
ax1.scatter(p38693, sigma0_38693, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O $\sigma_{0}$', c = '#4C6793')
ax1.set_ylabel('Conductivity $\sigma$ (S $\cdot$ m$^{-1}$)')
ax1.set_xlabel('Pressure (GPa)')
ax1.set_yscale('log')
ax1.legend()
ax2 = fig.add_subplot(212)
ax2.scatter(temp_38691, sigma0_38691, label = 'MgO $\sigma_{0}$', c = '#5C2E7E')
ax2.scatter(temp_38692, sigma0_38692, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O $\sigma_{0}$', c = '#e5b3fe')
ax2.scatter(temp_38693, sigma0_38693, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O $\sigma_{0}$', c = '#4C6793')
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
ax1.set_ylabel('Reflectivity (%)')
ax1.scatter(p38691, r_38691, color = 'white',label = 'MgO')
ax1.scatter(p38692, r_38692, color = 'white',label = '(Mg$_{0.98}$,Fe$_{0.02}$)O')
ax1.scatter(p38693, r_38693, color = 'white',label = '(Mg$_{0.95}$,Fe$_{0.05}$)O')
ax1.set_xlabel('Pressure (GPa)')
ax2 = ax1.twiny()
ax2.scatter(usr_38691, r_38691,c = '#348aa7', label = 'MgO')
ax2.scatter(usr_38692, r_38692,c = '#5dd39e', label = '(Mg$_{0.98}$,Fe$_{0.02}$)O')
ax2.scatter(usr_38693, r_38693,c = '#b5b8b2', label = '(Mg$_{0.95}$,Fe$_{0.05}$)O')
ax2.plot(usr_38691, ref_38691,c = '#348aa7', linewidth = '2')
ax2.plot(usr_38692, ref_38692,c = '#5dd39e', linewidth = '2')
ax2.plot(usr_38693, ref_38693,c = '#b5b8b2', linewidth = '2')
ax2.set_xlabel('U$_S$ Shock Velocity (km/s)')
ax2.fill_between(usr_38691, env1_up, env1_down, color = '#63b3ce', alpha = 0.4)
ax2.fill_between(usr_38692, env2_up, env2_down, color = '#86deb6', alpha = 0.4)
ax2.fill_between(usr_38693, env3_up, env3_down,color = '#b2b5af', alpha = 0.4)
ax2.legend()
plt.show()




"plots in progress"

#popt_ec38691, pcov_ec38691 = opt.curve_fit(ec_fit, temp_38691, sigma0_38691)
#popt_ec38692, pcov_ec38692 = opt.curve_fit(ec_fit, temp_38692, sigma0_38692)
#popt_ec38693, pcov_ec38693 = opt.curve_fit(ec_fit, temp_38693, sigma0_38693)


#sig_38691 = ec_predict(p38691, t_38691)
#sig_38692 = ec_predict(p38692, t_38692)
#sig_38693 = ec_predict(p38693, t_38693)

# fig = plt.figure(1)
# ax1 = fig.add_subplot(111)
# ax1.scatter(p38693, sig_38693, color = 'white')
# #ax1.scatter(t_38692, sig_38692, c = '#e5b3fe', label = '(Mg$_{0.98}$,Fe$_{0.02}$)O')
# #ax1.scatter(t_38691, sig_38691, c = '#7900FF', label = 'MgO')
# #ax1.legend()
# ax1.set_ylabel('$\sigma$ (S/m)')
# ax1.set_yscale('log')
# ax2 = ax1.twiny()
# ax2.set_xlabel('Temperature (K)')
# ax1.set_xlabel('Pressure (GPa)')
# ax2.scatter(t_38693, sig_38693, c = '#7900FF', label = '(Mg$_{0.95}$,Fe$_{0.05}$)O')
# plt.show()



#plt.scatter(temp_38691, log_chi38691, label = 'MgO')
#plt.scatter(temp_38692, log_chi38692, label = '(Mg$_{0.98}$,Fe$_{0.02}$)O')
#plt.scatter(temp_38693, log_chi38693, label = '(Mg$_{0.95}$,Fe$_{0.05}$)O')
#plt.legend()
#plt.ylabel('$\sigma$($\chi_{Fe}$)')
#plt.xlabel('Temperature (K)')
#plt.show()









