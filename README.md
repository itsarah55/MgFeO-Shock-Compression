(Mg,Fe)O Analysis Plots

These plots illustrate plotting raw data via IGOR analysis. The IGOR analysis should provide shock velocity, temperature, and reflectivity. 

"electricalconductivity_reflectivity" should give the majority of plots. There are fixed constants within the code, such as 

rho0 : initial density
molmass : molar mass 
n0 : initial refractive index
omega : laser frequency
e_0 : emissivity of free space
e_charge : electron charge 
mstar : from McWilliams 2012 for effective mass of an electron

This data set reads data from the temperature & shock velocity and reflectivity & shock velocity data sets from IGOR. It will then create a temperature fit in the same length as the reflectvity data (typically more data points). 

Plots from this script include: 

-Minimum Basal Magma Ocean Thickness based on electrical conductivity data;
-Electrical Conductivity & Pressure/Temperature with the expected electrical conductivity from McWilliams;
-Pressure/Shock Velocity & Reflectivity with Hill Function fits and 1-sigma envelopes;
-Temperature & Shock Velocity with power law fits;

"mgo_compare" This contains constants from impedance matching (shock velocity, particle velocity, pressure, and density) including data from McCoy 2019, Root 2015, McWilliams 2012, and Hansen 2021.

Plots from script:

-Shock Velocity and Particle Velocity relationship with cubic fit from Root 2015 and linear fit from McWilliams 2012

Each folder 38691-4 is the shot number and their designated data files. Please email me sarah.harter55@gmail.com for the IGOR experiment files.
-Pressure and Density 

"reflectivity_us" is the reflectivity comparison from McWilliams 2012, Bolis 2016, McCoy 2019, and Hansen 2021. Our data is filtered through a low-pass filter that attenuates the high-frquencies while keeping the low-frequencies. This is in effort to filter out any noise.

