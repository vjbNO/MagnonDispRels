"""
This script is based on the manual of Boris Computational Spintronics v3.0 by Serban Lepadatu, 2020 

It produces the magnon dispersion realtion for an easy axis toy antiferromagnet inspired by hematite
"""

from NetSocks import NSClient
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

font = {'size'   : 20}
mpl.rc('font', **font)

#setup communication with server
ns = NSClient()
ns.configure(True)

##### set up simulation #####

# geometry
mesh_step = 5e-9
meshdim = [1e-6, mesh_step, mesh_step]
cellsize = [mesh_step,mesh_step,mesh_step]

# run time and precision of output
time_step = 0.1e-12 # output time step
t0 = 10e-12 # time shift		
equiltime = 10e-12 # thermalization time
total_time = (2 * t0 ) + equiltime # total run time

#bias field and excitation strength (A/m)
H0 = 0 #5000e3
He = 500e3

# material parameters
A = 76e-15 # J/m
Ahom = -460e3 # J/m^3
K1 = 210 # J/m^3 ten times that of hematite to have larger gap to play with
Ms = 2.1e3 # A/m

#setup mesh and set magnetisation along +x direction
ns.setafmesh('hematite',meshdim) # anti ferromangetic mesh with two sublattices
ns.cellsize(cellsize)
ns.pbc('hematite', 'x', 5) 

ns.setangle(90, 0) # set ground state along the easy axis anisotropy which is set along x (implicitly, can be changed with ea1 parameter)

ns.setparam('hematite', 'damping_AFM', [0.001,0.001]) #set damping to zero to make peaks sharper
ns.setparam('hematite', 'Ms_AFM', [Ms,Ms])
ns.setparam('hematite', 'A_AFM', [A,A])
ns.setparam('hematite','Ah',[Ahom,Ahom])
ns.setparam('hematite','Anh',[0,0])
ns.setparam('hematite','K1_AFM',[K1,K1])

output_file = 'test' 

#make sure output file is wiped clean
ns.dp_newfile(output_file)

ns.setode('sLLG', 'RK4') # need stochastic LLG for coupling temperature
ns.setdt(1e-15) # time step of the solver, should be <= 1fs
ns.temperature(1)
ns.delmodule('hematite','Zeeman') # no need for magnetic field here.

##### Run simulation #####

time = 0.0
#ns.cuda(1) # turn on if you have graphic card

#simulate, saving data every time_step

ns.editstagestop(0, 'time', time + equiltime) # thermalization
ns.Run()

# now start recording data

while time < total_time:
    
    ns.editstagestop(0, 'time', time + equiltime + time_step)
    ns.Run()
    #get magnetisation profile along length through center
    ns.dp_getexactprofile([cellsize[0]/2, meshdim[1]/2 + cellsize[1]/2, 0], [meshdim[0] - cellsize[0]/2, meshdim[1]/2 + cellsize[1]/2, 0], mesh_step, 0)
    #save only the y component of magnetisation at time_step intervals
    ns.dp_div(2, Ms)
    ns.dp_saveappendasrow(output_file1, 2)
    time += time_step

##### Analysis #######

#get 2D list as position along horizontal, time along vertical
#pos_time = ns.Get_Data_Columns(output_file) # BORIS inbuilt function to read out data, handy but not necessary
pos_time = np.loadtxt(output_file)

# Splus = np.add(pos_time1,1j*pos_time2) # if you want to compute complex spin wave amplitude with handedness Splus and Smins, add a second output file above where you store the other transverse component
# Sminus = np.subtract(pos_time1,1j*pos_time2)

#2D FFT over time and space to get connection between frequency and k vector
fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

#get value ranges
freq_len = len(fourier_data)
k_len = len(fourier_data[0])
freq = np.fft.fftfreq(freq_len, time_step)
kvector = np.fft.fftfreq(k_len, mesh_step)

#maximum k and f values
k_max = 2*np.pi*kvector[int(0.5 * len(kvector))]*mesh_step # we scale the k vector with the lattice constant
f_min = np.abs(freq[0])
f_max = np.abs(freq[int(0.5 * len(freq))])/1e12 # to make it Tera herz
f_points = int(0.5 * freq_len)

#extract result from fourier data in a plottable form
result = [fourier_data[i] for i in range(int(0.5 * freq_len),freq_len)]

fig1,ax1 = plt.subplots()

#plot spin wave dispersion
ax1.imshow(result, origin='lower', interpolation='bilinear', 
           extent = [-k_max, k_max, f_min, f_max], 
           aspect ="auto")

ax1.set_xlabel('qa')
ax1.set_ylabel('f (THz)')

plt.tight_layout()

plt.savefig('test.pdf', dpi = 600)

plt.show()
