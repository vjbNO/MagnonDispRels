"""
This script is based on the manual of Boris Computational Spintronics v3.0 by Serban Lepadatu, 2020

It produces the magnon dispersion realtion for an easy axis toy antiferromagnet inspired by hematite
using magnetic field excitation (sinc function)
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

##### set up simulation ###

#Setup

fc = 5e12
time_step = 0.1e-12 
N=600

mesh_step = 5e-9

#mesh dimensions 
meshdim = [1e-6, mesh_step, mesh_step]
N=600
L = meshdim[0]
kc = 2*np.pi*N/(2*L)

cellsize = [mesh_step,mesh_step,mesh_step]
'''
#sinc pulse time centre (s) and total time to simulate (s)
#increase t0 if you want better resolution
t0 = 10e-12 #200e-12
equiltime = 0 #100e-12
total_time = 2 * t0

#bias field and excitation strength (A/m)
H0 = 0 #5000e3
He = 500e3

# material parameters
A = 76e-15 # J/m
Ahom = -460e3 # J/m^3
K1 = 2.1e3 # J/m^3 #ten times that of hematite to have larger gap to play with
Ms = 2.1e3 # A/m

#setup mesh 
ns.setafmesh('hematite',meshdim)
ns.cellsize(cellsize)
ns.pbc('hematite', 'x', 5)
#ns.pbc('hematite', 'y', 5)
#ns.pbc('hematite', 'z', 5)

ns.setangle(90, 0) #initialize along x direction (GS)
# which is easy axis direction (implicit), can be changed

#setup sinc pulse using a formula
ns.setstage('Hequation')
ns.editstagevalue(0, 'H0, He * sinc(kc*(x-Lx/2))*sinc(kc*(y-Ly/2))*sinc(2*PI*fc*(t-t0)), 
		He * sinc(kc*(x-Lx/2))*sinc(kc*(y-Ly/2))*sinc(2*PI*fc*(t-t0))')

#define the equation constants
#Bias field (A/m)
ns.equationconstants('H0', H0)
#Excitation field (A/m)
ns.equationconstants('He', He)
cutoff wavevector (rad/m)
ns.equationconstants('kc', kc)
#cutoff frequency
ns.equationconstants('fc', fc)
#time center (s)
ns.equationconstants('t0', t0)

ns.setparam('hematite', 'damping_AFM', [0.001,0.001]) 
#set damping to zero to make peaks sharper
ns.setparam('hematite', 'Ms_AFM', [Ms,Ms])
ns.setparam('hematite', 'A_AFM', [A,A])
ns.setparam('hematite','Ah',[Ahom,Ahom])
ns.setparam('hematite','Anh',[0,0])
ns.setparam('hematite','K1_AFM',[K1,K1])


#make sure output file is wiped clean
output_file = 'path'
ns.dp_newfile(output_file)

ns.setode('LLG', 'RK4')
ns.setdt(1e-15)

#### Run simulation #####

time = 0.0
#ns.cuda(1) # turn on if you have graphic card

#simulate, saving data every time_step

while time < total_time:
    
    ns.editstagestop(0, 'time', time + time_step)
    ns.Run()
    #get magnetisation profile along length through center
    ns.dp_getexactprofile([cellsize[0]/2, meshdim[1]/2 + cellsize[1]/2, 0], 
			[meshdim[0] - cellsize[0]/2, meshdim[1]/2 + cellsize[1]/2, 0], mesh_step, 0)
    #save only the y component of magnetisation at time_step intervals
    ns.dp_div(2, Ms)
    ns.dp_saveappendasrow(output_file, 2) 
    
    time += time_step

###### Analysis #####

#get 2D list as position along horizontal, time along vertical
pos_time = np.loadtxt(output_file)

#2D FFT
fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

#get value ranges
freq_len = len(fourier_data)
k_len = len(fourier_data[0])
freq = sp.fft.fftfreq(freq_len, time_step)
kvector = sp.fft.fftfreq(k_len, mesh_step)

#maximum k and f values
k_max = 2*np.pi*kvector[int(0.5 * len(kvector))]*mesh_step
f_min = np.abs(freq[0])
f_max = np.abs(freq[int(0.5 * len(freq))])/1e12
f_points = int(0.5 * freq_len)

#extract result from fourier data in a plottable form
result = [fourier_data[i] for i in range(int(0.5 * freq_len),freq_len)]

fig1,ax1 = plt.subplots()

#plot spin wave dispersion
ax1.imshow(result, origin='lower', interpolation='bilinear', extent = [-k_max, k_max, f_min, f_max], aspect ="auto")


ax1.set_xlabel('qa')
ax1.set_ylabel('f (THz)')
plt.tight_layout()

plt.savefig('name.pdf', dpi = 600)

plt.show()
