import numpy as np 
import matplotlib.pyplot as plt


#INSERT DATA 
data1 = np.loadtxt("AmpTimeRef.txt",dtype=float)
data2 = np.loadtxt("AmpTimeSample.txt",dtype=float)


t1= data1[:,0] 
y1 = data1[:,1]
t2 = data2[:,0] 
y2 = data2[:,1]

# INTERPOLATION
startt = 0
endt = 46.1 * 1e-12
dt = min(np.mean(np.diff(t2)), np.mean(np.diff(t1)))
t = np.arange(startt, endt + dt, dt)

sr = np.interp(t, t1, y1, left=0, right=0)
ss = np.interp(t, t2, y2, left=0, right=0)

# FFTS
fftr = np.fft.fft(sr, 16384)
ffts = np.fft.fft(ss, 16384)
freq = np.fft.fftfreq(16384, dt) #time to frequency


# MASKING THE NEGATIVE FREQUENCIES
mask = (freq >= 0) 
fftr = fftr[mask]
ffts = ffts[mask]
freq = freq[mask]

freq[freq == 0] = 1e-12  # Replace zeros with a small value to avoid division by zero

# TRANSFER FUNCTION
tamp = (ffts / fftr)
ln_exp = np.log(np.abs(tamp))
tphase = np.angle(tamp)

# PHASE
tphase_unwrapped = np.unwrap(tphase)
tphase_unwrapped -= tphase_unwrapped[0]-0.5*np.pi

uwr = np.unwrap(np.angle(fftr))
uws= np.unwrap(np.angle(ffts))
delta = uws - uwr

# REFRACTIVE INDEX THEORETICAL
dt =  6* 1e-12 #seconds
c = 3e8 #speed of light in m/s
d = 2e-3 #thickness of sample in m
n_0 = 1  # Refractive index of air

navg = 1 + (c * dt) / d
print("The average refractive index is: ", navg)

n = 1 + (np.abs(delta)* c) / (2 * np.pi * freq * 1e12  * d)

#NEWTON RAPHSON METHOD
def G_zero(n, f, ln_exp):

    ln_th =(np.log((4 * n_0 * n) / (n_0 + n)**2)
             -(1j * 2 * np.pi * f * d / c) * (n - n_0))
    return  ln_th - ln_exp

def G_zerod(n, f):
    p = 1 / n
    m = 2 / (n + n_0) + (1j * 2 * np.pi* f * d) / c 
    return p - m 

def new_raph(n_1, ln_exp, f, iterations, tol):
    n = n_1
    for i in range(iterations): 
        G_zero_ = G_zero(n,  f, ln_exp)
        G_zerod_ = G_zerod(n, f)
        dn = G_zero_ / G_zerod_
        n = n - dn
        if abs(G_zero_/G_zerod_) < tol:
            print(i, "iterations")
            break
    return n


# INITIAL GUESS AND CALCULATION
n_1 = 1.97 + 0.05j
n_ = []

for i, f in enumerate(freq, 0):
    ln_exp = np.log(np.abs(tamp[i])) + 1j * tphase_unwrapped[i]
    n_values = new_raph( n_1 ,ln_exp, f,  iterations=50, tol=1e-10)
    n_.append(n_values)


# print("Refractive index values:", n_)
n_ = np.array(n_)



#PLOTTING

#AMPLITUDE VERSUS TIME
plt.xscale("linear")
plt.yscale("linear")
plt.xlabel("Time (psec)")
plt.ylabel("Amplitude (a.u.)")
plt.title("Graph 0.1: Amplitude versus Time")
plt.plot(t,sr, linewidth = 1)
plt.plot(t,ss, color = "maroon",  linewidth = 1)
plt.legend([" Graph 0.1a:Amplitude versus Time", " Graph 0.1b:Amplitude versus Time, Sample"])
plt.show()

#THEORETICAL REFRACTIVE INDEX
plt.plot( freq, n , color = "darkblue", linewidth = 0.7)
plt.title("Graph 2.5: Refractive index (n) versus Frequency (f)", fontsize = 9)         
plt.xlabel("Frequency (THz)", fontsize = 11, loc = "right")
plt.ylabel("Refractive index (unitless)", fontsize = 11)  
plt.xscale("linear")
plt.yscale("linear")  
plt.xlim(xmax = 3, xmin = 0 )
plt.ylim(ymax = 10, ymin = -2)
plt.show()


#REFRACTIVE INDEX REAL PART
plt.title("Graph 2.6: Real part of Refractive index (n) versus Frequency (f)", fontsize = 9)
plt.plot ( freq * 1E12, np.real(n_), linewidth=0.7, color='darkblue')
#plt.xlim(xmax = 5, xmin = 0)
#plt.ylim(ymax = 10, ymin = -2)
plt.show()