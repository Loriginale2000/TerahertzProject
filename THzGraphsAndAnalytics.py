import matplotlib.pyplot as plt
import numpy as np

#First, I import data from files in the same directory to get the reference and sample data#
data1 = np.loadtxt("AmpTimeRef.txt",dtype=float)
data2 = np.loadtxt("AmpTimeSample.txt",dtype=float)

#The x1,x2 parameter are here multiplied with 1e12 to get the right units (pico-seconds)
x1 = data1[:,0] * 1e12
y1 = data1[:,1]
x2 = data2[:,0] * 1e12
y2 = data2[:,1]

plt.xscale("linear")
plt.yscale("linear")
plt.xlabel("Time (psec)")
plt.ylabel("Amplitude (a.u.)")
plt.title("Graph 0.1: Amplitude versus Time")
plt.plot(x1,y1)
plt.plot(x2,y2, color = "maroon")
plt.grid("--")
plt.legend([" Graph 0.1a:Amplitude versus Time", " Graph 0.1b:Amplitude versus Time, Sample"])
plt.show()



#Attempt to perform Fourier Transform (Reference Data)
#We are going to perform a fast Fourier transform
frequency1domain = np.fft.fft(y1)
f1axis=np.fft.fftfreq(len(y1),1.93e-14)
fx1rightunits = f1axis / 1e12

#First plot should be the real part of the FFT
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.plot(np.abs(fx1rightunits),(np.abs(frequency1domain)), linewidth = 0.7)
plt.yscale("log")
plt.xscale("linear")
plt.xlim(xmax = 5)
plt.title("Graph 1.1: Amplitude - Frequency")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (logarithmic scale)")
#Now we are going to get the second (imaginary) part
plt.subplot(1,2,2)
plt.plot(np.angle(frequency1domain))
plt.title("Graph 1.2: Phase - Frequency")
plt.yscale("linear")
plt.xscale("linear")
plt.xlim(xmax = 15, xmin = 0)
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (radians)")
plt.show()

#Attempt to perform Fourier Transform (Sample Data)
#We are going to perform a fast Fourier transform
frequency2domain = np.fft.fft(y2)
f2axis=np.fft.fftfreq(len(y2),0.02e-12)
fx2rightunits = f2axis / 1e12

#First plot should be the real part of the FFT
plt.figure(figsize = (15,10))
plt.subplot(1,2,1)
plt.plot(np.abs(fx2rightunits),(np.abs(frequency2domain)),color='maroon', linewidth = 0.7)
plt.yscale("log")
plt.xscale("linear")
plt.xlim(xmax = 5)
plt.title("Graph 1.3: Amplitude - Frequency (S.))")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (logarithmic scale)")
#Now we are going to get the second (imaginary) part
plt.subplot(1,2,2)
plt.plot((np.angle(frequency2domain, deg=False)), color = "maroon", linewidth = 1)
plt.title("Graph 1.4: Phase - Frequency (S.)")
plt.xscale("linear")
plt.yscale("linear")
plt.xlim(xmax = 15, xmin = 0)
plt.ylim(ymax = 3, ymin = -3)
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (radians)")
plt.show()

