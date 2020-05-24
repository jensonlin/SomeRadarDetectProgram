import numpy as np
import matplotlib.pyplot as plt

snr = 10
signal_lg = 35
sigma2_lg = signal_lg - snr
pfa = 1e-4
signal = np.sqrt(10**(signal_lg/10.0))
sigma = np.sqrt(10**(sigma2_lg/10.0))
#print(signal)
mu = 0
N = 200

si = np.zeros(N)
si[50] = signal    ####signal location 50
noise = np.random.normal(mu,sigma,size=len(si))
s = si + noise

fig1 = plt.figure()
plt.plot(10*np.log10(s**2))
plt.xlabel('range')
plt.ylabel('Amplitude/dB')
fig1.show()

total = np.sum(s**2)/N
#print(total,sigma**2)
g = 2
r = 10
n = 2*r
r_cell = np.zeros(n)
test = np.zeros(N)
test_ideal = np.zeros(N)
sig_loc = np.zeros(N)

for i in range(N):
    if i-g-r>=0 and i+g+r<=N-1:
        r_cell = np.concatenate((s[i-g-r:i-g-1],s[i+g+1:r+i+g]))

    sigma2_e = (np.sum(r_cell**2))/n
    a_temp = pfa**(-1/n)
    alpha = n*(a_temp-1)
    alpha0 = -np.log(pfa)
    test_ideal[i] = alpha0*sigma**2
    test[i] = alpha*np.abs(sigma2_e)
    if test[i]<s[i]**2 and i-g-r>=0 and i+g+r<=N-1:
        sig_loc[i] = 1


fig2 = plt.figure()

plt.plot(10*np.log10(s**2),label='signal with noise')
#plt.plot(s**2)
plt.plot(10*np.log10(test),label='adaptive threshold')
#plt.plot(test)
plt.plot(10*np.log10(test_ideal),'-.',label='invariant threshold')
plt.legend(loc='lower right')
plt.xlabel('range')
plt.ylabel('Amplitude/dB')
fig2.show()

fig3 = plt.figure()
plt.plot(sig_loc)
plt.ylabel('detection')
fig3.show()




