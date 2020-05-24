#########match-filter#######

import numpy as np
import matplotlib.pyplot as plt


def wgn(x,snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)
    mu = 0
    npower = xpower/snr
    sigma = np.sqrt(npower) #####standard-deviation
    return np.random.normal(mu,sigma,size=len(x)),sigma,mu

N = 200   ####signal length
s_o = np.zeros(N)   ####original-signal

s_o[20] = 2
s_o[100] = 1
s_o[150] = 3

[w,sqrt_sig,mean] = wgn(s_o,15)

s = s_o + w   #####generate-signal-with-white-gaussin-noise

h = np.conj(s[::-1])   ####conjugate-and-flip

test = np.convolve(s,h)

a_test = abs(test)

fig1 = plt.figure()

plt.plot(a_test,label='result-of-mf-ops')
plt.legend(loc='upper right')
#fig1.show()

######the aforementioned is 0 delay
######the following is with non-zero delay

delaylen = np.random.randint(0,500,size=1)
print('delay num: ',delaylen[0])

afore = np.random.normal(mean,sqrt_sig,size=delaylen)

total = np.concatenate((afore,s))

delay_out = abs(np.convolve(total,h))


fig2 = plt.figure()

plt.plot(delay_out,label='non-zero-delay-with-mf')
plt.legend(loc='upper right')
fig2.show()

####count the delay number,find out the location

loc = len(delay_out)-len(a_test)
print('target loc: ',loc)


