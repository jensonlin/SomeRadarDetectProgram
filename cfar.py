import numpy as np
import matplotlib.pyplot as plt

def wgn(x,snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)
    mu = 0
    npower = xpower/snr
    sigma = np.sqrt(npower)
    #return np.random.rayleigh(sigma,size=len(x))
    return np.random.normal(mu,sigma,size=len(x)),sigma

#mu = 0
#sigma = 1
N = 200
pfa = 1e-4
print(pfa)
s = np.zeros(N)
#s[25] = 2
s[100] = 1
#s[150] = 3

[w,sigma0]=wgn(s,15)
sigma0 = sigma0**2
#si = s + wgn(s,10) ##snr=10
si = s + w
#a = np.random.normal(mu,sigma,size=200)
#sm = np.sum(s1)/len(s1)
#print(s1)
s1 = si**2

#a[100] = 2



####CA_CFAR
####guard cell=4,refer cell=10
g = 2
r = 10
n = 2*r
r_cell = np.zeros(n)
test = np.zeros(N)
test_ideal = np.zeros(N)
sig_loc = np.zeros(N)

for i in range(N):
    if i <= g:
       r_cell = np.concatenate((s1[N-r-g+i-1:N-g+i-1],s1[i+g+1:r+i+g+1]))
       #print(r_cell)
    elif g< i <=r+g-1:
        r_cell = np.concatenate((s1[N-r+i-g:N-1],s1[0:i-g-1],s1[i+g+1:r+i+g+1]))
        #print(r_cell)
    elif N-1-r-g<i<N-1-g:
        ###index187 right side with 12points for gurad and reference
        ###so < should not contain =
        r_cell = np.concatenate((s1[i-g-r:i-g-1],s1[i+g+1:N-1],s1[0:r-N+i+g]))
        #print(r_cell)
    elif N-1-g<=i:
        r_cell = np.concatenate((s1[i-g-r:i-g-1],s1[g-N+1+i:g-N+i+r]))
        #print(r_cell)
    else:
        r_cell = np.concatenate((s1[i-g-r:i-g-1],s1[i+g+1:r+i+g]))
        #print(r_cell)
        
    sigma2_e = np.sum(r_cell)/n
    #print(sigma2_e)
    a_temp = pfa**(-1/n)
    alpha = n*(a_temp-1)
    alpha0 = -np.log(pfa)
    #print(alpha)
    test_ideal[i] = alpha0*sigma0   ###sigma appears in square not in db
    test[i] = alpha*sigma2_e
    #test[i] = sigma2_e
    if test[i]<s1[i]:
        sig_loc[i]=1
        


fig1 = plt.figure()

###the y axis is appeared in dB
#plt.plot(s1,'r-')
plt.plot(10*np.log10(s1),'r-',label='signal with white gaussin noise')
plt.ylabel('dB')
plt.legend(loc='lower right')


plt.hold(True)

#plt.plot(test,'b-')
plt.plot(10*np.log10(test_ideal),'b-.',label='the ideal threshold')
plt.legend(loc='lower right')


plt.hold(True)

plt.plot(10*np.log10(test),'-',label='the adative threshold')
plt.legend(loc='lower right')

plt.hold(False)

fig1.show()

fig2 = plt.figure()

plt.plot(sig_loc,label='signal location')
plt.legend(loc='upper right')

fig2.show()













