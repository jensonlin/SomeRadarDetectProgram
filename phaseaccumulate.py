import numpy as np
import matplotlib.pyplot as plt

def awgn(x,snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)
    mu = 0
    npower = xpower/snr
    sigma = np.sqrt(npower)
    return np.random.normal(mu,sigma,size=len(x)),sigma

N = 200
s = np.zeros(N)
sig_db = 35
s[50] = np.sqrt(10**(sig_db/10.0))
[w1,sigma] = awgn(s,5) ####this w is constant if awgn is called only once
s1 = s+w1   
total = 10*np.log(np.sum(s1**2)/N)####estimate sigma
print('noise power(one pulse period) : %ddB'%total)
print('signal power(one pulse period) ; %ddB'%sig_db)

w0 = np.zeros((N,10))
for k in range(10):
    [w,sigma] = awgn(s,5)
    w0[:,k] = w

s0 = np.concatenate((s+w0[:,0],s+w0[:,1],s+w0[:,2],s+w0[:,3],s+w0[:,4],s+w0[:,5],s+w0[:,6],s+w0[:,7],s+w0[:,8],s+w0[:,9]))##this just generates the same sequence
sc0 = np.zeros(N)


for j in range(10):
    sc0 = sc0 + s0[j*N:(j+1)*N]

sc = sc0
#sc = s+w   ###this for no accumulation
fig1 = plt.figure()
plt.plot(10*np.log10(s1**2),label='one period clutter')
plt.ylabel('Amplitude/dB')
plt.xlabel('range')
fig1.show()

####cfar####
#sc = sc**2
pfa = 1e-4
g = 2
r = 10
n = 2*r
r_cell = np.zeros(n)
test = np.zeros(N)
test_ideal = np.zeros(N)
sig_loc = np.zeros(N)
sigma0 = sigma**2

for i in range(N):
    if i-g-r>=0 and i+g+r<=N-1:
        r_cell = np.concatenate((sc[i-g-r:i-g-1],sc[i+g+1:r+i+g]))
        
    sigma2_e = np.sum(r_cell**2)/n
    #print(sigma2_e)
    a_temp = pfa**(-1/n)
    alpha = n*(a_temp-1)
    alpha0 = -np.log(pfa)
    #print(alpha)
    test_ideal[i] = alpha0*sigma0   ###sigma appears in square not in db,10sigma
    test[i] = alpha*sigma2_e
    #test[i] = sigma2_e
    if test[i]<sc[i]**2 and i-g-r>=0 and i+g+r<=N-1:
        sig_loc[i]=1

fig2 = plt.figure()

plt.plot(10*np.log10(sc**2),'r-',label='signal with white gaussian noise')
plt.ylabel('Amplitude/dB')
plt.xlabel('range')
plt.legend(loc='lower right')
plt.hold(True)
#plt.plot(test,'b-')
plt.plot(10*np.log10(test),'b-.',label='the adaptive threshold')
plt.legend(loc='lower right')
plt.hold(False)
fig2.show()

fig3 = plt.figure()

plt.plot(sig_loc,label='the estimate location')
plt.legend(loc='upper right')
plt.xlabel('range')
fig3.show()














