import numpy as np

import matplotlib.pyplot as plt



fs = 1e7
B = 1e6
T = 1e-4
t = np.arange(-T/2,T/2,1/fs)
t0 = np.arange(-T,T,1/fs)

#f0 = 2e6
####this sets up the signal
f0 = 0
p = 2*np.pi
fr = B/(2*T)
fre = fr*t
phase0 = f0+fre
phase = phase0*t
pai = phase*p
c = np.cos(pai)
s = np.sin(pai)

###fig1 for real ,fig2 for imag
fg1 = plt.figure()
plt.plot(t,c,'b-',label='Real_part')
plt.ylim((-1.4,1.4))
plt.xlabel('t/s')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')
#plt.savefig('D:\\real.png')
#plt.show()

fg2 = plt.figure()
plt.plot(t,s,'r-',label='Imag_part')
plt.ylim((-1.3,1.3))
plt.xlabel('t/s')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')
#plt.savefig('D:\\imga.png')
#plt.show()

#fg1.show()
#fg2.show()
###for fft
sigc = np.array(c)
sigs = np.array(s)
l = len(c)
sig = np.zeros_like(sigc,dtype=complex)  ##default dtype is float,should declare first

for i in range(l):
    sig[i] = complex(sigc[i],sigs[i])
  
sig01 = abs(np.fft.fft(sig))
sig0 = np.fft.fftshift(sig01)
l0 = len(sig0)
f0 = np.arange(l0)/l0
f = fs * f0-(fs/B)*B/2

fig3=plt.figure()
plt.plot(f,sig0,label='result_of_fft')
#plt.ylim((0,16))
plt.xlabel('f/hz')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')
#plt.savefig('D:\\fft.png')
#fig3.show()

###with match filter
h = np.zeros_like(sigc,dtype=complex)

for j in range(l):
    h[j] = complex(sigc[j],-sigs[j])
    
s0 = np.convolve(sig,h)
s02 = abs(s0)


lh = len(s02)
#m = max(s02)

fig4 = plt.figure()
plt.plot(t0,s02,label='with match_filter')
#plt.plot(abs(np.diff(s02)),label='diff_of_s02')
plt.legend(loc='upper right')
plt.xlabel('t/s')
plt.ylabel('Magnitude')
plt.grid(True)
#plt.savefig('D:\\mf.png')
fig4.show()



#####diff
di = np.array([])
#l
for k in range(lh-2):
    if s02[k+1]>s02[k]:
        if s02[k+1]>s02[k+2]:
            di = di.append(s02[k])
        else:
            di = di
    else:
        di = di
print(di)
#np.delete(extrem0,np.where(extrem0==peak),axis=0)
#e = max(extrem0)




###hamming window

w1 = np.cos(np.pi*t/T)
w = 0.08 + 0.92 * w1 * w1

hw = h * w
hw0 = abs(np.convolve(sig,hw))

#m = max(hw0)

#sli = np.zeros(74)
#for k in range(74):
#    sli[k] = hw0[k+25]
#m2 = max(sli)
#print('最大值:',m)
#print('副瓣:',m2)


fig5 = plt.figure()
plt.plot(t0,hw0,label='with hamming window')
#plt.plot(t0,s02,label='match filter')
plt.xlabel('t/s')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')
plt.grid(True)
fig5.show()

#fig6 = plt.figure()
#plt.scatter(t0,hw0,label='with hamming window')
#plt.legend(loc='upper right')
#fig6.show()

###for the second biggest
m = max(hw0)
m3d = 0.707*m
cout = 0
for n in range(l):
    if hw0[n]>=m3d:
        cout = cout + 1
        print(n)

print(cout)
t3d = 2*T/((fs/B)*200)*(cout-1)
print('三分贝宽度:',t3d*1e6,'微秒')


###full fig about changing fs
fig7 = plt.figure()
plt.subplot(1,3,1)
plt.plot(t*1e6,c,'b-',label='Real_part')
plt.ylim((-1.4,1.4))
plt.xlabel('t/μs')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')

plt.subplot(1,3,2)
plt.plot(t*1e6,s,'r-',label='Imag_part')
plt.ylim((-1.3,1.3))
plt.xlabel('t/μs')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')

plt.subplot(1,3,3)
plt.plot(f/1e6,sig0,label='result_of_fft')
#plt.ylim((0,16))
plt.xlabel('f/Mhz')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')


#fig7.show()

