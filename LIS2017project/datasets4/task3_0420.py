from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np

dt=0.001
t=np.arange(0.0,10.0,dt)
r=np.exp(-t[:1000]/0.05)
x=np.random.rand(len(t))
s=np.convolve(x,r)[:len(x)]*dt

fig=plt.figure(figsize=(9,4),facecolor='white')
ax=fig.add_subplot(121)
plt.plot(t,s)
plt.axis([0,1,1.1*np.amin(s),2*np.amax(s)])
plt.xlabel('time(s)')
plt.ylabel('current(nA)')
plt.title('Subplot1')

inset_axes=inset_axes(ax, width='50%',height=1.0,loc=1)
n,bins,patches=plt.hist(s,400,normed=1)
plt.xticks([])
plt.yticks([])
ax=fig.add_subplot(122)
plt.plot(t,s)
plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
plt.xlabel('time (s)')
plt.ylabel('current (nA)')
plt.title('Subplot 2: \n Gaussian colored noise')

plt.tight_layout()
plt.show()
