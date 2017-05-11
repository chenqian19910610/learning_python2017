# # """
# # format in python
# # """
# a='{1} {2} {0}'.format('one','two','three')
# print(a)
#
# a='{:>10}'.format('test') # align to the right
# a='{:^10}'.format('test') # align to the middle
# print(a)
#
# a='{:.3}'.format('abcdefghijk') # truncating words to specified length
# a='{:>10.3}'.format('abcdefghijk') # combine truncating and padding
# print(a)
#
# numbers=[23.00,0.123545684,1,4.2,9887.3] #floats
# for number in numbers:
#     print('{:10.4f}'.format(number))
#
# a=round(12358546546, -5) # round integers
# print(a)
#
# a='{:+d}'.format(42) # print signs
# b='{:d}'.format(-42)
# print(a,b)
#
# data={'first':'abc','second':'123'} #named placeholders
# print('{first}{second}'.format(**data))
#
# person={'firstname':'Qian','lastname':'Chen'} # how to get information !!!
# print('{p[firstname]} {p[lastname]}'.format(p=person))
# data=[4,5,6,7,8]
# print('{d[0]} {d[3]}'.format(d=data))
#
# class Plant(object):
#     type='tree'
#     kinds=[{'name':'oak'},{'name':'maple'}]
# a='{p.type}:{p.kinds[0][name]}'.format(p=Plant()) #按照给定的格式输出指定的值
# print(a)
#
# from datetime import datetime #print time
# a='{:%Y-%m-%d %H:%M}'.format(datetime(2017,4,30,16,25))
# print(a)
#
# a='{:{prec}}={:{prec}}'.format('chenqian',2.7184568,prec='.3')#truncate with parameterized format
# print(a)
#
# """
# numpy in python
# """
# import numpy as np
# a=np.array([1,2,3])
# b=np.transpose(a)
# print(np.dot(a,b))
#
# import sympy
# from sympy import *
# x=sympy.Symbol('x')
# y=sympy.Symbol('y')
# m=sympy.simplify(x+y+x-y)
# a=sympy.limit(sympy.sin(x),x,0)
# b=sympy.diff(sympy.tanh(x),x)
# print(m,a,b)
#
# def f(x):
#     y=sympy.sin(x)*sympy.tanh(x)
#     return y
# a=sympy.diff(f(x),x)
# b=sympy.lambdify(x,a)  # transform to lambda function to make calculation faster
# c=b(6)
# print(a,'the derivative is',c)
#
# import sympy
# x=sympy.Symbol('x')
# a=float(sympy.integrate(3*x**5+sympy.sin(x),(x,15,float('inf'))))
# print(a)
#
# M=sympy.zeros(3,5)
# I=sympy.eye(3)
# print(M,I)
#
# import numpy
# from numpy import array
# from numpy.linalg import inv,det
# M=[[25,67],[13,51]]
# N=array(M)
# print(det(M),inv(M),N)
#
#
# """
# plot KNN
# matplotlib: http://matplotlib.org/examples/index.html
# """
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn import neighbors,datasets
#
# n_neighbor=15
#
# iris=datasets.load_iris()
# X=iris.data[:,:2]
# y=iris.target
# print(X,y)
# h=.2  #step size in the mesh
#
# #create color maps
# cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
# cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
#
# for weight in ['uniform','distance']:
#     clf=neighbors.KNeighborsClassifier(n_neighbor,weights=weight)
#     clf.fit(X,y)
#
#     x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
#     y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
#     xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max))
#     Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
#     Z=Z.reshape(xx.shape)
#
#     plt.figure()
#     plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
#
#     plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold)
#     plt.xlim(xx.min(),xx.max())
#     plt.ylim(yy.min(),yy.max())
#
#     plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbor,weight))
#
# plt.show()
#

# import numpy as np
# rows=3
# columns=4
# a=np.matrix(np.random.randint(1,100,size=(rows,columns))) #generate a random matrix
# print(a)

"""
pyplot tutorial
"""
import matplotlib.pyplot as plt
import numpy as np

# plt.rcdefaults()
# fig,ax=plt.subplots()
#
# people=('Tom','ARPIL','DASDHAS')
# y_pos=np.arange(len(people))
# performance=3+10*np.random.rand(len(people))
# error=np.random.rand(len(people))
#
# ax.barh(y_pos,performance,xerr=error,align='center',color='green',ecolor='black')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(people)
# ax.invert_yaxis()
# ax.set_xlabel(performance)
# ax.set_title('adasdsd')
# plt.show()

# x=np.linspace(0,1,500)
# y=np.sin(4*np.pi*x)*np.exp(-5*x)
# fig,ax=plt.subplots()
# ax.fill(x,y,zorder=10)  #order of lines or texts, on top or under
# ax.grid(True,zorder=5)
# plt.show()

# x=np.linspace(0,2*np.pi,500)
# y1=np.sin(x)
# y2=np.sin(3*x)
# fig,ax=plt.subplots()
# ax.fill(x,y1,'b',x,y2,'r',alpha=0.1) #alpha-->the depth of color
# plt.show()

# x=np.linspace(0,10,500)
# dashes=[10,5,100,5] #10 points on, 5 points off
# fig,ax=plt.subplots()
# line1,=ax.plot(x,np.sin(x),'--',linewidth=2,label='dashes set retroactively')
# line1.set_dashes(dashes)
# line2,=ax.plot(x,-1*np.sin(x),dashes=[30,5,10,5],label='dashes set proactively')
# ax.legend(loc='lower right')
# plt.show()

# fig,ax=plt.subplots()
# for color in ['red','green','blue']:
#     n=750
#     x,y=np.random.rand(2,n)
#     scale=200*np.random.rand(n) #size of the circle
#     ax.scatter(x,y,c=color,s=scale,label=color,alpha=0.3,edgecolors='none')
# ax.legend()
# ax.grid(True)
# plt.show()

# import matplotlib.path as mpath
# import matplotlib.patches as mpatches
# fig, ax = plt.subplots()
#
# Path = mpath.Path
# path_data = [
#     (Path.MOVETO, (1.58, -2.57)),
#     (Path.CURVE4, (0.35, -1.1)),
#     (Path.CURVE4, (-1.75, 2.0)),
#     (Path.CURVE4, (0.375, 2.0)),
#     (Path.LINETO, (0.85, 1.15)),
#     (Path.CURVE4, (2.2, 3.2)),
#     (Path.CURVE4, (3, 0.05)),
#     (Path.CURVE4, (2.0, -0.5)),
#     (Path.CLOSEPOLY, (1.58, -2.57)),
#     ]
# codes, verts = zip(*path_data)
# path = mpath.Path(verts, codes)
# patch = mpatches.PathPatch(path, facecolor='r', alpha=0.5)
# ax.add_patch(patch)
#
# # plot control points and connecting lines
# x, y = zip(*path.vertices)
# line, = ax.plot(x, y, 'go-')
#
# ax.grid()
# ax.axis('equal')
# plt.show()

# methods=[None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning','hamming','hermite',
#          'kaiser','quadric','catrom','gaussian','bessel','mitchell','sinc','lanczos']
# np.random.seed(0)
# grid=np.random.rand(4,4)
# fig,axes=plt.subplots(3,6,figsize=(12,6),subplot_kw={'xticks':[],'yticks':[]})
# fig.subplots_adjust(hspace=0.3,wspace=0.05)
# for ax, interp_method in zip(axes.flat,methods):
#     ax.imshow(grid,interpolation=interp_method,cmap='plasma')
#     ax.set_title(interp_method)
# plt.show()

# np.random.seed(0)
# n_bins=10
# x=np.random.randn(1000,3)
# fig,axes=plt.subplots(nrows=2,ncols=2)
# ax0,ax1,ax2,ax3=axes.flatten()
# colors=['red','tan','lime']
# ax0.hist(x,n_bins,normed=1,histtype='bar',color=colors,label=colors)
# ax0.legend(prop={'size':10})
# ax0.set_title('bars with legend')
#
# ax1.hist(x, n_bins, normed=1, histtype='bar', stacked=True)
# ax1.set_title('stacked bar')
#
# ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
# ax2.set_title('stack step (unfilled)')
#
# # Make a multiple-histogram of data-sets with different length.
# x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
# ax3.hist(x_multi, n_bins, histtype='bar')
# ax3.set_title('different sample sizes')
#
# fig.tight_layout()
# plt.show()



"""
plot contour
"""
# import matplotlib
# import numpy as np
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
#
# matplotlib.rcParams['xtick.direction'] = 'out'
# matplotlib.rcParams['ytick.direction'] = 'out'
#
# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
# Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# # difference of Gaussians
# Z = 10.0 * (Z2 - Z1)
#
#
# # Create a simple contour plot with labels using default colors.  The
# # inline argument to clabel will control whether the labels are draw
# # over the line segments of the contour, removing the lines beneath
# # the label
# plt.figure()
# CS = plt.contour(X, Y, Z)
# plt.clabel(CS, inline=1, fontsize=10)
# plt.title('Simplest default with labels')
#
#
# # contour labels can be placed manually by providing list of positions
# # (in data coordinate). See ginput_manual_clabel.py for interactive
# # placement.
# plt.figure()
# CS = plt.contour(X, Y, Z)
# manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
# plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
# plt.title('labels at selected locations')
#
#
# # You can force all the contours to be the same color.
# plt.figure()
# CS = plt.contour(X, Y, Z, 6,
#                  colors='k',  # negative contours will be dashed by default
#                  )
# plt.clabel(CS, fontsize=9, inline=1)
# plt.title('Single color - negative contours dashed')
#
# # You can set negative contours to be solid instead of dashed:
# matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
# plt.figure()
# CS = plt.contour(X, Y, Z, 6,
#                  colors='k',  # negative contours will be dashed by default
#                  )
# plt.clabel(CS, fontsize=9, inline=1)
# plt.title('Single color - negative contours solid')
#
#
# # And you can manually specify the colors of the contour
# plt.figure()
# CS = plt.contour(X, Y, Z, 6,
#                  linewidths=np.arange(.5, 4, .5),
#                  colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5')
#                  )
# plt.clabel(CS, fontsize=9, inline=1)
# plt.title('Crazy lines')
#
#
# # Or you can use a colormap to specify the colors; the default
# # colormap will be used for the contour lines
# plt.figure()
# im = plt.imshow(Z, interpolation='bilinear', origin='lower',
#                 cmap=cm.gray, extent=(-3, 3, -2, 2))
# levels = np.arange(-1.2, 1.6, 0.2)
# CS = plt.contour(Z, levels,
#                  origin='lower',
#                  linewidths=2,
#                  extent=(-3, 3, -2, 2))
#
# # Thicken the zero contour.
# zc = CS.collections[6]
# plt.setp(zc, linewidth=4)
#
# plt.clabel(CS, levels[1::2],  # label every second level
#            inline=1,
#            fmt='%1.1f',
#            fontsize=14)
#
# # make a colorbar for the contour lines
# CB = plt.colorbar(CS, shrink=0.8, extend='both')
#
# plt.title('Lines with colorbar')
# #plt.hot()  # Now change the colormap for the contour lines and colorbar
# plt.flag()
#
# # We can still add a colorbar for the image, too.
# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
#
# # This makes the original colorbar look a bit out of place,
# # so let's improve its position.
#
# l, b, w, h = plt.gca().get_position().bounds
# ll, bb, ww, hh = CB.ax.get_position().bounds
# CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])
#
#
# plt.show()




"""
embed subplot inside a plot
"""
# from mpl_toolkits.axes_grid.inset_locator import inset_axes
# import matplotlib.pyplot as plt
# import numpy as np
#
# # create some data to use for the plot
# dt = 0.001
# t = np.arange(0.0, 10.0, dt)
# r = np.exp(-t[:1000]/0.05)               # impulse response
# x = np.random.randn(len(t))
# s = np.convolve(x, r)[:len(x)]*dt  # colored noise
#
# fig = plt.figure(figsize=(9, 4),facecolor='white')
# ax = fig.add_subplot(121)
#
# # the main axes is subplot(111) by default
# plt.plot(t, s)
# plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
# plt.xlabel('time (s)')
# plt.ylabel('current (nA)')
# plt.title('Subplot 1: \n Gaussian colored noise')
#
# # this is an inset axes over the main axes
# inset_axes = inset_axes(ax,
#                     width="50%", # width = 30% of parent_bbox
#                     height=1.0, # height : 1 inch
#                     loc=1)
# n, bins, patches = plt.hist(s, 400, normed=1)
# #plt.title('Probability')
# plt.xticks([])
# plt.yticks([])
#
# ax = fig.add_subplot(122)
# # the main axes is subplot(111) by default
# plt.plot(t, s)
# plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
# plt.xlabel('time (s)')
# plt.ylabel('current (nA)')
# plt.title('Subplot 2: \n Gaussian colored noise')
#
# plt.tight_layout()
# plt.show()


"""
animation, need GUI
"""
# x=np.arange(6)
# y=np.arange(5)
# z=x*y[:,np.newaxis]
# for i in range(5):
#     if i==0:
#         p=plt.imshow(z)
#         fig=plt.gcf()
#         plt.clim()
#         plt.title('boring slide show')
#     else:
#         z=z+2
#         p.set_data(z)
#
#         print('step',i)
#         plt.pause(0.5)

