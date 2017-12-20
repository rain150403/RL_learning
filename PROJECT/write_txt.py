import os
import numpy
from PIL import Image
from pylab import *

fi = open( "val_left.txt", "a")
def get_imlist(path):
	return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

c = get_imlist(r"/home/wangmeimei/traffic/val/left/")
print ('c:', c)
d = len(c)
print ('d:', d)

data = numpy.empty((d, 1*784))
print('data:', data)

label = zeros((1,3))
label[0][0] = 1
print label

while d>0:
	img = Image.open(c[d-1])
	print ('img,d:', img,d)
	#d = d - 1

	img_ndarray = numpy.asarray(img, dtype = 'float64')/256
	print( "img_ndarray:", img_ndarray.ndim)
	data[d-1] = numpy.ndarray.flatten(img_ndarray)
	print( "data:", data[d-1].ndim)

	#print("data:", data[d-1])
	
	numpy.savetxt(fi, data[d-1].reshape((1,-1)), fmt = "%4.3f", delimiter=' ')
	numpy.savetxt(fi, label, fmt = "%d")	
	#numpy.savetxt(fi, data[d-1].reshape((1,-1)), fmt = "%4.3f", delimiter=' ', newline=' ')
	#numpy.savetxt(fi, label, fmt = "%d" , newline='\n')

	d = d-1

