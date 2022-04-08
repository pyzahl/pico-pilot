import time
import random as rng
import cv2
import netCDF4 as nc
import struct
import array
import math
import numpy as np
from skimage.color import rgb2gray

ZAV=8.0 # 8 Ang/Volt in V

def init_force_map(bias=0.02, level=0.111, zoff=0.0):
	print("Measuring Z at ref")
	# set ref condition
	gxsm.set ("dsp-fbs-bias","0.1") # set Bias to 0.1V
	gxsm.set ("dsp-fbs-mx0-current-set","0.01") # Set Current Setpoint to 0.01nA

	time.sleep(1) # NOW SHOULD ME ON TOP OF MOLECULE
	gxsm.set ("dsp-fbs-bias","0.02") # set Bias to 20mV
	gxsm.set ("dsp-fbs-mx0-current-set","0.05") # Set Current Setpoint to 50pA
	# read Z ref and set
	svec=gxsm.rtquery ("z")
	print('RTQ z',  svec[0]*ZAV)
	z=svec[0]*ZAV
	for i in range(0,10):
		svec=gxsm.rtquery ("z")
		time.sleep(0.05)
		z=z+svec[0]*ZAV
	z=z/11 + zoff  # zoff=0 for auto

	print("Setting Z-Pos/Setpoint = {:8.2f} A".format( z))
	gxsm.set ("dsp-adv-dsp-zpos-ref", "{:8.2f}".format( z))
	gxsm.set ("dsp-fbs-bias","%f" %bias)
	gxsm.set ("dsp-adv-scan-fast-return","5")
	gxsm.set ("dsp-fbs-scan-speed-scan","8")
	gxsm.set ("dsp-fbs-ci","3")
	gxsm.set ("dsp-fbs-cp","0")
	levelreg = level*0.99
	gxsm.set ("dsp-fbs-mx0-current-level","%f"%level)
	gxsm.set ("dsp-fbs-mx0-current-set","%f"%levelreg)
	gxsm.set ("dsp-fbs-bias","%f" %bias)

def exit_force_map(bias=0.2, current=0.02):
	gxsm.set ("dsp-adv-scan-fast-return","1")
	gxsm.set ("dsp-fbs-mx0-current-set","%f"%current)
	gxsm.set ("dsp-fbs-mx0-current-level","0.00")
	gxsm.set ("dsp-fbs-ci","35")
	gxsm.set ("dsp-fbs-cp","40")
	gxsm.set ("dsp-fbs-scan-speed-scan","250")
	gxsm.set ("dsp-fbs-bias","%f" %bias)

def locate_molecule(image,thresh_val):
	data = image
	ds = nc.Dataset(data)
	img = ds['FloatField'][0][0][:][:]
	img = img * ds['dz'][0]
	norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	gray_img = rgb2gray(norm_img)
	max_thresh = 255
	thresh = thresh_val
	def thresh_callback(val):
    		threshold = val
    		canny_output = cv2.Canny(gray_img, threshold, threshold*2)
    		contours,_ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    		contours_poly = [None]*len(contours)
    		boundRect = [None]*len(contours)
    		centers = [None]*len(contours)
    		radius = [None]*len(contours)
    		x_y_coord = []
    		for i, c in enumerate(contours):
    			contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    			boundRect[i] = cv2.boundingRect(contours_poly[i])
    			centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    			if radius[i] < 3 and radius[i] > 1:
    				x_y_coord.append(centers[i])
    			
    		return (x_y_coord)
	x_y_molecules = thresh_callback(thresh)
	return (x_y_molecules)
    	

		

## SIMULATOR ###

## Initial Set Points ## 
gxsm.set('RangeX', '100')
gxsm.set('RangeY', '100')
gxsm.set('PointsX', '400')
gxsm.set('PointsY', '400')

## Load the Image you wanna load. Make sure you change the path for the correct image!! ####

basefile = '/home/steven/Nextcloud/Data/Percy-P0/Exxon-Yunlong/20220401-Cu111-PP-TS-prev-warmup65C/Cu111-PP-TS360-M-Xp-Topo.nc'
print('Loading World to SIM WORLD CHANNEL 11')
gxsm.load(10, basefile)
time.sleep(1)

##### Locate Molecules using OpenCV Some functions might be extra could be use for later...###
print('Finding Molecules')
molecule_coord = locate_molecule(basefile, 115)
time.sleep(1)
print('Found Molecuesl:',len(molecule_coord))

##### Clean Up the channel being used################
print('Removing all Rectangles!')
r=gxsm.marker_getobject_action(10, 'Rectangle','REMOVE-ALL')
print(r)
time.sleep(1)

print('Cleanup Points')
r=gxsm.marker_getobject_action(10, 'Point','REMOVE-ALL')
print(r)
r=gxsm.marker_getobject_action(10, '*Marker','REMOVE-ALL')
print(r)
time.sleep(1)

#### Here add for loop to add boxes and labels to each molecule ######
print('Marking Molecules')

for i in range(len(molecule_coord)):	
	gxsm.add_marker_object(10, 'PointM0'+str(i),1, int(molecule_coord[i][0]),int(molecule_coord[i][1]), 1.0)
#gxsm.add_marker_object(10, 'PointM02', 1,  194, 186, 1.0)
#gxsm.add_marker_object(10, 'PointM03', 1,  217, 116, 1.0)
time.sleep(1)


print('List Objects, Mark Mol, Setup Rects')
k=0
for i in range(len(molecule_coord)):
	o=gxsm.get_object (10, i+k) ## adjust for inserted object -- always pre pended to list!
	print('O', i, ' => ', o)
	if o == 'None':
		break
	print('Marking M', i)
	r=gxsm.add_marker_object(10, 'RectangleM{:02d}'.format(i), 0xff00fff0, round(o[1]),round(o[2]),50)
	k=k+1 # we have not one more object prepended to the object list!
	print(r)

time.sleep(1)



# make sure STM safe mode
exit_force_map(0.1, current=0.003)

for i in range(len(molecule_coord)):
	time.sleep(1)
	print('selecting M',i)
	r=gxsm.marker_getobject_action(10, 'RectangleM{:02d}'.format(i),'GET-COORDS')
	print(r)
	if r != 'OK':
		break
	gxsm.set('PointsX', '330') # readjust points
	gxsm.set('PointsY', '330')

	print('STM: Scanning M',i)
	gxsm.set ("RangeX","50")
	gxsm.set ("RangeY","50")
	gxsm.startscan()
	time.sleep(1)
	gxsm.set ("dsp-fbs-scan-speed-scan","150")
	print('waiting....')
	print(gxsm.waitscan(True))
	print('completed.')

	print('moving tip on top of molecule to measure Z at ref conditons for HR-AFM')
	x=y=50./2.
	gxsm.moveto_scan_xy(x,y)
	print('waiting')
	time.sleep(2)

	print('HR-AFM transitioning...')
	init_force_map(0.02, level=0.08, zoff = 1.0)
	print('HR-AFM: Scanning M',i)
	gxsm.startscan()
	time.sleep(1)
	print('waiting....')
	print(gxsm.waitscan(True))
	print('completed.')

	exit_force_map(0.1, current=0.003)

######## NOTES TO WORK ON ########
#Some overlap with Finding molecules write another if statement if molecules are overlapping
#Add function to get rid of nearest neighbors.
#Make sure tip stays on molecule. Not sure if it. Simulator moving a little fast. 
#The boxes have to be away from the edges!!! Or else they behave poorly. and give a weird rectangle shape. 
#Also if the initial file has an offset the next scan is going to be in the wrong place 






