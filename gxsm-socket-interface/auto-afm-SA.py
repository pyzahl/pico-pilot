######## Before you run change file path. If not enough molecules found Lower threshold

######## NOTES TO WORK ON ########


#Tip goes to the center of the brightest spot and makes a box around it. 
#The boxes have to be away from the edges!!! Or else they behave poorly. and give a weird rectangle shape. Anything around the edges does not image the molecule!! 

##################  Imports ############################

from string import *
import os

import time
import random as rng
import cv2
#import netCDF4 as nc
import struct
import array
import math
import numpy as np
from skimage.color import rgb2gray
import itertools
import tensorflow as tf
from tensorflow import keras


###############################################
## INSTRUMENT
ZAV=8.0 # 8 Ang/Volt in V

### SETUP
map_ch=6   # MAP IMAGE CHANNEL  (0....N)
map_diffs=gxsm.get_differentials(map_ch)

# Up to 8 ScriptControls (sc)
# Level, I in pA
sc = dict(STM_Range=70, AFM_Range=45,  Molecule=1,  I_ref=30, CZ_Level=80.0,  Z_down=0.3, Z_start=0.0, Tip_Z=0.0,  Tip_Z_Ref=0.0, AutoAFMSpeed=1)

#sc['STM_Range']

STM_points = 160
STM_dx=sc['STM_Range']/STM_points

AFM_points = 330
AFM_dx=sc['AFM_Range']/AFM_points

#CZAFM_Iref=0.04
#CZAFM_Zoff=-0.1
#CZ_FUZZY_level =0.1

do_auto_locate = False
do_stm = True
do_afm = True


# Setup SCs
def SetupSC():
	for i, e in enumerate(sc.items()):
		id='py-sc{:02d}'.format(i+1)
		print (id, e[0], e[1])
		gxsm.set_sc_label(id, e[0])
		gxsm.set(id, '{:.4f}'.format(e[1]))

SetupSC()

# Read / Update dict
def GetSC():
	for i, e in enumerate(sc.items()):
		id='py-sc{:02d}'.format(i+1)
		print (id, ' => ', e[0], e[1])
		sc[e[0]] = float(gxsm.get(id))
		print (id, '<=', sc[e[0]])

# Update SCs
def SetSC():
	for i, e in enumerate(sc.items()):
		id='py-sc{:02d}'.format(i+1)
		gxsm.set(id, '{:.4f}'.format(e[1]))


#GetSC()
#SetSC()

gxsm.set('script-control','1')

def export_drawing(ch=0, postfix='-dwg'):

	full_original_name = gxsm.chfname(ch).split()[0]

	print(full_original_name)
	folder = os.path.dirname(full_original_name)
	#print(folder)

	ncfname = os.path.basename(full_original_name)
	#print(ncfname)

	name, ext = os.path.splitext(ncfname)
	#print(name, '  +  ', ext)

	dest_name = folder+'/'+name+postfix
	print(dest_name)

	gxsm.chmodea(ch)
	gxsm.autodisplay()
	time.sleep(1)
	gxsm.save_drawing(ch, 0,0, dest_name+'.png')
	gxsm.save_drawing(ch, 0,0, dest_name+'.pdf')



def init_force_map_ref_xy(bias=0.02, level=0.111, ref_i=0.05, zoff=0.0, xy_list=[[0,0]]):
	print("Measuring Z at ref")
	# set ref condition
	gxsm.set ("dsp-fbs-bias","0.1") # set Bias to 0.1V
	gxsm.set ("dsp-fbs-mx0-current-set","{:8.4f}".format( ref_i))   # Set Current Setpoint to reference value (nA)
	gxsm.set ("dsp-fbs-mx0-current-level","0.00")

	time.sleep(1) # NOW SHOULD ME ON TOP OF MOLECULE
	gxsm.set ("dsp-fbs-bias","0.02") # set Bias to 20mV
	gxsm.set ("dsp-fbs-mx0-current-set","0.05") # Set Current Setpoint to 50pA
	# read Z ref and set
	svec=gxsm.rtquery ("z")
	print('RTQ z',  svec[0]*ZAV)
	pts=1
	z=svec[0]*ZAV
	zmin=zmax=z
	for r in xy_list:
		gxsm.moveto_scan_xy(r[0], r[1])
		time.sleep(0.1)
		for i in range(0,5):
			svec=gxsm.rtquery ("z")
			time.sleep(0.02)
			zxy=svec[0]*ZAV
			if zmin > zxy:
				zmin=zxy
			if zmax < zxy:
				zmax=zxy
			print(r, " => Z: ", zxy, " Min/Max: ", zmin, zmax)
			z=z+zxy
			pts=pts+1
	z=z/pts + zoff  # zoff=0 for auto
	time.sleep(1) # NOW SHOULD ME ON TOP OF MOLECULE

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
	return z

def exit_force_map(bias=0.2, current=0.02):
	gxsm.set ("dsp-adv-scan-fast-return","1")
	gxsm.set ("dsp-fbs-mx0-current-set","%f"%current)
	gxsm.set ("dsp-fbs-mx0-current-level","0.00")
	gxsm.set ("dsp-fbs-ci","35")
	gxsm.set ("dsp-fbs-cp","40")
	gxsm.set ("dsp-fbs-scan-speed-scan","250")
	gxsm.set ("dsp-fbs-bias","%f" %bias)
	
def process(input_list, threshold=20):
    combos = itertools.combinations(input_list, 2)
    points_to_remove = [point2 for point1, point2 in combos if math.dist(point1, point2)<=threshold]
    points_to_keep = [point for point in input_list if point not in points_to_remove]
    return points_to_keep

def auto_afm_scanspeed(y):
	ms = gxsm.get_slice(2, 0,0, y,1) # ch, v, t, yi, yn   ## AFM dFreq in CH3
	med = np.median(ms)
	dFspan = np.max(ms) - np.min(ms)
	if dFspan > 1.0:
		gxsm.set ("dsp-adv-scan-fast-return","5")
		time.sleep(1)
		gxsm.set ("dsp-fbs-scan-speed-scan","8")
	elif dFspan > 0.8:
		gxsm.set ("dsp-adv-scan-fast-return","5")
		time.sleep(1)
		gxsm.set ("dsp-fbs-scan-speed-scan","10")
	elif dFspan > 0.5:
		gxsm.set ("dsp-adv-scan-fast-return","5")
		time.sleep(1)
		gxsm.set ("dsp-fbs-scan-speed-scan","15")
	elif dFspan > 0.4:
		gxsm.set ("dsp-adv-scan-fast-return","2")
		time.sleep(1)
		gxsm.set ("dsp-fbs-scan-speed-scan","20")
	elif dFspan > 0.3:
		gxsm.set ("dsp-adv-scan-fast-return","2")
		time.sleep(1)
		gxsm.set ("dsp-fbs-scan-speed-scan","30")
	else:
		gxsm.set ("dsp-adv-scan-fast-return","1")
		time.sleep(1)
		gxsm.set ("dsp-fbs-scan-speed-scan","50")
	print('Median: ', np.median(ms))
	print('Min: ', np.min(ms))
	print('Max: ', np.max(ms))
	print('Range: ', np.max(ms) - np.min(ms))


def get_gxsm_img_bypkt(ch):
	# fetch dimensions
	dims=gxsm.get_dimensions(ch)
	print (dims)
	geo=gxsm.get_geometry(ch)
	print (geo)
	diffs=gxsm.get_differentials(ch)
	print (diffs)
	m = np.zeros((dims[1],dims[0]), dtype=float)
	for y in range (0,dims[1]):
		for x in range (0, dims[0]):
			v=0
			m[y][x]=gxsm.get_data_pkt (ch, x, y, v, 0)*diffs[2]  # Z value in Ang now
	return m


def get_gxsm_img(ch):
	dims=gxsm.get_dimensions(ch)
	return gxsm.get_slice(ch, 0,0, 0,dims[1]) # ch, v, t, yi, yn


def get_gxsm_img_cm(ch):
	# fetch dimensions
	dims=gxsm.get_dimensions(ch)
	print (dims)
	geo=gxsm.get_geometry(ch)
	print (geo)
	diffs=gxsm.get_differentials(ch)
	print (diffs)
	m = np.zeros((dims[1],dims[0]), dtype=float)

	for y in range (0,dims[1]):
		for x in range (0, dims[0]):
			v=0
			m[y][x]=gxsm.get_data_pkt (ch, x, y, v, 0)*diffs[2] # Z value in Ang now

	cmx = 0
	cmy = 0
	csum = 0
	cmed = np.median(m)
	print ('Z base: ', cmed)
	b=2
	for y in range (b,dims[1]-b):
		for x in range (b, dims[0]-b):
			v=0
			m[y][x]=m[y][x] - cmed # Z value in Ang now
			if m[y][x] > 0.5:
				cmx = cmx+x*m[y][x]
				cmy = cmy+y*m[y][x]
				csum = csum + m[y][x]
	if csum > 0:
		cmx = cmx/csum
		cmy = cmy/csum
	else:
		cmx = dims[0]/2
		cmy = dims[1]/2
	gxsm.add_marker_object(ch, 'PointCM',1, int(round(cmx)), int(round(cmy)), 1.0)
	export_drawing(ch, '-CM')
	return m, cmx, cmy
	
def prepare(ch):
	IMG_SIZE = 256
	img_array = get_gxsm_img(ch)
	img_array = img_array/255.0
	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
	new_array = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
	return new_array
	


def locate_molecule_nc(image,thresh_val):
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
    				if centers[i] not in x_y_coord:
    					x_y_coord.append(centers[i])
    			
    		return (x_y_coord)
	x_y_molecules = thresh_callback(thresh)
	return (x_y_molecules)
    	
def locate_molecule_ch(ch,thresh_val):
	img = get_gxsm_img(ch)
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
    				if centers[i] not in x_y_coord:
    					x_y_coord.append(centers[i])
    			
    		return (x_y_coord)
	x_y_molecules = thresh_callback(thresh)
	return (x_y_molecules)



def do_stm_and_lock_on_center(mi):		
	print('STM: Scanning M',mi)
	gxsm.startscan()
	time.sleep(2)
	gxsm.set ("dsp-fbs-scan-speed-scan","225")
	time.sleep(1)
	print('waiting....')

	l=0
	while l >= 0  and int(gxsm.get('script-control')) >0:
		l =gxsm.waitscan(False)
		#print ('Line=',l)
		time.sleep(2)

	gxsm.stopscan()
	time.sleep(2)

	
	print('STM completed. Centering.')

	r=gxsm.marker_getobject_action(0, 'PointCM','REMOVE')
	print(r)
	time.sleep(1)
	m, cx, cy = get_gxsm_img_cm(0)
	time.sleep(1)
	r=gxsm.marker_getobject_action(0, 'PointCM','SET-OFFSET')
	print(r)

	time.sleep(2)
	r=gxsm.marker_getobject_action(0, 'PointCM','REMOVE')
	print(r)


def do_HR_AFM(mi, tipz):
	GetSC()
	z = sc['Tip_Z_Ref']
	sc['Tip_Z'] = -tipz
	sc['Molecule'] = mi
	SetSC()
	gxsm.set ("dsp-adv-dsp-zpos-ref", "{:8.2f}".format( z-tipz))
	print('HR-AFM: Scanning M',mi, ' at Z=', z-tipz)
	gxsm.startscan()
	time.sleep(2)
	print('waiting....')
	gxsm.set ("dsp-fbs-scan-speed-scan", "10")
	time.sleep(1)
	gxsm.set ("dsp-fbs-scan-speed-scan", "8")
	l=0
	lp=1
	#while gxsm.waitscan(False) >= 0 and  int(gxsm.get('script-control')) >0:
	while l >= 0  and int(gxsm.get('script-control')) >0:
		l =gxsm.waitscan(False)
		#print ('Line=',l)
		time.sleep(5)
		GetSC()
		if l > lp and sc['AutoAFMSpeed'] > 0:
			auto_afm_scanspeed(l)
			lp=l+1
	print('HR-AFM completed, saving...')



## SIMULATOR ###
## Load the Image you wanna load. Make sure you change the path for the correct image!! ####

#basefile = '/home/steven/Nextcloud/Data/Percy-P0/Exxon-Yunlong/20220412-Cu111-PP-TI/Cu111-PP-TI016-M-Xp-Topo.nc'
#print('Loading World to SIM WORLD CHANNEL 11')
#gxsm.load(map_ch, basefile)
#time.sleep(1)

print('Map in CH', map_ch+1)

print('Removing all Rectangles!')
r=gxsm.marker_getobject_action(map_ch, 'Rectangle','REMOVE-ALL')
print(r)


if do_auto_locate:
	##### Locate Molecules using OpenCV Some functions might be extra could be use for later...###
	print('Finding Molecules')
	#molecule_coord = locate_molecule_nc(basefile, 65)
	molecule_coord = locate_molecule_ch(map_ch, 65)
	pro_molecule_coord = process(molecule_coord)
	time.sleep(1)
	print('Found Molecuesl:',len(pro_molecule_coord))

	##### Clean Up the channel being used################
	print('Removing all Rectangles!')
	r=gxsm.marker_getobject_action(map_ch, 'Rectangle','REMOVE-ALL')
	print(r)
	time.sleep(1)

	print('Cleanup Points')
	r=gxsm.marker_getobject_action(map_ch, 'Point','REMOVE-ALL')
	print(r)
	r=gxsm.marker_getobject_action(map_ch, '*Marker','REMOVE-ALL')
	print(r)
	time.sleep(1)

	#### Here add for loop to add boxes and labels to each molecule ######
	print('Marking Molecules at')
	print (pro_molecule_coord)

	for i in range(len(pro_molecule_coord)):	
		gxsm.add_marker_object(map_ch, 'PointM{:02d}'.format(i),1, int(pro_molecule_coord[i][0]),int(pro_molecule_coord[i][1]), 1.0)

	time.sleep(1)

	gxsm.set('script-control','2')
	print('waiting as long as sc>1')
	while int(gxsm.get('script-control')) > 1:
		time.sleep(0.5)

print('List Objects, Mark Mol, Setup Rects')
k=0
for i in range(0, 50): ##len(pro_molecule_coord)):
	o=gxsm.get_object (map_ch, i+k) ## adjust for inserted object -- always pre pended to list!
	print('O', i, ' => ', o)
	if o == 'None':
		break
	print('Marking M', i)
	r=gxsm.add_marker_object(map_ch, 'RectangleM{:02d}'.format(i), 0xff00fff0, round(o[1]),round(o[2]), sc['AFM_Range']/map_diffs[0])
	k=k+1 # we have not one more object prepended to the object list!
	print(r)

SetSC()
time.sleep(1)

gxsm.set('script-control','2')
print('waiting as long as sc>1 -- check configurations now')
while int(gxsm.get('script-control')) > 1:
	time.sleep(0.5)

GetSC()

# make sure STM safe mode
exit_force_map(0.1, current=0.006)



gxsm.set('script-control','3')

for mi in range(0,50):  ##len(molecule_coord)):
	sc['Molecule'] = mi
	SetSC()

	if int(gxsm.get('script-control')) < 1:
		break

	time.sleep(1)
	print('selecting M',mi)
	r=gxsm.marker_getobject_action(map_ch, 'RectangleM{:02d}'.format(mi),'GET-COORDS')
	print(r)
	if r != 'OK':
		break

	GetSC()
	STM_points = round(sc['STM_Range']/STM_dx)
	gxsm.set('PointsX', '{}'.format(STM_points)) # readjust points
	gxsm.set('PointsY', '{}'.format(STM_points))

	gxsm.set ('RangeX','{}'.format(sc['STM_Range'])) # Readjust range to make all of them the same size. 
	gxsm.set ('RangeY','{}'.format(sc['STM_Range']))
	time.sleep(1)

	if do_stm:
		do_stm_and_lock_on_center(mi)
		time.sleep(1)


	if int(gxsm.get('script-control')) < 1:
		break

	# Setup AFM Scan
	GetSC()
	AFM_points = round(sc['AFM_Range']/AFM_dx)
	gxsm.set('PointsX', '{}'.format(AFM_points)) # readjust points
	gxsm.set('PointsY', '{}'.format(AFM_points))

	gxsm.set ('RangeX','{}'.format(sc['AFM_Range'])) # Readjust range to make all of them the same size. 
	gxsm.set ('RangeY','{}'.format(sc['AFM_Range']))
	time.sleep(1)

	# only do reconfigure scan geom -- todo: do not save
	gxsm.startscan()
	time.sleep(3)
	gxsm.stopscan()
	# do STM orbital scans +2V/-1.5V or so?

	if do_afm:
		print('moving tip on top of molecule to measure Z at ref conditons for HR-AFM')

		# initial setpoint determinaion on this grid -- make better: assure on molecule!
		ds = 2.0
		c   = 0.0
		ref_xy_list = [ ]
		for i in np.arange(-1,2):
			for j in np.arange(-1,2):
				ref_xy_list.append([c+i*ds, c+j*ds])

		print('HR-AFM transitioning...')
		GetSC()
		z=init_force_map_ref_xy(0.02, level=sc['CZ_Level']*1e-3, ref_i=sc['I_ref']*1e-3, zoff = sc['Z_start'], xy_list=ref_xy_list)
		sc['Tip_Z_Ref'] = z
		SetSC()
		tipz =0
		while int(gxsm.get('script-control')) >1 and tipz <= sc['Z_down'] and tipz < 2.0:
			do_HR_AFM(mi, tipz)
			time.sleep(4)
			CATEGORIES = ["Far", "Good"]
			model = tf.keras.models.load_model('/home/steven/BNL/GUI/pico-pilot/gxsm-socket-interface/GoodvBadv1.h5')
			prediction = model.predict(prepare(map_ch))
			print(CATEGORIES[int(prediction[0][0])])
			if CATEGORIES[int(prediction[0][0])] == "Far":
				tipz = tipz+0.3

			if tipz == 0:
				print("Check")
				# set RectangleID/Label to McBSP Freq file name -- 1st of Z series
				full_original_name = gxsm.chfname(2).split()[0]
				ncfname = os.path.basename(full_original_name)
				bname, ext = os.path.splitext(ncfname)
				print(bname)
				### CUSTOM FOR TOIS FILE NAMEING SCHEME ###
				filenumber = bname[11:14]
				print('File Number: ', filenumber)
				r = gxsm.marker_getobject_action(map_ch, 'RectangleM{:02d}'.format(mi),'SET-LABEL-TO:'+filenumber)
				print(r)

				
			tipz = tipz+0.3			

		if int(gxsm.get('script-control')) >5:
				print('waiting for re run as long as sc>5')
				while int(gxsm.get('script-control')) > 5:
					time.sleep(0.5)

			
		exit_force_map(0.1, current=0.006)





