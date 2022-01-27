#!/usr/bin/env python3

## GXSM NetCDF San Image Viwer and Tag Tool with GXSM remote inspection and control socket client functionality
## https://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu
## nccopy -k nc4 testV3.nc testV4.nc
## F**edUp libs:  percy@ltncafm:~/SVN/Gxsm-3.0-spmc$ exec env LD_LIBRARY_PATH=/home/percy/anaconda3/pkgs/cudatoolkit-10.0.130-0/lib  /usr/bin/python3.7 ./gxsm-scan-image-viewer-sok-client.py 
## https://cs.stanford.edu/people/karpathy/rcnn/
##https://www.kaggle.com/kmader/keras-rcnn-based-overview-wip#Overview



##################################		All the libraries to access			############################################
import sys
import os		# use os because python IO is bugy
import time
import threading


from netCDF4 import Dataset, chartostring, stringtochar, stringtoarr
import struct
import array
import math

import numpy as np





################################################# 	Connecting to Gxsm 		##########################################################################################################
from gxsm_socket_client import *  #Socket Client to connect to Gxsm

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

gxsm =  SocketClient(HOST, PORT) #This is to connect to the software
###################################################################################################################################################################




#gxsm.request_set_parameter("id","value") changes whatever id you want to set. Would ideally want these connected to an entry box on a GUI to set to not manually change everytime. 


#gxsm.request_start_scan() Starts scan. Kinda explains itself. Would like to make it do start what ever task instead of start scan. Like start spectroscopy, start scan, start pulsing (Might be easy, may be more complicated


#gxsm.request_autoupdate() returns what line the scan is currently on. Can I use this to check what action to do next?

##Set Parameters you want. Setting RangeX as place holderfor now

gxsm.request_set_parameter("RangeX","300")

###Start Scan
#gxsm.request_start_scan()

##Save an image ############## Does not work without the update. Can I change the save file path from here as well?
gxsm.request_autoupdate()

rootgrp = Dataset(".nc", "w", format="NETCDF4")

rootgrp.close()

###Load an Image######## The load_CDF is attached to a different class. So I need to add this class to my files I assume (?)
print("NetCDF File: ", test003-M-Xp-Topo.nc)
    
