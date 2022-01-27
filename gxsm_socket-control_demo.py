#!/usr/bin/env python3

## GXSM NetCDF San Image Viwer and Tag Tool with GXSM remote inspection and control socket client functionality

import sys
import os		# use os because python IO is bugy
import time
import threading

from netCDF4 import Dataset, chartostring, stringtochar, stringtoarr
import struct
import array
import math

import numpy as np

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, Gio, Gtk, Gdk, GObject, GdkPixbuf
#import cairo

from netCDF4 import Dataset, chartostring, stringtochar, stringtoarr
import struct
import array
import math

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.backends.backend_gtk3agg import (FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar
from matplotlib.patches import Rectangle
from matplotlib import cm

from PIL import Image


from gxsm_socket_client import *  # Socket Client -- bridge to Gxsm when running embedded socket-server.py in python console

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server




        
###################################################################
## from gxsm xsm.C / gxsm units mechanism "id" :
## unit symbol, ascii simple, scaling to base, formatting options
###################################################################
gxsm_units = {
        "AA": [ u"\u00C5",   "Ang",    1e0, ".1f", ".3f" ],
        "nm": [ "nm",  "nm",     10e0, ".1f", ".3f" ],
        "um": [ u"\u00B5m",  "um",     10e3, ".1f", ".3f" ],
        "mm": [ "mm",  "mm",     10e6, ".1f", ".3f" ],
        "BZ": [ "%BZ", "BZ",     1e0, ".1f", ".2f" ],
        "sec": ["\"",  "\"",      1e0, ".1f", ".2f" ],
        "V": [  "V",   "V",       1e0, ".2f", ".3f" ],
        "mV": [ "mV",  "mV",      1e-3, ".2f", ".3f" ],
        "V": [  "*V", "V",      1e0, ".2f", ".3f" ],
        "*dV": [ "*dV","dV",     1e0, ".2f", ".3f" ],
        "*ddV": [ "*ddV","ddV",  1e0, ".2f", ".3f" ],
        "*V2": [ "V2", "V2",       1e0, ".2f", ".3f" ],
        "1": [  " ",   " ",       1e0, ".3f", ".4f" ],
        "0": [  " ",   " ",       1e0, ".3f", ".4f" ],
        "B": [  "Bool",   "Bool", 1e0, ".3f", ".4f" ],
        "X": [  "X",   "X",       1e0, ".3f", ".4f" ],
        "xV": [  "xV",   "xV",    1e0, ".3f", ".4f" ],
        "deg": [ u"\u00B0", "deg",       1e0, ".3f", ".4f" ],
        "Amp": [ "A",  "A",       1e9, "g", "g" ],
        "nA": [ "nA",  "nA",      1e0, ".2f", ".3f" ],
        "pA": [ "pA",  "pA",      1e-3, ".1f", ".2f" ],
        "nN": [ "nN",  "nN",      1e0, ".2f", ".3f" ],
        "Hz": [ "Hz",  "Hz",      1e0, ".2f", ".3f" ],
        "mHz": [ "mHz",  "mHz",   1e-3, ".2f", ".3f" ],
        "K": [  "K",   "K",       1e0, ".2f", ".3f" ],
        "amu": ["amu", "amu",     1e0, ".1f", ".2f" ],
        "CPS": ["Cps", "Cps",     1e0, ".1f", ".2f" ],
        "CNT": ["CNT", "CNT",     1e0, ".1f", ".2f" ],
        "Int": ["Int", "Int",     1e0, ".1f", ".2f" ],
        "A/s": ["A/s", "A/s",     1e0, ".2f", ".3f" ],
        "s": ["s", "s",           1e0, ".2f", ".3f" ],
        "ms": ["ms", "ms",        1e0, ".2f", ".3f" ],
}




#### Application stuff

APP_TITLE="GXSM Socket Client Demo and NetCDF data view"

MENU_XML="""
<?xml version="1.0" encoding="UTF-8"?>
<interface>
  <menu id="app-menu">
    <section>
      <item>
        <attribute name="action">app.connect</attribute>
        <attribute name="label" translatable="yes">Connect Gxsm</attribute>
      </item>
      <item>
        <attribute name="action">app.open</attribute>
        <attribute name="label" translatable="yes">Open NetCDF</attribute>
      </item>
    </section>
  </menu>
  <menu id="app-menubar">
    <submenu id="app-file-menu">
      <attribute name="label" translatable="yes">_File</attribute>
      <section>
	<attribute name="id">file-section</attribute>
        <item>
          <attribute name="action">app.connect</attribute>
          <attribute name="label" translatable="yes">Connect Gxsm</attribute>
        </item>
        <item>
          <attribute name="action">app.open</attribute>
          <attribute name="label" translatable="yes">Open NetCDF</attribute>
        </item>
      </section>
      <section>
        <item>
          <attribute name="action">app.about</attribute>
          <attribute name="label" translatable="yes">_About</attribute>
        </item>
        <item>
          <attribute name="action">app.quit</attribute>
          <attribute name="label" translatable="yes">_Quit</attribute>
          <attribute name="accel">&lt;Primary&gt;q</attribute>
        </item>
      </section>
    </submenu>
  </menu>
</interface>
"""


class AppWindow(Gtk.ApplicationWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gxsm = None
        self.cdf_image_data_filename = ''
        self.connect("destroy", self.quit)
        self.rootgrp = None
        self.tag_labels = []
        self.NCFile_tags = { 'RegionTags': [], 'AITags': [], 'TrainingTags': [], 'ValidationTags': [] }
        self.click_action = 'Tagging'
        
        grid = Gtk.Grid()
        self.grid=grid
        self.add(grid)

        ## FILE SELECTION
        
        y=0
        button1 = Gtk.Button(label="Open NetCDF")
        button1.connect("clicked", self.on_file_clicked)
        grid.attach(button1, 0, y, 2, 1)
        y=y+1

        ## IMAGEPARAMETERS
        l = Gtk.Label(label="Image Parameters:")
        grid.attach(l, 0, y, 2, 1)
        y=y+1
        l = Gtk.Label(label="Bias")
        grid.attach(l, 0, y, 1, 1)
        self.bias = Gtk.Entry()
        grid.attach(self.bias, 1, y, 1, 1)
        y=y+1
        l = Gtk.Label(label="Current")
        grid.attach(l, 0, y, 1, 1)
        self.current = Gtk.Entry()
        grid.attach(self.current, 1, y, 1, 1)
        y=y+1
        l = Gtk.Label(label="Nx")
        grid.attach(l, 0, y, 1, 1)
        self.Nx = Gtk.Entry()
        grid.attach(self.Nx, 1, y, 1, 1)
        y=y+1
        l = Gtk.Label(label="Ny")
        grid.attach(l, 0, y, 1, 1)
        self.Ny = Gtk.Entry()
        grid.attach(self.Ny, 1, y, 1, 1)
        y=y+1
        l = Gtk.Label(label="XRange")
        grid.attach(l, 0, y, 1, 1)
        self.Xrange = Gtk.Entry()
        grid.attach(self.Xrange, 1, y, 1, 1)
        y=y+1
        l = Gtk.Label(label="YRange")
        grid.attach(l, 0, y, 1, 1)
        self.Yrange = Gtk.Entry()
        grid.attach(self.Yrange, 1, y, 1, 1)
        y=y+1
        l = Gtk.Label(label="dx")
        grid.attach(l, 0, y, 1, 1)
        self.dX = Gtk.Entry()
        grid.attach(self.dX, 1, y, 1, 1)
        y=y+1
        l = Gtk.Label(label="dy")
        grid.attach(l, 0, y, 1, 1)
        self.dY = Gtk.Entry()
        grid.attach(self.dY, 1, y, 1, 1)
        y=y+1
        l = Gtk.Label(label="dz")
        grid.attach(l, 0, y, 1, 1)
        self.dZ = Gtk.Entry()
        grid.attach(self.dZ, 1, y, 1, 1)
        y=y+1


        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.axy = self.fig.add_subplot(111)
        self.axline = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)  # a Gtk.DrawingArea
        self.canvas.set_size_request(600, 600)
        self.grid.attach(self.canvas, 3,1, 100,100)
        self.cbar = None
        self.im   = None
        self.colormap = cm.RdYlGn
        #self.colormap = cm.Greys  #magma
        #cmaps['Sequential'] = [
        #    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        #    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        #    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        #cmaps['Perceptually Uniform Sequential'] = [
        #    'viridis', 'plasma', 'inferno', 'magma', 'cividis']

        #cmaps['Sequential (2)'] = [
        #    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        #    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        #    'hot', 'afmhot', 'gist_heat', 'copper']

        # Create toolbar
        self.toolbar = NavigationToolbar (self.canvas, self)
        self.grid.attach(self.toolbar, 2,0,100,1)
        
        c=110
        y=1
        buttonG1 = Gtk.Button(label="GXSM ! Live")
        buttonG1.connect("clicked", self.live_clicked)
        grid.attach(buttonG1, c, y, 2, 1)
        y=y+1
        buttonG2 = Gtk.Button(label="GXSM ! Start")
        buttonG2.connect("clicked", self.start_clicked)
        grid.attach(buttonG2, c, y, 2, 1)
        y=y+1
        buttonG3 = Gtk.Button(label="GXSM ! Stop")
        buttonG3.connect("clicked", self.stop_clicked)
        grid.attach(buttonG3, c, y, 2, 1)
        y=y+1
        buttonG4 = Gtk.Button(label="GXSM ! Auto Save")
        buttonG4.connect("clicked", self.autosave_clicked)
        grid.attach(buttonG4, c, y, 2, 1)
        y=y+1
        buttonG5 = Gtk.Button(label="GXSM ! Auto Update")
        buttonG5.connect("clicked", self.autoupdate_clicked)
        grid.attach(buttonG5, c, y, 2, 1)
        y=y+1

        self.show_all()

    def quit(self, button):
        if self.rootgrp:
            self.rootgrp.close()
        return
        
    def add_filters(self, dialog):
        filter_py = Gtk.FileFilter()
        filter_py.set_name("Unidata NetCDF (GXSM)")
        filter_py.add_mime_type("application/x-netcdf")
        dialog.add_filter(filter_py)

        filter_any = Gtk.FileFilter()
        filter_any.set_name("Any files")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)
       
    def on_file_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
			   Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        #dialog.set_transient_for(self.main_widget)
        self.add_filters(dialog)
        #dialog.modal = True
        response = dialog.run()
        try:
            if response == Gtk.ResponseType.OK:
                print("Open clicked")
                print("File selected: " + dialog.get_filename())
                self.cdf_image_data_filename = dialog.get_filename()
                self.load_CDF ()
        finally:
            dialog.destroy()

        
        
    def load_CDF(self):

        # (re) load NetCDF file
        print("NetCDF File: ", self.cdf_image_data_filename)
        if self.rootgrp:
            self.rootgrp.close()
        self.tag_labels = []

        # connect to NetCDF file
        self.rootgrp = Dataset (self.cdf_image_data_filename, "r+")
        print(self.rootgrp['FloatField'])

        # copy image data in "FloatField" variable from NetCDF file
        self.ImageData = self.rootgrp['FloatField'][0][0][:][:]
        print (self.ImageData)

        # get ranges
        self.Xlookup = self.rootgrp['dimx'][:]
        self.Ylookup = self.rootgrp['dimy'][:]

        # get bias value
        if 'sranger_mk2_hwi_bias' in self.rootgrp.variables:
            print('Bias    = ', self.rootgrp['sranger_mk2_hwi_bias'][0], self.rootgrp['sranger_mk2_hwi_bias'].var_unit)
            self.ub = self.rootgrp['sranger_mk2_hwi_bias'][0]
            self.bias.set_text ('{0:.4f} '.format(self.rootgrp['sranger_mk2_hwi_bias'][0])+self.rootgrp['sranger_mk2_hwi_bias'].var_unit)
        else:
            self.bias.set_text ('??')

        # get current setpoint
        if 'sranger_mk2_hwi_mix0_current_set_point' in self.rootgrp.variables:
            print('Current (old name) = ', self.rootgrp['sranger_mk2_hwi_mix0_current_set_point'][0], self.rootgrp['sranger_mk2_hwi_mix0_current_set_point'].var_unit)
            self.cur = self.rootgrp['sranger_mk2_hwi_mix0_current_set_point'][0]
            self.current.set_text ('{0:.4f} '.format(self.rootgrp['sranger_mk2_hwi_mix0_current_set_point'][0])+self.rootgrp['sranger_mk2_hwi_mix0_current_set_point'].var_unit)
        else:
            self.current.set_text ('??')

        # ... many more as needed

        # want at least need geometry data
        print('X Range = ', self.rootgrp['rangex'][0], self.rootgrp['rangex'].var_unit)
        self.xr = self.rootgrp['rangex'][0]
        self.Xrange.set_text ('{0:.1f} '.format(self.rootgrp['rangex'][0])+self.rootgrp['rangex'].var_unit)
        
        print('Y Range = ', self.rootgrp['rangey'][0], self.rootgrp['rangey'].var_unit)
        self.yr = self.rootgrp['rangey'][0]
        self.Yrange.set_text ('{0:.1f} '.format(self.rootgrp['rangey'][0])+self.rootgrp['rangey'].var_unit)

        print('Nx      = ', len(self.rootgrp['dimx']))
        self.nx = len(self.rootgrp['dimx'])
        self.Nx.set_text ('{} px'.format(self.nx))
        print('Ny      = ', len(self.rootgrp['dimy']))
        self.ny = len(self.rootgrp['dimy'])
        self.Ny.set_text ('{} px'.format(self.ny))


        print('dx      = ', self.rootgrp['dx'][0], self.rootgrp['dx'].var_unit)
        self.dx = self.rootgrp['dx'][0]
        self.dX.set_text ('{0:.4f} '.format(self.rootgrp['dx'][0])+gxsm_units[self.rootgrp['dx'].unit][0])
        print('dy      = ', self.rootgrp['dy'][0], gxsm_units[self.rootgrp['dy'].unit][0])
        self.dy = self.rootgrp['dy'][0]
        self.dY.set_text ('{0:.4f} '.format(self.rootgrp['dy'][0])+gxsm_units[self.rootgrp['dy'].unit][0])
        print('dz      = ', self.rootgrp['dz'][0], gxsm_units[self.rootgrp['dz'].unit][0])
        self.dz = self.rootgrp['dz'][0]
        self.dZ.set_text ('{0:.4f} '.format(self.rootgrp['dz'][0])+gxsm_units[self.rootgrp['dz'].unit][0])
        
        # scale image data to Angstroems!
        self.ImageData = self.ImageData * self.dz # scale to unit

        # create/update figure and image
        self.update_image(self.ImageData.max(), self.ImageData.min())
        self.axy.set_title('...'+self.cdf_image_data_filename[-40:]) # -45 for 800px
        self.axy.set_xlabel('X in '+gxsm_units[self.rootgrp['dx'].unit][0])
        self.axy.set_ylabel('Y in '+gxsm_units[self.rootgrp['dy'].unit][0])
        if self.cbar:
            self.cbar.remove () #fig.delaxes(self.figure.axes[1])
        self.cbar = self.fig.colorbar(self.im, extend='both', shrink=0.9, ax=self.axy)
        self.cbar.set_label('Z in '+gxsm_units[self.rootgrp['dz'].unit][0])
        self.fig.canvas.draw()
        print ('NCLoad tags from group {}'.format (self.tact_label_group))
        self.load_tags_from_netcdf (self.tact_label_group)



    def update_image(self, vmax, vmin):
    
        self.im=self.axy.imshow(self.ImageData, interpolation='bilinear', cmap=self.colormap,
                           origin='upper', extent=[0, self.xr, 0, self.yr],
                           vmax=vmax, vmin=vmin)
        if self.cbar:
            self.cbar.remove () #fig.delaxes(self.figure.axes[1])
        self.cbar = self.fig.colorbar(self.im, extend='both', shrink=0.9, ax=self.axy)
        self.cbar.set_label('Z in '+gxsm_units[self.rootgrp['dz'].unit][0])
        self.fig.canvas.draw()

                
            

    def live_clicked(self, widget):
        if self.gxsm == None:
            self.gxsm = SocketClient(HOST, PORT)
        self.gxsm.request_autoupdate()
        JSONfn = self.gxsm.request_query_info_args('chfname', 0)
        ##{"result": [{"query": "chfname", "value": "/mnt/XX/BNLBox2T/Ilya-P38804/20191106-Bi2Se3/Bi2Se3-C60-322-M-Xp-Topo.nc"}]}
        fn = JSONfn['result'][0]['value']
        if os.path.isfile(fn):
            self.cdf_image_data_filename = fn
            self.load_CDF ()
        JSONfn = self.gxsm.request_query_info_args('y_current', 0)
        yi = JSONfn['result'][0]['value']
        print ("Current CH[0]: {}  AT Y-index={}".format(fn, yi))
        
    def start_clicked(self, widget):
        if self.gxsm == None:
            self.gxsm = SocketClient(HOST, PORT)
        ret = self.gxsm.request_start_scan()
        print(ret)
        
    def stop_clicked(self, widget):
        if self.gxsm == None:
            self.gxsm = SocketClient(HOST, PORT)
        ret = self.gxsm.request_stop_scan()
        print(ret)
        
    def autosave_clicked(self, widget):
        if self.gxsm == None:
            self.gxsm = SocketClient(HOST, PORT)
        ret=self.gxsm.request_autosave()
        print(ret)
        
    def autoupdate_clicked(self, widget):
        if self.gxsm == None:
            self.gxsm = SocketClient(HOST, PORT)
        ret=self.gxsm.request_autoupdate()
        print(ret)


############################################################
# G Application Core
############################################################

            
class Application(Gtk.Application):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, application_id="org.gnome.gxsm-sok-ai-app",
                         flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE,
                         **kwargs)
        self.window = None

        self.add_main_option("test", ord("t"), GLib.OptionFlags.NONE,
                             GLib.OptionArg.NONE, "Command line test", None)

    def do_startup(self):
        Gtk.Application.do_startup(self)

        ## File Menu Actions
        
        action = Gio.SimpleAction.new("connect", None)
        action.connect("activate", self.on_connect_gxsm)
        self.add_action(action)

        action = Gio.SimpleAction.new("open", None)
        action.connect("activate", self.on_open_netcdf)
        self.add_action(action)

        action = Gio.SimpleAction.new("folder", None)
        action.connect("activate", self.on_set_folder)
        self.add_action(action)

        action = Gio.SimpleAction.new("statistics", None)
        action.connect("activate", self.on_folder_statistics)
        self.add_action(action)

        action = Gio.SimpleAction.new("list", None)
        action.connect("activate", self.on_list_netcdf)
        self.add_action(action)

        action = Gio.SimpleAction.new("about", None)
        action.connect("activate", self.on_about)
        self.add_action(action)

        action = Gio.SimpleAction.new("quit", None)
        action.connect("activate", self.on_quit)
        self.add_action(action)
        
        builder = Gtk.Builder.new_from_string(MENU_XML, -1)
        self.set_app_menu(builder.get_object("app-menu"))
        self.set_menubar(builder.get_object("app-menubar"))
        
    def do_activate(self):
        # We only allow a single window and raise any existing ones
        if not self.window:
            # Windows are associated with the application
            # when the last one is closed the application shuts down
            self.window = AppWindow(application=self, title=APP_TITLE)

        self.window.present()

    def do_command_line(self, command_line):
        options = command_line.get_options_dict()
        # convert GVariantDict -> GVariant -> dict
        options = options.end().unpack()

        if "test" in options:
            # This is printed on the main instance
            print("Test argument recieved: %s" % options["test"])

        self.activate()
        return 0



    def on_connect_gxsm(self, action, param):
        self.window.connect_clicked(None)
        
    def on_open_netcdf(self, action, param):
        self.window.on_file_clicked(None)

    def on_set_folder(self, action, param):
        self.window.on_folder_clicked(None)

    def on_folder_statistics(self, action, param):
        self.window.on_folder_statistics()

    def on_list_netcdf(self, action, param):
        self.window.list_CDF_clicked(None)
    
    def on_about(self, action, param):
        about_dialog = Gtk.AboutDialog(transient_for=self.window, modal=True,
                                       program_name="gxsm-scan-image-viewer.py: Gxsm3 Remote AI Client Tag Tool",
                                       authors=["Percy Zahl"],
                                       #documenters = ["--"],
                                       copyright="GPL",
                                       website="http://www.gxsm.sf.net",
                                       logo=GdkPixbuf.Pixbuf.new_from_file_at_scale("./gxsm3-icon.svg", 200,200, True),
                                       )
        about_dialog.set_copyright(
            "Copyright \xc2\xa9 2019 Percy Zahl.\n"
            "This is a experimental GXSM3 Remote Control Client,\n"
            "a Gxsm Scan Data NetCDF4 Image Viewer\n"
            "and Region Tagging/Image Categorization Tool\n"
            "...with future explorative AI Client functionality."
        )
        about_dialog.present()

    def on_quit(self, action, param):
        self.quit()


        
if __name__ == "__main__":
    app = Application()
    app.run(sys.argv)
