# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:13:38 2017

@author: Jules
"""

from cx_Freeze import setup, Executable
import os

os.environ['TCL_LIBRARY'] = "D:\\WinPython-64bit-3.5.3.1Qt5\\python-3.5.3.amd64\\tcl\\tcl8.6"
os.environ['TK_LIBRARY'] = "D:\\WinPython-64bit-3.5.3.1Qt5\\python-3.5.3.amd64\\tcl\\tk8.6"

base = None


executables = [Executable("Iguana.py", base=base, icon = "C:\\Users\\Jules\\Desktop\\Iguana_sources\\Iguane.ico")]

packages = ['scipy','PyQt5.QtWidgets','PyQt5.QtGui','PyQt5.QtCore','sys','os','py2cytoscape.data.cyrest_client','psutil','networkx','pandas','numpy','igraph','sklearn','re','xgboost','random']
files = ['optimizationComponent.lp','clingo.exe','interface_ui.py', 'Iguane.ico']
add = ['scipy.sparse.csgraph._shortest_path']

options = {
    'build_exe': {
        'packages':packages,
        "include_files": files,
        'includes':add
    },

}

setup(
    name = "Iguana",
    options = options,
    version = "3.0",
    description = 'Iguana (feat. iggy-POC)',
    executables = executables
)
