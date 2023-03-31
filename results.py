"""
Created on Wen august 22 2022
@author: jca

sliding window: https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
balanced histogram: https://theailearner.com/2019/07/19/balanced-histogram-thresholding/
Crack width initially based on the GithHub repository: https://github-com.translate.goog/Garamda/Concrete_Crack_Detection_and_Analysis_SW?_x_tr_sl=ko&_x_tr_tl=en&_x_tr_hl=es&_x_tr_pto=wapp


"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import Dictionary as dict
from skimage.util import invert
from numpy import empty
import time
import glob

start_time = time.time()

path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'


pixel_width=0.08
winW = 28
winH = 28
method_threshold=2.5    # Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
save_info=True
crack=0

# region crack info
# Crack 1
_DCS6931_078= [[168, 140], [168, 168], [168, 196], [196, 196]]
_DCS6931_111= [[168, 0], [196, 0], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [140,140], [140, 168], [112, 168],  [112, 196], [140, 196]]
_DCS6931_144= [[112, 0], [140, 0], [140, 28], [140, 56], [112, 84], [112, 112], [84, 140], [112, 140], [84, 168], [112, 168], [84, 196], [112, 196]]
_DCS6931_177= [[84, 0], [112, 0], [84, 28], [84, 56], [84, 84], [84, 112], [84, 140], [84, 168], [84, 196]]
_DCS6931_210= [[84, 0], [84, 28], [56, 56], [56, 140], [84, 140], [84, 168], [84, 196], [112, 196]]
_DCS6931_243= [[112, 0], [112, 28], [112, 56], [112, 84], [112, 112], [112, 140], [112, 168], [112, 196]]
_DCS6931_276= [[196, 0]]

# Crack 2
_DCS6932_193=[[112, 196], [196, 140], [196, 168], [168, 196], [196, 196]]
_DCS6932_194=[[28, 56], [56, 56], [84, 56], [112, 56], [140, 56], [168, 56], [196, 56], [0, 84], [28, 84], [56, 84], [84, 84], [140, 84], [0, 112]]
_DCS6932_195= [[0, 56], [28, 56], [56, 56], [84, 56], [112, 56], [112, 84], [140, 84], [168, 84], [196, 84], [196, 112]]
_DCS6932_196=[[0, 112], [28, 112], [56, 112], [84, 112]]
_DCS6932_225= [[196, 84], [168, 112], [84, 140], [112, 140], [140, 140], [84, 168], [0, 196], [28, 196], [56, 196]]
_DCS6932_226=[[56, 0], [84, 0], [112, 0], [140, 0], [0, 28], [28, 28], [56, 28], [112, 28], [0, 56]]

# Crack 3
_DCS6928_137= [[28, 0], [0, 28], [56, 28], [56, 56], [84, 56], [112, 84], [112, 112], [140, 112], [140, 140], [140, 168], [140, 196], [168, 196]]
_DCS6928_170= [[168, 0], [168, 28], [168, 56], [168, 84], [168, 112], [196, 140], [196, 168], [196, 196]]
_DCS6928_203= [[168, 0], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168], [168, 196]]
_DCS6928_236= [[196, 0], [196, 28], [168, 56], [196, 56], [168, 84], [196, 84], [196, 112], [196, 140], [196, 168], [196, 196]]
_DCS6928_270= [[0, 0], [0, 28], [0, 56], [0, 84], [0, 112], [0, 140], [0, 168], [28, 168], [28, 196]]
_DCS6928_303= [[0, 0], [28, 0], [0, 28], [28, 28], [0, 56], [0, 84], [0, 112], [0, 140], [0, 168], [0, 196]]
_DCS6928_336= [[0, 0], [0, 28], [0, 56], [0, 84], [28, 112], [0, 140], [28, 140], [28, 168], [0, 196]]
_DCS6928_369= [[0, 0], [0, 28], [0, 56], [0, 84], [0, 112], [0, 140]]

# Crack 4
_DCS6928_447= [[168, 168], [196, 168], [140, 196], [168, 196]]
_DCS6928_448= [[0, 140], [28, 140], [0, 168], [28, 168], [56, 168], [84, 168], [112, 168], [140, 168], [168, 168], [196, 168]]
_DCS6928_449= [[0, 168], [0, 196], [28, 196]]
_DCS6928_482= [[56, 0], [84, 0], [84, 28], [112, 56], [112, 84], [140, 84], [140, 112], [168, 112]]
_DCS6928_483= [[0, 140], [28, 140], [140, 140], [168, 140], [196, 140], [28, 168], [56, 168], [84, 168], [112, 168]]
_DCS6928_484= [[0, 112], [28, 112], [56, 112], [0, 140], [84, 140], [112, 140], [140, 168], [168, 168], [196, 168]]
_DCS6928_485= [[0, 168], [0, 196], [28, 196], [56, 196], [84, 196], [112, 196], [140, 196]]
_DCS6928_486= [[56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [28, 168], [56, 168], [0, 196], [28, 196]]
_DCS6928_487= [[0, 112], [28, 112], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140]]
_DCS6928_488= [[0, 112], [28, 112], [28, 140]]
_DCS6928_518= [[140, 0], [168, 0], [196, 0]]

#Crack 5
_DCS7230_144 = [[84, 84], [112, 84], [140, 84], [168, 84], [196, 84], [0, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112]]
_DCS7230_145 = [[0, 56], [196, 56], [0, 84], [28, 84], [56, 84], [84, 84], [168, 84], [196, 84], [84, 112], [112, 112], [140, 112], [168, 112]]
_DCS7230_146 = [[28, 28], [56, 28], [84, 28], [28, 56], [84, 56], [84, 84], [112, 84], [112, 112], [140, 112], [168, 112], [168, 140], [196, 140], [196, 168], [0, 196]]
_DCS7230_147 = [[28, 168], [56, 168], [84, 168], [84, 196], [112, 196], [140, 196], [168, 196]]
_DCS7230_180 = [[168, 0], [196, 0], [0, 28]]
_DCS7230_181 = [[28, 0], [28, 28], [56, 28], [84, 28], [112, 28], [140, 28], [140, 56], [168, 56], [196, 56], [0, 84]]

#Crack 6
_DCS7230_183 = [[0, 196], [112, 196], [140, 196], [168, 196], [196, 196], [196, 196]]
_DCS7230_184 = [[112, 112], [140, 112], [168, 112], [196, 112], [0, 140], [56, 140], [84, 140], [0, 168], [56, 168]]
_DCS7230_185 = [[56, 84], [84, 84], [28, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 140], [0, 168]]
_DCS7230_186 = [[140, 112], [168, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140]]
_DCS7230_215 = [[84, 0], [28, 28], [56, 28], [84, 28], [112, 28], [140, 28], [168, 28], [196, 28], [0, 56]]
_DCS7230_216 = [[56, 0], [84, 0], [112, 0], [28, 28], [56, 28]]

#Crack 7
_DCS7230_344 = [[28, 56], [56, 56], [56, 84], [84, 112], [112, 112], [112, 140], [140, 140], [168, 168], [196, 168], [196, 196], [196, 196]]
_DCS7230_378 = [[28, 0], [28, 28], [56, 28], [56, 56], [56, 84], [56, 112], [84, 112], [84, 140], [84, 168], [112, 168], [112, 196], [140, 196]]
_DCS7230_411 = [[140, 0], [140, 28], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168], [168, 196]]
_DCS7230_444 = [[168, 0], [168, 28], [168, 56], [168, 84], [168, 112], [196, 140], [196, 168], [196, 196]]
_DCS7230_477 = [[196, 0], [196, 28], [168, 56], [196, 56], [168, 84], [168, 112], [168, 140], [168, 168], [196, 168], [196, 196]]
_DCS7230_510 = [[196, 0], [0, 28], [0, 56], [0, 84]]
_DCS7230_511 = [[28, 56], [28, 84], [56, 84], [56, 112], [84, 140], [84, 168], [112, 168], [112, 196], [140, 196]]

#Crack 8
_DCS7230_545 = [[28, 84], [56, 84], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140], [168, 140], [196, 140], [0, 168]]
_DCS7230_546 = [[28, 112], [56, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [0, 168]]
_DCS7230_547 = [[28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [140, 168], [168, 168], [196, 168], [0, 196], [196, 196]]
_DCS7230_548 = [[28, 168], [56, 168], [28, 196], [56, 196], [84, 196], [112, 196], [140, 196]]
_DCS7230_581 = [[140, 0], [168, 0], [196, 0], [196, 28], [0, 56]]
_DCS7230_582 = [[28, 28], [56, 28], [56, 56], [84, 56], [112, 84], [140, 84]]

#Crack 9
_DCS7221_114 = [[0, 196], [168, 196], [196, 196], [196, 196]]
_DCS7221_115 = [[56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [0, 168], [28, 168], [56, 168]]
_DCS7221_116 = [[84, 56], [112, 56], [140, 56], [56, 84], [84, 84], [168, 84], [196, 84], [0, 112], [28, 112], [56, 112]]
_DCS7221_145 = [[196, 168], [0, 196], [28, 196], [56, 196], [84, 196], [168, 196], [196, 196]]
_DCS7221_146 = [[168, 28], [196, 28], [0, 56], [112, 56], [140, 56], [168, 56], [196, 56], [0, 84], [84, 84], [112, 84], [56, 112], [84, 112], [28, 140], [56, 140]]
_DCS7221_147 = [[56, 0], [84, 0], [112, 0], [140, 0], [168, 0], [28, 28], [56, 28]]
_DCS7221_178 = [[112, 0], [140, 0], [168, 0],  [196, 0], [0, 28]]

#Crack 10
_DCS7221_117 = [[28, 112], [56, 112], [56, 140], [84, 140], [112, 168], [140, 196], [168, 196]]
_DCS7221_119 = [[0, 168], [0, 196], [168, 196], [196, 196]]
_DCS7221_120 = [[28, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [0, 168], [168, 168], [196, 168]]
_DCS7221_121 = [[28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [196, 168], [0, 196]]
_DCS7221_122 = [[28, 168], [56, 168], [84, 168]]
_DCS7221_150 = [[168, 0], [196, 0], [0, 28]]
_DCS7221_151 = [[28, 0], [56, 0], [84, 0], [56, 28], [84, 28], [112, 28], [140, 28], [168, 28], [196, 28], [0, 56]]
_DCS7221_152 = [[56, 0], [84, 0], [112, 0], [140, 0], [168, 0], [28, 28], [56, 28]]

#Crack 11
_DCS7221_343 = [[112, 0], [140, 0], [140, 28], [168, 28], [168, 56], [196, 56], [196, 84], [0, 112], [0, 140], [0, 168]]
_DCS7221_344 = [[28, 140], [56, 168], [84, 168], [84, 196], [112, 196]]
_DCS7221_377 = [[112, 0], [140, 28], [140, 56], [168, 56], [168, 84], [196, 84], [196, 112], [0, 140]]
_DCS7221_378 = [[0, 84], [196, 84], [0, 112], [28, 112], [168, 112], [0, 140], [28, 140], [56, 140], [140, 140], [56, 168], [84, 168], [112, 168], [140, 168], [84, 196]]
_DCS7221_411 = [[56, 0], [84, 0], [56, 28], [84, 28], [28, 56], [56, 56], [28, 84], [28, 112], [56, 112], [56, 140], [84, 140], [84, 168], [112, 168], [112, 196], [140, 196]]
_DCS7221_444 = [[140, 0], [140, 28], [168, 28], [168, 56], [196, 56], [196, 84], [0, 112], [0, 140]]
_DCS7221_445 = [[28, 112], [28, 140], [28, 168], [28, 196], [56, 196]]
_DCS7221_478 = [[56, 0], [56, 28], [84, 56], [84, 84], [112, 84], [112, 112], [140, 112], [140, 140], [140, 168], [140, 196], [168, 196]]
_DCS7221_511 = [[168, 0], [168, 28], [168, 56], [196, 56], [196, 84], [196, 112], [196, 140], [196, 168], [196, 196], [196, 196]]
_DCS7221_544 = [[0, 28], [0, 56]]
_DCS7221_545 = [[28, 28], [28, 56], [28, 84], [28, 112], [28, 140]]
_DCS7221_578 = [[28, 56], [28, 84], [28, 112], [28, 140], [28, 168], [28, 196]]

#Crack 12
_DCS7221_606 = [[0, 168]]
_DCS7221_607 = [[56, 84], [84, 84], [112, 84], [140, 84], [168, 84], [196, 84], [0, 112], [28, 112], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140], [28, 140], [196, 140], [0, 168]]
_DCS7221_608 = [[0, 140], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [0, 168], [84, 168], [112, 168], [140, 168]]
_DCS7221_609 = [[56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140], [28, 140], [56, 140], [84, 140], [112, 140], [168, 140], [196, 140]]
_DCS7221_610 = [[196, 28], [0, 56], [84, 56], [112, 56], [140, 56], [168, 56], [196, 56], [28, 84], [56, 84], [84, 84], [112, 84], [140, 84], [168, 84], [28, 112], [56, 112],[168, 84], [196, 84], [0, 112]]
_DCS7221_611 = [[28, 84], [56, 84], [28, 112], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [112, 140], [168, 140], [196, 140], [0, 168]]

#Crack 13
_DCS7221_612 = [[168, 112], [196, 112], [0, 140], [28, 140], [56, 140], [112, 140], [140, 140], [28, 168], [56, 168], [84, 168], [112, 168]]
_DCS7221_613 = [[28, 84], [56, 84], [84, 84], [112, 84], [196, 84], [0, 112], [112, 112], [140, 112], [168, 112], [196, 112]]
_DCS7221_614 = [[0, 28], [168, 28], [196, 28], [0, 56], [112, 56], [140, 56], [168, 56], [28, 84], [56, 84], [84, 84], [112, 84], [28, 112]]
_DCS7221_615 = [[28, 0], [28, 28], [56, 28], [84, 28], [112, 28], [140, 28], [140, 56], [168, 56], [196, 56], [0, 84], [0, 112]]
_DCS7221_616 = [[28, 56], [56, 56], [84, 56], [112, 56], [140, 56], [168, 56], [196, 56], [0, 84], [56, 84], [84, 84], [112, 84]]
_DCS7221_617 = [[28, 56], [56, 56]]

#
Crack1=['_DCS6931_078','_DCS6931_111','_DCS6931_144','_DCS6931_177','_DCS6931_210','_DCS6931_243','_DCS6931_276']
Crack2=[ '_DCS6932_193' ,'_DCS6932_194','_DCS6932_195','_DCS6932_196','_DCS6932_225','_DCS6932_226']
Crack3=[ '_DCS6928_137' ,'_DCS6928_170','_DCS6928_203','_DCS6928_236','_DCS6928_270','_DCS6928_303','_DCS6928_336','_DCS6928_369']
Crack4=[ '_DCS6928_447' ,'_DCS6928_448','_DCS6928_449','_DCS6928_482','_DCS6928_483','_DCS6928_484','_DCS6928_485','_DCS6928_486','_DCS6928_487','_DCS6928_488','_DCS6928_518']
Crack5=[ '_DCS7230_144' ,'_DCS7230_145','_DCS7230_146','_DCS7230_147','_DCS7230_180','_DCS7230_181']
Crack6=[ '_DCS7230_183' ,'_DCS7230_184','_DCS7230_185','_DCS7230_186','_DCS7230_215','_DCS7230_216']
Crack7=[ '_DCS7230_344' ,'_DCS7230_378','_DCS7230_411','_DCS7230_444','_DCS7230_477','_DCS7230_510','_DCS7230_511']
Crack8=[ '_DCS7230_545' ,'_DCS7230_546','_DCS7230_547','_DCS7230_548','_DCS7230_581','_DCS7230_582']
Crack9=[ '_DCS7221_114' ,'_DCS7221_115','_DCS7221_116','_DCS7221_145','_DCS7221_146','_DCS7221_147','_DCS7221_178']
Crack10=[ '_DCS7221_117' ,'_DCS7221_119','_DCS7221_120','_DCS7221_121','_DCS7221_122','_DCS7221_150','_DCS7221_151','_DCS7221_152']
Crack11=[ '_DCS7221_343' ,'_DCS7221_344','_DCS7221_377','_DCS7221_378','_DCS7221_411','_DCS7221_444','_DCS7221_445','_DCS7221_478','_DCS7221_511','_DCS7221_544','_DCS7221_545','_DCS7221_578']
Crack12=[ '_DCS7221_606' ,'_DCS7221_607','_DCS7221_608','_DCS7221_609','_DCS7221_610','_DCS7221_611']
Crack13=[ '_DCS7221_612' ,'_DCS7221_613','_DCS7221_614','_DCS7221_615','_DCS7221_616','_DCS7221_617']

WindowsCrack1=[_DCS6931_078, _DCS6931_111, _DCS6931_144, _DCS6931_177, _DCS6931_210, _DCS6931_243, _DCS6931_276]
WindowsCrack2=[ _DCS6932_193 ,_DCS6932_194,_DCS6932_195,_DCS6932_196,_DCS6932_225,_DCS6932_226]
WindowsCrack3=[ _DCS6928_137 ,_DCS6928_170,_DCS6928_203,_DCS6928_236,_DCS6928_270,_DCS6928_303,_DCS6928_336,_DCS6928_369]
WindowsCrack4=[ _DCS6928_447 ,_DCS6928_448,_DCS6928_449,_DCS6928_482,_DCS6928_483,_DCS6928_484,_DCS6928_485,_DCS6928_486,_DCS6928_487,_DCS6928_488,_DCS6928_518]
WindowsCrack5=[ _DCS7230_144  ,_DCS7230_145 ,_DCS7230_146 ,_DCS7230_147 ,_DCS7230_180 ,_DCS7230_181 ]
WindowsCrack6=[ _DCS7230_183  ,_DCS7230_184 ,_DCS7230_185 ,_DCS7230_186 ,_DCS7230_215 ,_DCS7230_216 ]
WindowsCrack7=[ _DCS7230_344  ,_DCS7230_378 ,_DCS7230_411 ,_DCS7230_444 ,_DCS7230_477 ,_DCS7230_510 ,_DCS7230_511 ]
WindowsCrack8=[ _DCS7230_545  ,_DCS7230_546 ,_DCS7230_547 ,_DCS7230_548 ,_DCS7230_581 ,_DCS7230_582 ]
WindowsCrack9=[ _DCS7221_114  ,_DCS7221_115 ,_DCS7221_116 ,_DCS7221_145 ,_DCS7221_146 ,_DCS7221_147 ,_DCS7221_178 ]
WindowsCrack10=[ _DCS7221_117  ,_DCS7221_119 ,_DCS7221_120 ,_DCS7221_121 ,_DCS7221_122 ,_DCS7221_150 ,_DCS7221_151 ,_DCS7221_152 ]
WindowsCrack11=[ _DCS7221_343  ,_DCS7221_344 ,_DCS7221_377 ,_DCS7221_378 ,_DCS7221_411 ,_DCS7221_444 ,_DCS7221_445 ,_DCS7221_478 ,_DCS7221_511 ,_DCS7221_544 ,_DCS7221_545 ,_DCS7221_578 ]
WindowsCrack12=[ _DCS7221_606  ,_DCS7221_607 ,_DCS7221_608 ,_DCS7221_609 ,_DCS7221_610 ,_DCS7221_611 ]
WindowsCrack13=[ _DCS7221_612  ,_DCS7221_613 ,_DCS7221_614 ,_DCS7221_615 ,_DCS7221_616 ,_DCS7221_617 ]

# Crack dimensions in terms of rows and columns of subimages for each crack (first element is the number of the crack)
crackgeometry = [[1,1,7],[2,5,2],[3,2,8],[4,9,3],[5,5,3],[6,5,3],[7,4,6],[8,5,2],[9,5,4],[10,6,3],[11,6,8],[12,6,2],[13,6,2]]

# endregion

crack=[Crack1,Crack2,Crack3,Crack4,Crack5,Crack6,Crack7,Crack8,Crack9,Crack10,Crack11,Crack12,Crack13] # list of cracks to check ()
windows=[WindowsCrack1,WindowsCrack2,WindowsCrack3,WindowsCrack4,WindowsCrack5,WindowsCrack6,WindowsCrack7,WindowsCrack8,WindowsCrack9,WindowsCrack10,WindowsCrack11,WindowsCrack12,WindowsCrack13] # list of windows for each subimage ()

resultlis=[]



for h in range (0,len(crack)):
    pathsubfolder = '\Crack ' + str(h + 1)  # Name of the folder where the predicted subimages detected as cracked are located
    path2 = path + pathsubfolder  # Complete the path name with the folder name

    # In case MAD is used, creates a folder for the specific MAD threshold used
    if method_threshold != 0:
        try:
            os.chdir(path2)  # Access the path
            # Create the Balanced folder if it doesn't exist
            os.mkdir(os.path.join(path2, 'MAD k=' + str(method_threshold)))
            print(f'Folder created successfully in {path2}')
        except OSError as e:
            print(f'Error creating folder in {path2}: {str(e)}')

    pathMAD=path2+'\MAD k='+str(method_threshold)
    pathBHist = path2 + '\Balanced'
    if method_threshold==0:
        path3=pathBHist
    else:
        path3 = pathMAD

    for i in range (0, len(crack[h])):



        os.chdir(path2)                     # Access the path

        selectedimage,imageBW=dict.selectimg(crack[h][i])  # Get the subimage (selected image) and turns it into greyscale (imageBW)
        resultado,wind=dict.joinwindows(imageBW,windows[h],i,winH,winW,method_threshold)# Get the different windows for each subimage together to create an image with only the crack
        resultlis.append(resultado)
        resultImage=dict.cleanimage(resultado,3)    # Takes the image with only the crack and removes small objects according to the specified size

        # widths: list of widths per pixel of skeleton
        # coordsk: List of coordinates of the skeleton's pixels
        # skframes: Skeleton image
        # edgesframes: Image of the edges of the crack
        # completeList: List of results, creates a vector with x,y coord of the skeleton and the corresponding width for
        #               that skeleton pixel and in mm according to the measure of pixel width
        widths,coordsk,skframes,edgesframes,completeList = dict.CrackWidth(resultImage//255,pixel_width)

        resname='res'+crack[h][i]                   # name of the image that will be saved
        skframesname = 'skframes_' + crack[h][i]  # name of the skeleton image that will be saved
        edgesframesname = 'edgesframes_' + crack[h][i]  # name of the edges image that will be saved
        completeListname = 'completeList_' + crack[h][i]+'.txt'  # name of the image that will be saved

        # If the image without small object, skeletons, edges and lists want to be saved
        if save_info==True:
            os.chdir(path3)
            dict.imgSaving(path3, resname,resultImage)  # The image where small object have been removed is saved in the path
            dict.imgSaving(path3, skframesname, skframes)  # the image where skeleton is saved in the path
            dict.imgSaving(path3, edgesframesname, edgesframes)  # the image where edges of the crack is saved in the path
            with open(path3 + '//'+completeListname, "w") as output:# saves the list as a txt file
                output.write(str(completeList))

        # Sub image with the crack obtained.
        # ===============================================
        width=selectedimage.shape[0]
        height=selectedimage.shape[1]
        finalsubimg = empty([width, height, 3], dtype=np.uint8) # creates the image with the obtained crack .
        for x in range(0, width):
            for y in range(0, height):
                if skframes[x, y] > 0:
                    finalsubimg[x, y] = [255, 0, 0] # If pixel is part of skeleton paint red
                elif resultImage[x, y] > 0:
                    finalsubimg[x, y] = [255, 255, 0]# If pixel is part of crack paint yellow
                elif edgesframes[x, y] > 0:
                    finalsubimg[x, y] = [0, 0, 255]# If pixel is part of edge, paint blue

                else:
                    finalsubimg[x, y] = selectedimage[x, y]

        finalsubimgname = 'finalsubimg' + crack[h][i]  # name of the image that will be saved
        plt.figure('final sub Image', figsize=(10, 10))

        plt.imshow(finalsubimg)
        plt.show()
        # If the final subimage want to be saved
        if save_info == True:
            dict.imgSaving(path3, finalsubimgname, finalsubimg)  # the image where skeleton is saved in the path

    # Image with the crack obtained.
    # ===============================================
    nc=crackgeometry[h][1]  # Columns of subimages for the final image
    nr=crackgeometry[h][2]  # Rows of subimages for the final image
    subimglist= [os.path.join(path3, f) for f in os.listdir(path3) if "finalsubimg" in f ] # List of cracked processed subimages paths
    path4= path2+'\\01_Uncracked_subimg\\'  # Path where the uncracked sub images are for the current crack
    uncrksubimglist=glob.glob(path4 +'*.jpg')   # List of uncracked subimages paths
    subimglist=subimglist + uncrksubimglist     # Addition of uncracked subimages paths
    sortedlist = sorted(subimglist, key=lambda im: int((im.split('_')[-1]).split('.')[0])) # List sorted in ascending order
    # merge sub images
    image = dict.merge_images(sortedlist, nr, nc)
    # merge sub images with their corresponding label
    image_div = dict.merge_images_with_labels(sortedlist, nr, nc)

    # Save the resulting processed crack and processed crack with subimage divisions and labels
    if save_info == True:
        newname = path3.split('\\')[7]+'MAD.jpg'        # Name for the resulting processed crack
        newname2 = path3.split('\\')[7]+' divMAD.jpg'   # Name for the resulting processed crack with divisions
        image.save(path3 +'\\'+ newname)                # Save image in the corresponding path
        image_div.save(path3 +'\\'+ newname2)           # Save image in the corresponding path

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")