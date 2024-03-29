"""
Created on Fri march 31 2023

@author: jca


"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import Dictionary as dict
import seaborn as sns
from scipy.stats import norm
from skimage.util import invert
import glob
import xlsxwriter
# from Crack_width_calculation import CrackWidth
from numpy import empty
from scipy.misc import face
from PIL import Image, ImageDraw
import time


start_time = time.time()


path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'


pixel_width=0.1
threshold=3
winW = 28
winH = 28


print_outl = True

# region crack info
# Initialize list crack
crack = 0
# Crack 1
_DCS6931_078 = [[168, 140], [168, 168], [168, 196], [196, 196]]
_DCS6931_111 = [[168, 0], [196, 0], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [140, 140], [140, 168],
                [112, 168], [112, 196], [140, 196]]
_DCS6931_144 = [[112, 0], [140, 0], [140, 28], [140, 56], [112, 84], [112, 112], [84, 140], [112, 140], [84, 168],
                [112, 168], [84, 196], [112, 196]]
_DCS6931_177 = [[84, 0], [112, 0], [84, 28], [84, 56], [84, 84], [84, 112], [84, 140], [84, 168], [84, 196]]
_DCS6931_210 = [[84, 0], [84, 28], [56, 56], [56, 140], [84, 140], [84, 168], [84, 196], [112, 196]]
_DCS6931_243 = [[112, 0], [112, 28], [112, 56], [112, 84], [112, 112], [112, 140], [112, 168], [112, 196]]
_DCS6931_276 = [[196, 0]]

# Crack 2
_DCS6932_193 = [[112, 196], [196, 140], [196, 168], [168, 196], [196, 196]]
_DCS6932_194 = [[28, 56], [56, 56], [84, 56], [112, 56], [140, 56], [168, 56], [196, 56], [0, 84], [28, 84],
                [56, 84],
                [84, 84], [140, 84], [0, 112]]
_DCS6932_195 = [[0, 56], [28, 56], [56, 56], [84, 56], [112, 56], [112, 84], [140, 84], [168, 84], [196, 84],
                [196, 112]]
_DCS6932_196 = [[0, 112], [28, 112], [56, 112], [84, 112]]
_DCS6932_225 = [[196, 84], [168, 112], [84, 140], [112, 140], [140, 140], [84, 168], [0, 196], [28, 196], [56, 196]]
_DCS6932_226 = [[56, 0], [84, 0], [112, 0], [140, 0], [0, 28], [28, 28], [56, 28], [112, 28], [0, 56]]

# Crack 3
_DCS6928_137 = [[28, 0], [0, 28], [56, 28], [56, 56], [84, 56], [112, 84], [112, 112], [140, 112], [140, 140],
                [140, 168], [140, 196], [168, 196]]
_DCS6928_170 = [[168, 0], [168, 28], [168, 56], [168, 84], [168, 112], [196, 140], [196, 168], [196, 196]]
_DCS6928_203 = [[168, 0], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168], [168, 196]]
_DCS6928_236 = [[196, 0], [196, 28], [168, 56], [196, 56], [168, 84], [196, 84], [196, 112], [196, 140], [196, 168],
                [196, 196]]
_DCS6928_270 = [[0, 0], [0, 28], [0, 56], [0, 84], [0, 112], [0, 140], [0, 168], [28, 168], [28, 196]]
_DCS6928_303 = [[0, 0], [28, 0], [0, 28], [28, 28], [0, 56], [0, 84], [0, 112], [0, 140], [0, 168], [0, 196]]
_DCS6928_336 = [[0, 0], [0, 28], [0, 56], [0, 84], [28, 112], [0, 140], [28, 140], [28, 168], [0, 196]]
_DCS6928_369 = [[0, 0], [0, 28], [0, 56], [0, 84], [0, 112], [0, 140]]

# Crack 4
_DCS6928_447 = [[168, 168], [196, 168], [140, 196], [168, 196]]
_DCS6928_448 = [[0, 140], [28, 140], [0, 168], [28, 168], [56, 168], [84, 168], [112, 168], [140, 168], [168, 168],
                [196, 168]]
_DCS6928_449 = [[0, 168], [0, 196], [28, 196]]
_DCS6928_482 = [[56, 0], [84, 0], [84, 28], [112, 56], [112, 84], [140, 84], [140, 112], [168, 112]]
_DCS6928_483 = [[0, 140], [28, 140], [140, 140], [168, 140], [196, 140], [28, 168], [56, 168], [84, 168],
                [112, 168]]
_DCS6928_484 = [[0, 112], [28, 112], [56, 112], [0, 140], [84, 140], [112, 140], [140, 168], [168, 168], [196, 168]]
_DCS6928_485 = [[0, 168], [0, 196], [28, 196], [56, 196], [84, 196], [112, 196], [140, 196]]
_DCS6928_486 = [[56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [28, 168], [56, 168],
                [0, 196],
                [28, 196]]
_DCS6928_487 = [[0, 112], [28, 112], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140]]
_DCS6928_488 = [[0, 112], [28, 112], [28, 140]]
_DCS6928_518 = [[140, 0], [168, 0], [196, 0]]

# Crack 5
_DCS7230_144 = [[28, 84], [28, 112], [56, 84], [56, 112], [84, 84], [84, 112], [112, 84], [112, 112], [140, 84],
                [140, 112], [168, 84], [168, 112], [196, 84], [196, 112]]
_DCS7230_145 = [[0, 84], [28, 84], [56, 84], [56, 112], [84, 112], [112, 112], [140, 112], [140, 84], [168, 84],
                [168, 56], [196, 56]]
_DCS7230_146 = [[0, 28], [0, 56], [28, 28], [56, 28], [56, 56], [56, 84], [84, 84], [84, 112], [112, 112],
                [140, 112], [140, 140], [168, 140], [168, 168], [196, 168]]
_DCS7230_147 = [[0, 168], [28, 168], [56, 168], [56, 196], [84, 168], [84, 196], [112, 196], [140, 196]]
_DCS7230_180 = [[140, 0], [168, 0], [196, 0]]
_DCS7230_181 = [[0, 0], [0, 28], [28, 28], [56, 28], [84, 28], [112, 28], [112, 56], [140, 56], [168, 56],
                [196, 56]]

# Crack 6
_DCS7230_183 = [[84, 196], [112, 196], [140, 196], [168, 196], [196, 196], [196, 168]]
_DCS7230_184 = [[0, 168], [28, 168], [28, 140], [56, 140], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112],
                [196, 112]]
_DCS7230_185 = [[0, 84], [0, 112], [28, 84], [28, 112], [56, 84], [56, 112], [84, 112], [112, 112], [140, 112],
                [168, 112], [168, 140], [196, 140]]
_DCS7230_186 = [[0, 140], [28, 140], [56, 140], [84, 140], [112, 140], [112, 112], [140, 112], [168, 112]]
_DCS7230_215 = [[0, 28], [28, 28], [56, 0], [56, 28], [84, 28], [112, 28], [140, 28], [168, 28], [196, 28]]
_DCS7230_216 = [[0, 28], [28, 28], [28, 0], [56, 0], [84, 0]]

# Crack 7
_DCS7230_344 = [[0, 56], [28, 56], [28, 84], [56, 84], [56, 112], [84, 112], [84, 140], [112, 140], [140, 140],
                [140, 168], [168, 168], [168, 196], [196, 196]]
_DCS7230_378 = [[0, 0], [0, 28], [28, 28], [28, 56], [28, 84], [28, 112], [56, 112], [56, 140], [56, 168],
                [84, 168], [84, 196], [112, 196]]
_DCS7230_411 = [[112, 0], [112, 28], [140, 0], [140, 28], [140, 56], [140, 84], [140, 112], [140, 140], [140, 168],
                [140, 196]]
_DCS7230_444 = [[140, 0], [140, 28], [140, 56], [140, 84], [140, 112], [168, 112], [168, 140], [168, 168],
                [168, 196]]
_DCS7230_477 = [[168, 0], [168, 28], [168, 56], [140, 56], [140, 84], [140, 112], [140, 140], [140, 168],
                [168, 168], [168, 196]]
_DCS7230_510 = [[168, 0], [196, 0], [196, 28], [196, 56]]
_DCS7230_511 = [[0, 56], [0, 84], [28, 84], [28, 112], [28, 140], [56, 140], [56, 168], [84, 168], [56, 140],
                [84, 196], [112, 196]]

# Crack 8
_DCS7230_545 = [[0, 84], [28, 84], [28, 112], [56, 112], [84, 84], [84, 112], [112, 112], [140, 112], [140, 140],
                [168, 140], [196, 112], [196, 140]]
_DCS7230_546 = [[0, 112], [0, 140], [28, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140],
                [196, 140]]
_DCS7230_547 = [[0, 140], [28, 140], [56, 140], [84, 140], [112, 140], [112, 168], [140, 168], [168, 168],
                [196, 168], [196, 196]]
_DCS7230_548 = [[0, 168], [0, 196], [28, 196], [56, 196], [84, 196], [112, 196]]
_DCS7230_581 = [[112, 0], [140, 0], [168, 0], [168, 28], [196, 28]]
_DCS7230_582 = [[0, 28], [28, 28], [28, 56], [56, 56], [56, 84], [84, 84], [112, 84], [140, 84]]

# Crack 9
_DCS7221_114 = [[140, 196], [168, 196], [196, 196], [196, 168]]
_DCS7221_115 = [[0, 140], [28, 168], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [140, 168],
                [168, 140], [196, 140], [196, 112]]
_DCS7221_116 = [[0, 140], [0, 112], [28, 112], [28, 84], [56, 84], [56, 56], [84, 56], [84, 28], [112, 28],
                [112, 56], [112, 84], [140, 56], [140, 84], [168, 84], [196, 84]]
_DCS7221_145 = [[0, 168], [0, 196], [28, 168], [28, 196], [56, 196], [140, 196], [168, 196], [168, 168], [196, 168],
                [196, 140]]
_DCS7221_146 = [[0, 140], [28, 140], [28, 112], [56, 112], [56, 84], [84, 84], [84, 56], [112, 56], [112, 28],
                [140, 28], [140, 56], [168, 28], [168, 56], [196, 28], [196, 56]]
_DCS7221_147 = [[0, 28], [28, 28], [28, 0], [56, 0], [84, 0], [112, 0], [140, 0]]
_DCS7221_178 = [[56, 0], [84, 0], [112, 0], [140, 0], [168, 0], [196, 0]]

# Crack 10
_DCS7221_117 = [[28, 112], [28, 140], [56, 140], [56, 168], [84, 168], [84, 196], [112, 168], [112, 196],
                [140, 196], [168, 196]]
_DCS7221_119 = [[140, 196], [168, 196], [168, 168], [196, 168], [196, 140]]
_DCS7221_120 = [[0, 112], [0, 140], [28, 140], [56, 140], [84, 140], [84, 168], [112, 140], [140, 140], [140, 168],
                [168, 140], [168, 168], [196, 140]]
_DCS7221_121 = [[0, 140], [28, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140],
                [168, 168], [196, 168]]
_DCS7221_122 = [[0, 168], [28, 168], [56, 168], [84, 168]]
_DCS7221_150 = [[168, 0], [196, 0]]
_DCS7221_151 = [[0, 0], [28, 0], [28, 28], [56, 0], [56, 28], [84, 28], [112, 28], [140, 28], [168, 28], [196, 28]]
_DCS7221_152 = [[0, 28], [28, 0], [28, 28], [56, 0], [84, 0], [112, 0], [140, 0]]

# Crack 11
_DCS7221_343 = [[84, 0], [112, 0], [112, 28], [140, 28], [140, 56], [168, 56], [168, 84], [196, 84], [196, 112],
                [196, 140]]
_DCS7221_344 = [[0, 140], [0, 168], [28, 168], [56, 168], [56, 196], [84, 196]]
_DCS7221_377 = [[84, 0], [84, 28], [112, 28], [112, 56], [140, 56], [140, 84], [168, 84], [168, 112], [196, 112]]
_DCS7221_378 = [[0, 112], [0, 140], [28, 140], [28, 168], [56, 168], [56, 196]]
_DCS7221_411 = [[28, 0], [56, 0], [56, 28], [28, 28], [28, 56], [0, 56], [0, 84], [0, 112], [28, 112], [56, 140],
                [56, 168], [84, 168], [84, 196], [112, 196]]
_DCS7221_444 = [[112, 0], [112, 28], [140, 28], [140, 56], [168, 56], [168, 84], [196, 56], [196, 84], [196, 112]]
_DCS7221_445 = [[0, 112], [0, 140], [0, 168], [0, 196], [28, 196]]
_DCS7221_478 = [[28, 0], [28, 28], [56, 28], [56, 56], [56, 84], [84, 84], [84, 112], [112, 112], [112, 140],
                [112, 168], [112, 196], [140, 196]]
_DCS7221_511 = [[140, 0], [140, 28], [168, 28], [140, 56], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168],
                [168, 196], [196, 196]]
_DCS7221_544 = [[196, 0], [196, 28], [196, 168], [196, 196]]
_DCS7221_545 = [[0, 28], [0, 56], [0, 84], [0, 112], [0, 140]]
_DCS7221_577 = [[196, 0], [196, 28], [196, 56]]
_DCS7221_578 = [[0, 56], [0, 84], [0, 112], [0, 140], [0, 168], [0, 196]]

# Crack 12
_DCS7221_606 = [[196, 112], [196, 140]]
_DCS7221_607 = [[0, 112], [0, 140], [28, 112], [28, 84], [56, 84], [56, 112], [84, 84], [84, 112], [112, 84],
                [112, 112], [140, 84], [140, 112], [168, 112], [168, 140], [196, 112], [196, 140]]
_DCS7221_608 = [[0, 140], [28, 140], [56, 140], [56, 168], [84, 140], [84, 168], [112, 140], [112, 168], [140, 140],
                [168, 140], [196, 140]]
_DCS7221_609 = [[0, 140], [28, 112], [28, 140], [56, 112], [56, 140], [84, 112], [84, 140], [112, 112], [140, 112],
                [140, 140], [168, 112], [168, 140], [196, 112]]
_DCS7221_610 = [[0, 112], [28, 112], [28, 84], [56, 84], [56, 56], [84, 56], [84, 84], [112, 56], [112, 84],
                [140, 84], [140, 56], [168, 28], [196, 28]]
_DCS7221_611 = [[0, 84], [0, 112], [28, 84], [28, 112], [56, 112], [84, 112], [84, 140], [112, 112], [112, 140],
                [140, 112], [140, 140], [168, 140], [168, 168], [196, 140], [196, 168]]

# Crack 13
_DCS7221_612 = [[0, 140], [0, 168], [28, 140], [28, 168], [56, 168], [84, 168], [84, 140], [112, 140], [112, 112],
                [140, 112], [168, 112], [196, 112]]
_DCS7221_613 = [[0, 84], [28, 84], [56, 84], [84, 84], [84, 112], [112, 112], [140, 112], [168, 84], [168, 112],
                [196, 84]]
_DCS7221_614 = [[0, 84], [28, 84], [56, 84], [84, 84], [84, 56], [112, 56], [140, 56], [140, 28], [168, 28],
                [196, 28], [196, 0]]
_DCS7221_615 = [[0, 0], [0, 28], [28, 0], [28, 28], [56, 28], [84, 28], [112, 28], [112, 56], [140, 56], [168, 56],
                [196, 56], [196, 84]]
_DCS7221_616 = [[0, 56], [28, 56], [56, 56], [56, 84], [84, 84], [84, 56], [112, 56], [140, 56], [168, 56],
                [196, 56]]
_DCS7221_617 = [[0, 56], [28, 56]]

#
Crack1 = ['_DCS6931_078', '_DCS6931_111', '_DCS6931_144', '_DCS6931_177', '_DCS6931_210', '_DCS6931_243',
          '_DCS6931_276']
Crack2 = ['_DCS6932_193', '_DCS6932_194', '_DCS6932_195', '_DCS6932_196', '_DCS6932_225', '_DCS6932_226']
Crack3 = ['_DCS6928_137', '_DCS6928_170', '_DCS6928_203', '_DCS6928_236', '_DCS6928_270', '_DCS6928_303',
          '_DCS6928_336', '_DCS6928_369']
Crack4 = ['_DCS6928_447', '_DCS6928_448', '_DCS6928_449', '_DCS6928_482', '_DCS6928_483', '_DCS6928_484',
          '_DCS6928_485', '_DCS6928_486', '_DCS6928_487', '_DCS6928_488', '_DCS6928_518']
Crack5 = ['_DCS7230_144', '_DCS7230_145', '_DCS7230_146', '_DCS7230_147', '_DCS7230_180', '_DCS7230_181']
Crack6 = ['_DCS7230_183', '_DCS7230_184', '_DCS7230_185', '_DCS7230_186', '_DCS7230_215', '_DCS7230_216']
Crack7 = ['_DCS7230_344', '_DCS7230_378', '_DCS7230_411', '_DCS7230_444', '_DCS7230_477', '_DCS7230_510',
          '_DCS7230_511']
Crack8 = ['_DCS7230_545', '_DCS7230_546', '_DCS7230_547', '_DCS7230_548', '_DCS7230_581', '_DCS7230_582']
Crack9 = ['_DCS7221_114', '_DCS7221_115', '_DCS7221_116', '_DCS7221_145', '_DCS7221_146', '_DCS7221_147',
          '_DCS7221_178']
Crack10 = ['_DCS7221_117', '_DCS7221_119', '_DCS7221_120', '_DCS7221_121', '_DCS7221_122', '_DCS7221_150',
           '_DCS7221_151', '_DCS7221_152']
Crack11 = ['_DCS7221_343', '_DCS7221_344', '_DCS7221_377', '_DCS7221_378', '_DCS7221_411', '_DCS7221_444',
           '_DCS7221_445', '_DCS7221_478', '_DCS7221_511', '_DCS7221_544', '_DCS7221_545', '_DCS7221_577',
           '_DCS7221_578']
Crack12 = ['_DCS7221_606', '_DCS7221_607', '_DCS7221_608', '_DCS7221_609', '_DCS7221_610', '_DCS7221_611']
Crack13 = ['_DCS7221_612', '_DCS7221_613', '_DCS7221_614', '_DCS7221_615', '_DCS7221_616', '_DCS7221_617']

Crack14 = ['_DCS0113_02', '_DCS0113_03', '_DCS0113_110', '_DCS0113_111', '_DCS0113_16', '_DCS0113_29',
           '_DCS0113_30',
           '_DCS0113_43', '_DCS0113_44', '_DCS0113_57', '_DCS0113_70', '_DCS0113_71', '_DCS0113_84', '_DCS0113_97']

WindowsCrack1 = [_DCS6931_078, _DCS6931_111, _DCS6931_144, _DCS6931_177, _DCS6931_210, _DCS6931_243, _DCS6931_276]
WindowsCrack2 = [_DCS6932_193, _DCS6932_194, _DCS6932_195, _DCS6932_196, _DCS6932_225, _DCS6932_226]
WindowsCrack3 = [_DCS6928_137, _DCS6928_170, _DCS6928_203, _DCS6928_236, _DCS6928_270, _DCS6928_303, _DCS6928_336,
                 _DCS6928_369]
WindowsCrack4 = [_DCS6928_447, _DCS6928_448, _DCS6928_449, _DCS6928_482, _DCS6928_483, _DCS6928_484, _DCS6928_485,
                 _DCS6928_486, _DCS6928_487, _DCS6928_488, _DCS6928_518]
WindowsCrack5 = [_DCS7230_144, _DCS7230_145, _DCS7230_146, _DCS7230_147, _DCS7230_180, _DCS7230_181]
WindowsCrack6 = [_DCS7230_183, _DCS7230_184, _DCS7230_185, _DCS7230_186, _DCS7230_215, _DCS7230_216]
WindowsCrack7 = [_DCS7230_344, _DCS7230_378, _DCS7230_411, _DCS7230_444, _DCS7230_477, _DCS7230_510, _DCS7230_511]
WindowsCrack8 = [_DCS7230_545, _DCS7230_546, _DCS7230_547, _DCS7230_548, _DCS7230_581, _DCS7230_582]
WindowsCrack9 = [_DCS7221_114, _DCS7221_115, _DCS7221_116, _DCS7221_145, _DCS7221_146, _DCS7221_147, _DCS7221_178]
WindowsCrack10 = [_DCS7221_117, _DCS7221_119, _DCS7221_120, _DCS7221_121, _DCS7221_122, _DCS7221_150, _DCS7221_151,
                  _DCS7221_152]
WindowsCrack11 = [_DCS7221_343, _DCS7221_344, _DCS7221_377, _DCS7221_378, _DCS7221_411, _DCS7221_444, _DCS7221_445,
                  _DCS7221_478, _DCS7221_511, _DCS7221_544, _DCS7221_545, _DCS7221_577, _DCS7221_578]
WindowsCrack12 = [_DCS7221_606, _DCS7221_607, _DCS7221_608, _DCS7221_609, _DCS7221_610, _DCS7221_611]
WindowsCrack13 = [_DCS7221_612, _DCS7221_613, _DCS7221_614, _DCS7221_615, _DCS7221_616, _DCS7221_617]
WindowsCrack13 = [_DCS7221_612, _DCS7221_613, _DCS7221_614, _DCS7221_615, _DCS7221_616, _DCS7221_617]

# Crack dimensions in terms of rows and columns of subimages for each crack (first element is the number of the crack,
# second number is the number of columns and third is rows)

crackgeometry = [[1, 1, 7], [2, 5, 2], [3, 2, 8], [4, 9, 3], [5, 5, 3], [6, 5, 3], [7, 4, 6], [8, 5, 2], [9, 5, 4],
                 [10, 6, 3], [11, 6, 8], [12, 6, 2], [13, 6, 2], [14, 6, 10]]
# list of cracks to check
crack = [Crack1, Crack2, Crack3, Crack4, Crack5, Crack6, Crack7, Crack8, Crack9, Crack10, Crack11, Crack12, Crack13
         ]
# list of windows for each subimage
windows = [WindowsCrack1, WindowsCrack2, WindowsCrack3, WindowsCrack4, WindowsCrack5, WindowsCrack6, WindowsCrack7,
           WindowsCrack8, WindowsCrack9, WindowsCrack10, WindowsCrack11, WindowsCrack12,
           WindowsCrack13]
# endregion


for h in range (9,10):#0,len(crack)
    pathsubfolder = '\Crack ' + str(h + 1)  # Name of the folder where the predicted subimages detected as cracked are located
    path2 = path + pathsubfolder  # Complete the path name with the folder name
    os.chdir(path2)  # Access the path

    try:
        os.chdir(path2)  # Access the path
        # Create the Balanced folder if it doesn't exist
        os.mkdir(os.path.join(path2, 'window_histogramsMAD k=' + str(threshold)))
        print(f'Folder created successfully in {path2}')
    except OSError as e:
        print(f'Error creating folder in {path2}: {str(e)}')
    path3 = path2 + '\window_histogramsMAD k=' + str(threshold)

    for i in range(0, len(crack[h])):
        os.chdir(path2)  # Access the path
        # Get the subimage (selected image) and turns it into greyscale (imageBW)
        selectedimage,imageBW=dict.selectimg(crack[h][i])
        for j in range(0, len(windows[h][i])):
            x = windows[h][i][j][0]  # x coordinate of the upper left corner of the window to evaluate
            y = windows[h][i][j][1]  # y coordinate of the upper left corner of the window to evaluate
            window = imageBW[y:y + winH, x:x + winW]  # window to evaluate from x and y coord
            WinData = window.ravel()
            Outl, med, medlist, inliers, trhs = dict.detect_outliers_mad(WinData, threshold)
            clone = imageBW.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

            # Generates subimage, histogram, kde, outliers and hist with outliers and median plot
            figure = dict.plot_hist_kde_outl(window,clone, Outl, medlist)

            if print_outl == True:
                name='histograms'+str(crack[h][i])+str(windows[h][i][j])
                os.chdir(path3)  # Access the path
                figure.savefig(name+'.png')
                # Close the figure to release memory (optional)
                plt.close(figure)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

