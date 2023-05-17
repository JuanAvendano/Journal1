"""
Created on Sat april 01, 2023,
@author: jca

Process on a single crack to obtain skeleton subimages, edges subimages, crack binary subimages, coord and widths per
subimage, final subimages (with crack, edges and skeleton pixels), and final images reconstructed with and without
divisions

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import Dictionary as dict
from numpy import empty
import time
import glob

start_time = time.time()


def unit_result(n, k, pixelwidth, save_info, saveimgparts, savesubimg, saveimg):
    # # ================================================================================================================
    # Inputs
    # # ================================================================================================================
    # Path for the different possible cracks
    path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'
    # Number of the cracked that is going to be processed
    n = n
    # Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
    method_threshold = k
    # pixel_width in mm
    pixel_width = pixelwidth
    # If the info related to x,y coordinates and widths want to be saved as text file
    save_info = save_info
    # If image without small object, skeletons, edges want to be saved as png
    save_img_parts = saveimgparts
    # If the generated final subimages want to be saved
    save_subimg = savesubimg
    # If the generated images want to be saved
    save_img = saveimg
    # Size of the window to use on every subimage
    winAarray = [28]  # , 32, 37, 56

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
    _DCS7230_144 = [[84, 84], [112, 84], [140, 84], [168, 84], [196, 84], [0, 112], [84, 112], [112, 112], [140, 112],
                    [168, 112], [196, 112]]
    _DCS7230_145 = [[0, 56], [196, 56], [0, 84], [28, 84], [56, 84], [84, 84], [168, 84], [196, 84], [84, 112],
                    [112, 112],
                    [140, 112], [168, 112]]
    _DCS7230_146 = [[28, 28], [56, 28], [84, 28], [28, 56], [84, 56], [84, 84], [112, 84], [112, 112], [140, 112],
                    [168, 112], [168, 140], [196, 140], [196, 168], [0, 196]]
    _DCS7230_147 = [[28, 168], [56, 168], [84, 168], [84, 196], [112, 196], [140, 196], [168, 196]]
    _DCS7230_180 = [[168, 0], [196, 0], [0, 28]]
    _DCS7230_181 = [[28, 0], [28, 28], [56, 28], [84, 28], [112, 28], [140, 28], [140, 56], [168, 56], [196, 56],
                    [0, 84]]

    # Crack 6
    _DCS7230_183 = [[0, 196], [112, 196], [140, 196], [168, 196], [196, 196], [196, 196]]
    _DCS7230_184 = [[112, 112], [140, 112], [168, 112], [196, 112], [0, 140], [56, 140], [84, 140], [0, 168], [56, 168]]
    _DCS7230_185 = [[56, 84], [84, 84], [28, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 140], [0, 168]]
    _DCS7230_186 = [[140, 112], [168, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140]]
    _DCS7230_215 = [[84, 0], [28, 28], [56, 28], [84, 28], [112, 28], [140, 28], [168, 28], [196, 28], [0, 56]]
    _DCS7230_216 = [[56, 0], [84, 0], [112, 0], [28, 28], [56, 28]]

    # Crack 7
    _DCS7230_344 = [[0, 56], [28, 56], [28, 84], [56, 84], [56, 112], [84, 112], [84, 140], [112, 140], [140, 140],
                    [140, 168], [168, 168], [168, 196], [196, 196]]
    _DCS7230_378 = [[0, 0], [0, 28], [28, 28], [28, 56], [28, 84], [28, 112], [56, 112], [56, 140], [56, 168],
                    [84, 168], [84, 196], [112, 196]]
    _DCS7230_411 = [[112, 0], [112, 28], [140, 0], [140, 28], [140, 56], [140, 84], [140, 112], [140, 140], [140, 168],
                    [140, 196]]
    _DCS7230_444 = [[140, 0], [140, 56], [140, 84], [140, 112], [168, 112], [168, 140], [168, 168], [168, 196]]
    _DCS7230_477 = [[168, 0], [168, 28], [168, 56], [140, 56], [140, 84], [140, 112], [140, 140], [140, 168],
                    [168, 168], [168, 196]]
    _DCS7230_510 = [[168, 0], [196, 0], [196, 28], [196, 56]]
    _DCS7230_511 = [[0, 56], [0, 84], [28, 84], [28, 112], [28, 140], [56, 140], [56, 168], [84, 168], [56, 140],
                    [112, 196]]

    # Crack 8
    _DCS7230_545 = [[28, 84], [56, 84], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140],
                    [168, 140], [196, 140], [0, 168]]
    _DCS7230_546 = [[28, 112], [56, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140],
                    [196, 140],
                    [0, 168]]
    _DCS7230_547 = [[28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [140, 168], [168, 168], [196, 168],
                    [0, 196],
                    [196, 196]]
    _DCS7230_548 = [[28, 168], [56, 168], [28, 196], [56, 196], [84, 196], [112, 196], [140, 196]]
    _DCS7230_581 = [[140, 0], [168, 0], [196, 0], [196, 28], [0, 56]]
    _DCS7230_582 = [[28, 28], [56, 28], [56, 56], [84, 56], [112, 84], [140, 84]]

    # Crack 9
    _DCS7221_114 = [[0, 196], [168, 196], [196, 196], [196, 196]]
    _DCS7221_115 = [[56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [0, 168], [28, 168],
                    [56, 168]]
    _DCS7221_116 = [[84, 56], [112, 56], [140, 56], [56, 84], [84, 84], [168, 84], [196, 84], [0, 112], [28, 112],
                    [56, 112]]
    _DCS7221_145 = [[196, 168], [0, 196], [28, 196], [56, 196], [84, 196], [168, 196], [196, 196]]
    _DCS7221_146 = [[168, 28], [196, 28], [0, 56], [112, 56], [140, 56], [168, 56], [196, 56], [0, 84], [84, 84],
                    [112, 84],
                    [56, 112], [84, 112], [28, 140], [56, 140]]
    _DCS7221_147 = [[56, 0], [84, 0], [112, 0], [140, 0], [168, 0], [28, 28], [56, 28]]
    _DCS7221_178 = [[112, 0], [140, 0], [168, 0], [196, 0], [0, 28]]

    # Crack 10
    _DCS7221_117 = [[28, 112], [56, 112], [56, 140], [84, 140], [112, 168], [140, 196], [168, 196]]
    _DCS7221_119 = [[0, 168], [0, 196], [168, 196], [196, 196]]
    _DCS7221_120 = [[28, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140],
                    [0, 168],
                    [168, 168], [196, 168]]
    _DCS7221_121 = [[28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [196, 168],
                    [0, 196]]
    _DCS7221_122 = [[28, 168], [56, 168], [84, 168]]
    _DCS7221_150 = [[168, 0], [196, 0], [0, 28]]
    _DCS7221_151 = [[28, 0], [56, 0], [84, 0], [56, 28], [84, 28], [112, 28], [140, 28], [168, 28], [196, 28], [0, 56]]
    _DCS7221_152 = [[56, 0], [84, 0], [112, 0], [140, 0], [168, 0], [28, 28], [56, 28]]

    # Crack 11
    _DCS7221_343 = [[84, 0], [112, 0], [112, 28], [140, 28], [140, 56], [168, 56], [168, 84], [196, 84], [196, 112],
                    [196, 140]]
    _DCS7221_344 = [[0, 140], [0, 168], [28, 168], [56, 168], [56, 196], [84, 196]]
    _DCS7221_377 = [[84, 0], [84, 28], [112, 28], [112, 56], [140, 56], [140, 84], [168, 84], [168, 112], [196, 112]]
    _DCS7221_378 = [[0, 112], [0, 140], [28, 140], [28, 168], [56, 168], [56, 196]]
    _DCS7221_411 = [[28, 0], [28, 196], [56, 28], [28, 28], [28, 56], [0, 56], [0, 84], [0, 112], [28, 112], [56, 140],
                    [56, 168], [84, 168], [84, 196], [112, 196]]
    _DCS7221_444 = [[112, 0], [112, 28], [140, 28], [140, 56], [168, 56], [168, 84], [196, 84], [196, 112]]
    _DCS7221_445 = [[0, 112], [0, 140], [0, 168], [0, 196], [28, 196]]
    _DCS7221_478 = [[28, 0], [28, 28], [56, 28], [56, 56], [56, 84], [84, 84], [84, 112], [112, 84], [112, 140],
                    [112, 168], [112, 196], [140, 196]]
    _DCS7221_511 = [[140, 0], [140, 28], [140, 56], [168, 84], [168, 112], [168, 140], [168, 168], [168, 196],
                    [196, 196]]
    _DCS7221_544 = [[168, 196], [196, 28], [196, 168], [196, 196]]
    _DCS7221_545 = [[0, 28], [0, 56], [0, 84], [0, 112], [0, 140]]
    _DCS7221_578 = [[0, 56], [0, 84], [0, 112], [0, 140], [0, 168], [0, 196]]

    # Crack 12
    _DCS7221_606 = [[0, 168]]
    _DCS7221_607 = [[56, 84], [84, 84], [112, 84], [140, 84], [168, 84], [196, 84], [0, 112], [28, 112], [56, 112],
                    [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140], [28, 140], [196, 140],
                    [0, 168]]
    _DCS7221_608 = [[0, 140], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [0, 168],
                    [84, 168], [112, 168], [140, 168]]
    _DCS7221_609 = [[56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140], [28, 140],
                    [56, 140],
                    [84, 140], [112, 140], [168, 140], [196, 140]]
    _DCS7221_610 = [[196, 28], [0, 56], [84, 56], [112, 56], [140, 56], [168, 56], [196, 56], [28, 84], [56, 84],
                    [84, 84],
                    [112, 84], [140, 84], [168, 84], [28, 112], [56, 112], [168, 84], [196, 84], [0, 112]]
    _DCS7221_611 = [[28, 84], [56, 84], [28, 112], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [112, 140],
                    [168, 140], [196, 140], [0, 168]]

    # Crack 13
    _DCS7221_612 = [[168, 112], [196, 112], [0, 140], [28, 140], [56, 140], [112, 140], [140, 140], [28, 168],
                    [56, 168],
                    [84, 168], [112, 168]]
    _DCS7221_613 = [[28, 84], [56, 84], [84, 84], [112, 84], [196, 84], [0, 112], [112, 112], [140, 112], [168, 112],
                    [196, 112]]
    _DCS7221_614 = [[0, 28], [168, 28], [196, 28], [0, 56], [112, 56], [140, 56], [168, 56], [28, 84], [56, 84],
                    [84, 84],
                    [112, 84], [28, 112]]
    _DCS7221_615 = [[28, 0], [28, 28], [56, 28], [84, 28], [112, 28], [140, 28], [140, 56], [168, 56], [196, 56],
                    [0, 84],
                    [0, 112]]
    _DCS7221_616 = [[28, 56], [56, 56], [84, 56], [112, 56], [140, 56], [168, 56], [196, 56], [0, 84], [56, 84],
                    [84, 84],
                    [112, 84]]
    _DCS7221_617 = [[28, 56], [56, 56]]

    # Crack14
    _DCS0113_02 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_03 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_110 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_111 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_16 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_29 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_30 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_43 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_44 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_57 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_70 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_71 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_84 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]

    _DCS0113_97 = [[i, j] for i in range(0, 197, 28) for j in range(0, 197, 28)]


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
               '_DCS7221_445', '_DCS7221_478', '_DCS7221_511', '_DCS7221_544', '_DCS7221_545', '_DCS7221_578']
    Crack12 = ['_DCS7221_606', '_DCS7221_607', '_DCS7221_608', '_DCS7221_609', '_DCS7221_610', '_DCS7221_611']
    Crack13 = ['_DCS7221_612', '_DCS7221_613', '_DCS7221_614', '_DCS7221_615', '_DCS7221_616', '_DCS7221_617']

    Crack14 = ['_DCS0113_02', '_DCS0113_03', '_DCS0113_110', '_DCS0113_111', '_DCS0113_16', '_DCS0113_29', '_DCS0113_30',
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
                      _DCS7221_478, _DCS7221_511, _DCS7221_544, _DCS7221_545, _DCS7221_578]
    WindowsCrack12 = [_DCS7221_606, _DCS7221_607, _DCS7221_608, _DCS7221_609, _DCS7221_610, _DCS7221_611]
    WindowsCrack13 = [_DCS7221_612, _DCS7221_613, _DCS7221_614, _DCS7221_615, _DCS7221_616, _DCS7221_617]
    WindowsCrack13 = [_DCS7221_612, _DCS7221_613, _DCS7221_614, _DCS7221_615, _DCS7221_616, _DCS7221_617]
    WindowsCrack14 = [_DCS0113_02, _DCS0113_03, _DCS0113_110, _DCS0113_111, _DCS0113_16, _DCS0113_29, _DCS0113_30,
               _DCS0113_43, _DCS0113_44, _DCS0113_57, _DCS0113_70, _DCS0113_71, _DCS0113_84, _DCS0113_97]
    # Crack dimensions in terms of rows and columns of subimages for each crack (first element is the number of the crack,
    # second number is the number of columns and third is rows)

    crackgeometry = [[1, 1, 7], [2, 5, 2], [3, 2, 8], [4, 9, 3], [5, 5, 3], [6, 5, 3], [7, 4, 6], [8, 5, 2], [9, 5, 4],
                     [10, 6, 3], [11, 6, 8], [12, 6, 2], [13, 6, 2],[14,6,10]]

    # endregion

    # list of cracks to check
    crack = [Crack1, Crack2, Crack3, Crack4, Crack5, Crack6, Crack7, Crack8, Crack9, Crack10, Crack11, Crack12, Crack13, Crack14]
    # list of windows for each subimage
    windows = [WindowsCrack1, WindowsCrack2, WindowsCrack3, WindowsCrack4, WindowsCrack5, WindowsCrack6, WindowsCrack7,
               WindowsCrack8, WindowsCrack9, WindowsCrack10, WindowsCrack11, WindowsCrack12,
               WindowsCrack13,WindowsCrack14]
    # List for the subimages results
    resultlis = []

    # List of cracks to check
    crack = [crack[n - 1]]
    # list of windows for each subimage
    windows = [windows[n - 1]]

    # # ================================================================================================================
    # Process
    # # ================================================================================================================
    for h in range(0, len(crack)):

        # # ============================================================================================================
        # 1. Paths arrangement
        # # ============================================================================================================
        # Name of the folder where the information of the desired crack is located
        pathsubfolder = '\Crack ' + str(h + n)
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

        # Path where results will be saved if using MAD
        pathMAD = path2 + '\MAD k=' + str(method_threshold)
        # Path where results will be saved if using Balanced Thresholding
        pathBHist = path2 + '\Balanced'
        # Selection of the path to be used according to the method selected
        if method_threshold == 0:
            path3 = pathBHist
        else:
            path3 = pathMAD

        # # ============================================================================================================
        # 2. Work over the crack used.
        # # ============================================================================================================
        for i in range(0, len(crack[h])):
            # Access the path
            os.chdir(path2)

            # # =======================================================================================================.
            # 2.1 Process over each subimage for the different windows sizes.
            # # =======================================================================================================.
            for k in range(0, len(winAarray)):
                # Size of the window used (must be squared)
                winH = winAarray[k]
                winW = winH

                # 2.1.1 Selects image
                # Get the subimage (selected image) and turns it into greyscale (imageBW)
                # ========================================================================================================
                selectedimage, imageBW = dict.selectimg(crack[h][i])

                # 2.1.2 Method and joining
                # Applies method and joins the windows for each subimage together to create a subimage with only the crack
                # =======================================================================================================
                resultado, wind = dict.joinwindows(imageBW, windows[h], i, winH, winW, method_threshold)

                # Adds the result to a list
                resultlis.append(resultado)

                # 2.1.3 Remove small objects
                # Takes the image with only the crack and removes small objects according to the specified size
                # ========================================================================================================
                resultImage = dict.cleanimage(resultado, 3)

                # 2.1.4 Crack width calculation
                # Takes the resultImage and determines skeleton, edges, calculates widths and generates a list with results
                # ========================================================================================================

                # widths: list of widths per pixel of skeleton
                # coordsk: List of coordinates of the skeleton's pixels
                # skframes: Skeleton image
                # edgesframes: Image of the edges of the crack
                # completeList: List of results, creates a vector with x,y coord of the skeleton and the corresponding width
                #   for that skeleton pixel and in mm according to the measure of pixel width
                widths, coordsk, skframes, edgesframes, completeList = dict.CrackWidth(resultImage // 255, pixel_width)
                # 2.1.5 Save information
                # If selected, saves the result image, skeleton, edges (binary images) and list with results
                # ========================================================================================================

                # Names for the results images
                resname = 'res' + crack[h][i]  # name of the image that will be saved
                skframesname = 'skframes_' + crack[h][i]  # name of the skeleton image that will be saved
                edgesframesname = 'edgesframes_' + crack[h][i]  # name of the edges image that will be saved
                completeListname = 'completeList_' + crack[h][i] + '.txt'  # name of the image that will be saved

                # Saves the image without small object, skeletons, edges
                # ===============================================
                if save_img_parts == True:
                    os.chdir(path3)
                    # The image where small object have been removed is saved in the path
                    dict.BinarySaving(path3, resname, resultImage)
                    # the image where skeleton is saved in the path
                    dict.BinarySaving(path3, skframesname, skframes)
                    # the image where edges of the crack is saved in the path
                    dict.BinarySaving(path3, edgesframesname, edgesframes)

                # saves the list as a txt file
                # ===============================================
                if save_info == True:
                    # Columns names
                    column_names = ['Y coord', 'X coord', 'Width (pxl)', 'Width (mm)', 'Danger group']
                    header = '\t'.join(column_names)
                    # Writes header with the names of the columns and fills every row with the values in a certain format
                    with open(path3 + '//' + completeListname, "w") as output:
                        output.write(header + '\n')
                        for row in completeList:
                            row_str = '\t'.join('{:.2f}'.format(value) for value in row)
                            output.write(row_str + '\n')

                # 2.1.6 Sub image with the crack obtained.
                # Colors over the subimage, the obtained skeleton, crack pixels and edges.
                # ========================================================================================================
                width = selectedimage.shape[0]
                height = selectedimage.shape[1]
                finalsubimg = empty([width, height, 3], dtype=np.uint8)  # creates the image with the obtained crack .
                for x in range(0, width):
                    for y in range(0, height):
                        if skframes[x, y] > 0:
                            finalsubimg[x, y] = [255, 0, 0]  # If pixel is part of skeleton paint red
                        elif resultImage[x, y] > 0:
                            finalsubimg[x, y] = [255, 255, 0]  # If pixel is part of crack paint yellow
                        elif edgesframes[x, y] > 0:
                            finalsubimg[x, y] = [0, 0, 255]  # If pixel is part of edge, paint blue

                        else:
                            finalsubimg[x, y] = selectedimage[x, y]

                # Saves the final subimage as a png if save_img selected
                # ======================================================
                # Name of the image that will be saved
                finalsubimgname = 'finalsubimg' + crack[h][i]
                # Plot the final sub image
                plt.figure('final sub Image', figsize=(10, 10))
                plt.imshow(finalsubimg)
                plt.show()

                # If the final subimage want to be saved
                if save_subimg:
                    os.chdir(path)
                    # The image is saved in the path
                    dict.imgSaving(path3, finalsubimgname, finalsubimg)
                    # finalsubimgRGB = Image.fromarray(finalsubimg)
                    # # finalsubimgRGB=finalsubimgRGB.convert('sRGB')
                    # finalsubimgRGB.save('output.png')

        # # ================================================================================================================
        # 3. Image with the crack obtained joining the different subimages processed
        # # ================================================================================================================
        # 3.1 Image geometry in terms of subimages
        # # ===============================================================================================================

        # Columns of subimages for the final image
        nc = crackgeometry[n-1][1]
        # Rows of subimages for the final image
        nr = crackgeometry[n-1][2]

        # 3.2 List of paths for the cracked processed and uncracked subimages
        # # ===============================================================================================================
        # List of cracked processed subimages (finalsubimg) paths
        subimglist = [os.path.join(path3, f) for f in os.listdir(path3) if "finalsubimg" in f]
        # Path where the uncracked sub images are for the current crack
        path4 = path2 + '\\01_Uncracked_subimg\\'
        # List of uncracked subimages paths
        uncrksubimglist = glob.glob(path4 + '*.png')
        # Addition of uncracked subimages path lists
        subimglist = subimglist + uncrksubimglist
        # List sorted in ascending order
        sortedlist = sorted(subimglist, key=lambda im: int((im.split('_')[-1]).split('.')[0]))

        # 3.3 Merge sub images (cracked processed and uncracked)
        # # ===============================================================================================================
        image = dict.merge_images(sortedlist, nr, nc)
        # merge sub images with their corresponding label
        image_div = dict.merge_images_with_labels(sortedlist, nr, nc)

        # Save the resulting processed crack and processed crack with subimage divisions and labels
        # # =======================================================================================
        if save_img == True:
            newname = path3.split('\\')[7] + 'MAD.png'  # Name for the resulting processed crack
            newname2 = path3.split('\\')[7] + ' divMAD.png'  # Name for the resulting processed crack with divisions
            image.save(path3 + '\\' + newname)  # Save image in the corresponding path
            image_div.save(path3 + '\\' + newname2)  # Save image in the corresponding path




# # unit_result(6, 2, 0.08, False, False, False, False)
# # Finish time counter
# end_time = time.time()
# # Total time for the code
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")
