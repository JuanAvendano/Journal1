"""Crack width calculation

This code aims to measure the width of the crack obtained in the previous steps. It is based on the the different papers
that perform this measurement generating a skeleton a measuring the length of the perpendicular segment that goes from
edge to edge

Initially based on the GithHub repository: https://github-com.translate.goog/Garamda/Concrete_Crack_Detection_and_Analysis_SW?_x_tr_sl=ko&_x_tr_tl=en&_x_tr_hl=es&_x_tr_pto=wapp
"""

# 4. Preprocess the frame images cropped by crack detection deep learning engine.
#    The preprocess consists of 3 stages.
#   1) Image Binarization : seperate crack section and the noncrack section.
#   2) Skeletonize : extract the central skeleton of the crack.
#   3) Edge detection : extract the edge of the crack.

#   At this stage, Image Binarization will be done.

import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage import data
from skimage.color import rgb2gray
from skimage.data import page
from skimage.filters import (threshold_sauvola)
from PIL import Image

sauvola_frames_Pw_bw = []
sauvola_frames_Pw = []

for i in range(0, len(cropped_frames)):
    img = cropped_frames[i]
    img_gray = rgb2gray(img)

    # window size와 k값은 'Concrete Crack Identification Using a UAV Incorporating Hybrid Image Processing' 논문이 제시한 값을
    # 그대로 사용하였습니다.

    # window size and k value were used without any changes from the
    # 'Concrete Crack Identification Using a UAV Incorporating Hybrid Image Processing' thesis.
    window_size_Pw = 71
    thresh_sauvola_Pw = threshold_sauvola(img_gray, window_size=window_size_Pw, k=0.42)

    binary_sauvola_Pw = img_gray > thresh_sauvola_Pw
    binary_sauvola_Pw_bw = img_gray > thresh_sauvola_Pw

    binary_sauvola_Pw_bw.dtype = 'uint8'

    binary_sauvola_Pw_bw *= 255


    # The list which saves the images after image binarization.

    sauvola_frames_Pw_bw.append(binary_sauvola_Pw_bw)
    sauvola_frames_Pw.append(binary_sauvola_Pw)

# 5. Extract the skeleton of the crack.

from skimage.morphology import skeletonize
from skimage.util import invert

skeleton_frames_Pw = []

for i in range(0, len(cropped_frames)):
    img_Pw = invert(sauvola_frames_Pw[i])

    skeleton_Pw = skeletonize(img_Pw)

    skeleton_Pw.dtype = 'uint8'

    skeleton_Pw *= 255

    # Skeletonize가 끝난 이미지를 저장하는 리스트입니다.
    # The list which saves the images after the skeletonization.
    skeleton_frames_Pw.append(skeleton_Pw)


# 6. Detect the edges of the crack.

import numpy as np
from scipy import ndimage as ndi
from skimage import feature

edges_frames_Pw = []
edges_frames_Pl = []

for i in range(0, len(cropped_frames)):
    edges_Pw = feature.canny(sauvola_frames_Pw[i], 0.09)

    edges_Pw.dtype = 'uint8'

    edges_Pw *= 255

    # Edge detection이 끝난 이미지를 저장하는 리스트입니다.
    # The list which saves the images after edge detection.
    edges_frames_Pw.append(edges_Pw)

# 7. Calculate the width of the crack.
# 1) Find skeleton using BFS
# 2) Set the direction of the crack by searching skeletion pixels which are 5 pixels away from the skeleton pixel.
# 3) Draw a perpendicular line of the direction
# 4) The perpendicular line meets the edge. The distance is calulated by counting pixels on the line.
# 5) Convert the number of pixels into real mm width, and classify the danger group.

import queue
import math

dx_dir_right = [-5, -5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5]
dy_dir_right = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1]

dx_dir_left = [5, 5, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -5]
dy_dir_left = [0, -1, -2, -3, -4, -5, -5, -5, -5, -5, -4, -3, -2, -1]

dx_bfs = [-1, -1, 0, 1, 1, 1, 0, -1]
dy_bfs = [0, 1, 1, 1, 0, -1, -1, -1]

save_result = []
save_risk = []

for k in range(0, len(skeleton_frames_Pw)):
    print('--------------''동영상 내 재생 시간 : ', (saving_bounding_boxes[k][0] // 6) * 0.25, '초', '-----------------')
    # BFS를 통해 Skeleton을 찾습니다.
    # Searching the skeleton through BFS.
    start = [0, 0]
    next = []
    q = queue.Queue()
    q.put(start)

    len_x = skeleton_frames_Pw[k].shape[0]
    len_y = skeleton_frames_Pw[k].shape[1]

    visit = np.zeros((len_x, len_y))
    crack_width_list = []

    # Skeleton pixel로 부터 균열의 진행 방향을 찾아냅니다.
    # Find out the direction of the crack from skeleton pixel.
    while (q.empty() == 0):
        next = q.get()
        x = next[0]
        y = next[1]
        right_x = right_y = left_x = left_y = -1

        if (skeleton_frames_Pw[k][x][y] == 255):
            # Skeleton을 바탕으로 균열의 진행 방향을 구합니다.
            # Estimating the direction of the crack from skeleton
            for i in range(0, len(dx_dir_right)):
                right_x = x + dx_dir_right[i]
                right_y = y + dy_dir_right[i]
                if (right_x < 0 or right_y < 0 or right_x >= len_x or right_y >= len_y):
                    right_x = right_y = -1
                    continue;
                if (skeleton_frames_Pw[k][right_x][right_y] == 255): break;
                if (i == 13): right_x = right_y = -1

            if (right_x == -1):
                right_x = x
                right_y = y

            for i in range(0, len(dx_dir_left)):
                left_x = x + dx_dir_left[i]
                left_y = y + dy_dir_left[i]
                if (left_x < 0 or left_y < 0 or left_x >= len_x or left_y >= len_y):
                    left_x = left_y = -1
                    continue;
                if (skeleton_frames_Pw[k][left_x][left_y] == 255): break;
                if (i == 13): left_x = left_y = -1

            if (left_x == -1):
                left_x = x
                left_y = y

            # acos 공식을 바탕으로 균열의 진행 방향을 각도(theta)로 나타냅니다.
            # Set the direction of the crack as angle(theta) by using acos formula
            base = right_y - left_y
            height = right_x - left_x
            hypotenuse = math.sqrt(base * base + height * height)

            if (base == 0 and height != 0):
                theta = 90.0
            elif (base == 0 and height == 0):
                continue
            else:
                theta = math.degrees(
                    math.acos((base * base + hypotenuse * hypotenuse - height * height) / (2.0 * base * hypotenuse)))

            theta += 90
            dist = 0

            # 균열 진행 방향의 수직선과 Edge가 만나면, 그 거리를 구합니다.
            # Calculate the distance if the perpendicular line meets the edge of the crack.
            for i in range(0, 2):

                pix_x = x
                pix_y = y
                if (theta > 360):
                    theta -= 360
                elif (theta < 0):
                    theta += 360

                if (theta == 0.0 or theta == 360.0):
                    while (1):
                        pix_y += 1
                        if (pix_y >= len_y):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;

                elif (theta == 90.0):
                    while (1):
                        pix_x -= 1
                        if (pix_x < 0):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;

                elif (theta == 180.0):
                    while (1):
                        pix_y -= 1
                        if (pix_y < 0):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;

                elif (theta == 270.0):
                    while (1):
                        pix_x += 1
                        if (pix_x >= len_x):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;
                else:
                    a = 1
                    radian = math.radians(theta)
                    while (1):
                        pix_x = x - round(a * math.sin(radian))
                        pix_y = y + round(a * math.cos(radian))
                        if (pix_x < 0 or pix_y < 0 or pix_x >= len_x or pix_y >= len_y):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;

                        if (theta > 0 and theta < 90):
                            if (pix_y + 1 < len_y and edges_frames_Pw[k][pix_x][pix_y + 1] == 255):
                                pix_y += 1
                                break;
                            if (pix_x - 1 >= 0 and edges_frames_Pw[k][pix_x - 1][pix_y] == 255):
                                pix_x -= 1
                                break;

                        elif (theta > 90 and theta < 180):
                            if (pix_y - 1 >= 0 and edges_frames_Pw[k][pix_x][pix_y - 1] == 255):
                                pix_y -= 1
                                break;
                            if (pix_x - 1 >= 0 and edges_frames_Pw[k][pix_x - 1][pix_y] == 255):
                                pix_x -= 1
                                break;

                        elif (theta > 180 and theta < 270):
                            if (pix_y - 1 >= 0 and edges_frames_Pw[k][pix_x][pix_y - 1] == 255):
                                pix_y -= 1
                                break;
                            if (pix_x + 1 < len_x and edges_frames_Pw[k][pix_x + 1][pix_y] == 255):
                                pix_x += 1
                                break;

                        elif (theta > 270 and theta < 360):
                            if (pix_y + 1 < len_y and edges_frames_Pw[k][pix_x][pix_y + 1] == 255):
                                pix_y += 1
                                break;
                            if (pix_x + 1 < len_x and edges_frames_Pw[k][pix_x + 1][pix_y] == 255):
                                pix_x += 1
                                break;
                        a += 1

                dist += math.sqrt((y - pix_y) ** 2 + (x - pix_x) ** 2)
                theta += 180

                # 균열의 폭을 저장하는 리스트입니다.
            # The list which saves the width of the crack.
            crack_width_list.append(dist)

        for i in range(0, 8):
            next_x = x + dx_bfs[i]
            next_y = y + dy_bfs[i]

            if (next_x < 0 or next_y < 0 or next_x >= len_x or next_y >= len_y): continue;
            if (visit[next_x][next_y] == 0):
                q.put([next_x, next_y])
                visit[next_x][next_y] = 1

    crack_width_list.sort(reverse=True)

    # 실제의 길이로 변환합니다.
    # Convert into real width.
    print(len(crack_width_list))
    if (len(crack_width_list) == 0):
        save_result.append(0)
        real_width = 0
    elif (len(crack_width_list) < 10):
        real_width = round(crack_width_list[len(crack_width_list) - 1] * 0.92, 2)
        save_result.append(real_width)
    else:
        real_width = round(crack_width_list[9] * 0.92, 2)
        save_result.append(real_width)

    print('균열 폭 : ', real_width)

    # 위험군을 분류합니다.
    # Classify the danger group.
    if (real_width >= 0.3):
        save_risk.append('상')
        print('위험군 : 상\n')
    elif (real_width < 0.3 and real_width >= 0.2):
        save_risk.append('중')
        print('위험군 : 중\n')
    else:
        save_risk.append('하')
        print('위험군 : 하\n')

# 해당 정보를 텍스트 파일에 저장합니다.
# Save those information into text files.
f1 = open("C:\\Users\\user\\Desktop\\width.txt", 'w')
f2 = open("C:\\Users\\user\\Desktop\\risk.txt", 'w')

for z in range(0, len(save_result)):
    f1.write(str(save_result[z]) + 'mm' + '\n')
f1.close()

for z in range(0, len(save_risk)):
    f2.write(str(save_risk[z]) + '\n')
f2.close()