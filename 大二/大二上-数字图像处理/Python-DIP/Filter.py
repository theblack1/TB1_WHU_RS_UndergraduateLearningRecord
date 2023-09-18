import cv2 as cv
import numpy as np


def med_filter(src):
    # 边界拓宽
    borden_img = cv.copyMakeBorder(src, 1, 1, 1, 1, cv.BORDER_REPLICATE)
    shape_ = borden_img.shape
    L1 = shape_[0]
    L2 = shape_[1]

    print('正在进行中值滤波')
    new_img = np.zeros(src.shape)
    for i in range(1, L1 - 1):
        if (i % 2 == 0):
            print('\r--{}/{}--{:.2f}%--'.format(i + 1, L1-2, (i+1)*100/(L1-2)),
                  end='', flush=True)
        #elif (i % 5 == 0):
        else:
            print('\r||{}/{}||{:.2f}%||'.format(i + 1, L1-2, (i+1)*100/(L1-2)),
                  end='', flush=True)
        for j in range(1, L2 - 1):
            for c in range(3):
                new_img[i - 1][j - 1][c] = np.median([
                    borden_img[i - 1][j - 1][c],
                    borden_img[i - 1][j][c],
                    borden_img[i - 1][j + 1][c],
                    borden_img[i][j - 1][c],
                    borden_img[i][j][c],
                    borden_img[i][j + 1][c],
                    borden_img[i + 1][j - 1][c],
                    borden_img[i + 1][j][c],
                    borden_img[i + 1][j + 1][c],
                ])
    print()
    return new_img/256

def Filter(src, filter_name='med'):
    if (filter_name == 'med'):
        # 中值滤波
        output = med_filter(src)
    else:
        exit()

    return output
