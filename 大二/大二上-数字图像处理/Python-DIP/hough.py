import cv2 as cv
import numpy as np

def houghline(img):
    # read image and check
    img_p = img.copy() #用于显示概率Hough变换的检测结果
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    edges = cv.Canny(binary, threshold1=50, threshold2=200)
    lines = cv.HoughLines(edges, rho = 1, theta = 1 * np.pi/180, threshold=120, srn=0, stn = 0, min_theta=1, max_theta=2)

    for i in range(len(lines)):
        rho, theta = lines[i][0][0], lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow("Hough_line", img)

    # HoughLinesP()函数
    lines_p = cv.HoughLinesP(edges, rho = 1, theta = np.pi/180, threshold = 50, minLineLength= 30, maxLineGap=10)

    for i in range(len(lines_p)):
        x_1, y_1, x_2, y_2 = lines_p[i][0]
        cv.line(img_p, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)

    print("code successful!")
    cv.imshow("Hough_line_p", img_p)

    cv.waitKey(0)
    cv.destroyAllWindows()
