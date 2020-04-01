import cv2 as cv
import numpy as np

# Function for calculating image gradients with Sobel operators
def sobel_gradient(img):
    # Normalizing pixel values to be in the range [0,1]
    img = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # Sobel operators
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    rows, columns = np.shape(img)
    grad_x = np.zeros((rows, columns))
    grad_y = np.zeros((rows, columns))
    mag = np.zeros((rows, columns))

    for i in range(rows-2):
        for j in range(columns-2):
            grad_x[i, j] = np.sum(np.multiply(img[i:i+3, j:j+3], Gx))
            grad_y[i, j] = np.sum(np.multiply(img[i:i+3, j:j+3], Gy))
            mag[i, j] = np.sqrt(grad_x[i, j]**2+grad_y[i, j]**2)

    #cv.imshow('Gradient x', grad_x)
    #cv.imshow('Gradient y', grad_y)
    #cv.imshow('mag', mag)
    #cv.waitKey(0)
    return grad_x, grad_y, mag

# Function for finding corners using Harris detector (Shi-Thomasi method)
def Harris_detector(img, thresh):
    # Find gradients along x and y axis
    gx, gy, _ = sobel_gradient(img)
    rows, columns = img.shape
    Rs = np.zeros((rows-2, columns-2))
    corners = [] # pixel coordinates
    for i in range(1, rows-1, 1):
        for j in range(1, columns-1, 1):
            # Form matrix M = [Gx^2, GxGy; GxGy, Gy^2]
            M = np.array([[gx[i-1, j-1]**2, gx[i-1, j-1]*gy[i-1, j-1]], [gx[i-1, j-1]*gy[i-1, j-1], gy[i-1, j-1]**2]])
            # Find eigenvalues
            eigvals, _ = np.linalg.eig(M)
            # Find R
            R = min(eigvals)
            Rs[i-1, j-1] = R
            # Compare R with threshold and check if corner
            if R > thresh:
                corners.append([i, j])
    return corners, Rs


img = cv.imread('dices1.jpg', 1)
img_gray = cv.imread('dices1.jpg', 0)
# Calculating corners
corners, Rs = Harris_detector(img_gray, 1e-16)

# Plotting image with calculated corners
for i in range(len(corners)):
    img = cv.drawMarker(img, (corners[i][1], corners[i][0]), (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv.LINE_AA)
cv.imshow('corners', img)
cv.waitKey(0)
cv.destroyAllWindows()
