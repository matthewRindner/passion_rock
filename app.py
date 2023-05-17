import cv2
import numpy as np

# https://www.youtube.com/watch?v=3D7O_kZi8-o

# def drawBoundingBoxSetUp():          
#     cv2.setMouseCallback('output', drawBoundingBox)

def getCoordinatesInImg(img):
    cv2.imshow('output', img)
    cv2.setMouseCallback('output', printCoordinates)


def printCoordinates( clickEvent, x, y, flag, params):
    print('coordinate: ({0},{1})'.format(x, y))


def colorSegmentation(img):
    # Hue - (color (degrree on color wheel(0-360))
    # Saturation - intensity of color, 0% (black) - 100% (what ever the hue is), more satuartion -> more bright the color 
    # Value - Lightness/Brightness - the darkness, amount of black vs white in a color

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define a lower and upper bound HSV color values
    # Orange: rgb(255, 195, 180)
    # 0, 30, 120 -- 140, 160, 255
    # 0, 78, 124 -- 16, 255, 255
    lower_bound = np.array([0, 60, 120])
    upper_bound = np.array([10, 255, 255])

    # only keep the orange pixels
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    result_img = cv2.bitwise_and(img, img, mask=mask) 

    
    cv2.imshow('hsv',img_hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("output", result_img)

    return result_img

def colorChannelSperations(img):

    # colorspaces transformations
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Y – Luminance or Luma component obtained from RGB after gamma correction.
    # Cr = R – Y ( how far is the red component from Luma ).
    # Cb = B – Y ( how far is the blue component from Luma ).
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # L – Lightness ( Intensity ).
    # a – color component ranging from Green to Magenta.
    # b – color component ranging from Blue to Yellow.
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    cv2.imshow('colorspace ', img_gray )
    # cv2.imshow('hsv', img_hsv)


    # channel spliting
    blue, green, red = cv2.split(img)
    hue_hsv, saturation_hsv, value_hsv = cv2.split(img_hsv)
    l, a, b = cv2.split(img_lab)
    y, u, v = cv2.split(img_YUV)
    hue_hls, lightness_hls, saturation_hls = cv2.split(img_HLS)


    # show specific channels
    # cv2.imshow('h channel', hue_hls)
    # cv2.imshow('l channel', lightness_hls)
    # cv2.imshow('s channel', saturation_hls)

    # cv2.imshow('hue_channel', hue)
    # cv2.imshow('saturation_channel', saturation)
    # cv2.imshow('value_channel', value)


    # run canny for edge detection and show edge image
    edges = performCanny(img_gray)
    cv2.imshow('channel_edges', edges)



def nothing():
    pass

def auto_canny_edge_detection(image, sigma=0.33):
    print(sigma)
    md = np.median(image)
    lower_value = int(max(0, (1.0-sigma) * md))
    upper_value = int(min(255, (1.0+sigma) * md))
    return lower_value, upper_value

def cannyEdgeDetection(img):
    # convert to grey scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

    # create Canny edges image window
    # canny_thresh = 10
    # cv2.createTrackbar('threshold', 'output', canny_thresh, 255, nothing)

    # Detect edges using canny
    # cv2.Canny(image, T_lower, T_upper, aperture_size, L2Gradient)
    # Image: Input image to which Canny filter will be applied
    # T_lower: Lower threshold value in Hysteresis Thresholding
    # T_upper: Upper threshold value in Hysteresis Thresholding
    # aperture_size: Aperture size of the Sobel filter. defalut is 3
    # L2Gradient: Boolean parameter used for more precision in calculating Edge Gradient.
    lowerThresh, upperThresh = auto_canny_edge_detection(img_blur)
    print(lowerThresh, upperThresh)

    edges_auto_thresh = cv2.Canny(img_blur, lowerThresh, upperThresh)
    edges = cv2.Canny(img_blur, 100, 200)

    cv2.imshow('canny_edges_auto_thresh', edges_auto_thresh)
    cv2.imshow('Canny_edges', edges)
    # drawAndBoxContours(edges, img)

# img must be 1 channels == np.uint8 array
def performCanny(img):

    # remove noise
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    # set canny thresholds
    lowerThresh, upperThresh = auto_canny_edge_detection(img_blur)
    print(lowerThresh, upperThresh)

    edges_auto_thresh = cv2.Canny(img_blur, lowerThresh, upperThresh)
    edges = cv2.Canny(img_blur, 100, 200)

    return edges_auto_thresh


def drawAndBoxContours(edges, input_img):
    #  Drawing contours using edges
    contours, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    img_copy = input_img.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10:
            cv2.drawContours(image=img_copy, contours=[c], contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # # cv2.imshow('output', np.hstack([img_gray, img_canny]))
    cv2.imshow('contours', img_copy)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 10:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_copy, (x-3, y-3), (x + w+3, y+ h+3), (0, 255, 0), 2)
    cv2.imshow('boxes', img_copy)
    # print(hierarchy)



def sobelEdgeDetection( img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # param2 is the kernel, kernal specifies the degree of bluring (3x3 or 5x5 grid)
    kernel = np.ones((5,5),np.float32)/25
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    cv2.imshow('GaussianBlur', img_blur)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    
    # Display Sobel Edge Detection Images
    cv2.imshow('Sobel X', sobelx)
    cv2.imshow('Sobel Y', sobely)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    

    # # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    # contours, hierarchy = cv2.findContours(image=sobelxy, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # # draw contours on the original image
    # img_copy = img.copy()
    # cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # # cv2.imshow('contour', img_copy)

    
def thresholding(img):
    # findContours() and drawContours()
    # algorithms for contour detection: CHAIN_APPROX_SIMPLE, CHAIN_APPROX_NONE

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # While finding contours, first always apply binary thresholding or Canny edge detection to the grayscale image.

    # Apply image bluring
    img_blur = cv2.medianBlur(img_gray,5)
    
    # apply binary thresholding
    # if a pixel is higher than the value of param2, set it to param3
    ret, thresh_binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    
    # Apply Adaptive Thresholding
    # Block Size - It decides the size of neighbourhood area.
    # C - It is just a constant which is subtracted from the mean or weighted mean calculated.
    # threshold value is the mean of neighbourhood area.
    thresh_adapt_mean = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    # threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
    thresh_adapt_gaussian = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,2)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    # contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    # img_copy = img.copy()
    # cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    cv2.imshow("gray", img_gray)
    cv2.imshow("thresh_binary_v=150", thresh_binary)
    cv2.imshow("thresh_adapt_mean", thresh_adapt_mean)
    cv2.imshow("thresh_adapt_gaussian", thresh_adapt_gaussian)

    
def OtsuThresholding(img):
    # convert image to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # global thresholding
    ret1, global_thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, otsu_thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_gray,(5,5),0)
    ret3, otsu_thresh_gaussian = cv2.threshold(img_gray, 0, 255,
                cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    cv2.imshow("gray", img_gray)
    cv2.imshow("thres_otsu", global_thresh)
    cv2.imshow("thres_otsu_clean", otsu_thresh_gaussian)


# -----------------------------------------------------------------------------------------------

def main():
    # Create window with freedom of dimensions
    # cv2.namedWindow('input', cv2.WINDOW_NORMAL)  
    input_img = cv2.imread('assets/icb_2.1.jpg', 1)
    input_img = cv2.resize(input_img, (370, 529))
    # input_img = cv2.imread('assets/dragon.jpeg', 1)

    cv2.imshow('canny', performCanny(input_img))
    drawAndBoxContours(performCanny(input_img), input_img)
    # colorSegmentation(input_img)
    # getCoordinatesInImg(input_img)
    # cannyEdgeDetection((input_img))
    # thresholding(input_img)
    # OtsuThresholding(input_img)
    # sobelEdgeDetection(input_img)
    # colorChannelSperations(input_img)

    
    cv2.imshow('input', input_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()
