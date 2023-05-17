import cv2
import numpy as np

def nothing(x):
    pass


# # create trackbars for HSV color masking
cv2.namedWindow("hsv_Tracking")
cv2.createTrackbar("LH", "hsv_Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "hsv_Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "hsv_Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "hsv_Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "hsv_Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "hsv_Tracking", 255, 255, nothing)

# # # create trackbars for Canny edge
cv2.namedWindow('Canny_Tracking')
cv2.createTrackbar('lowerB','Canny_Tracking',0,1000,nothing)
cv2.createTrackbar('upperB','Canny_Tracking',1000,1000,nothing)

# # # create trackbars for binary threshold
# cv2.namedWindow('binary_threshold')
# cv2.createTrackbar('lowerB','Canny_Tracking',0,1000,nothing)
# cv2.createTrackbar('upperB','Canny_Tracking',1000,1000,nothing)

while True:
    cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE) 
    img = cv2.imread('assets/icb_2.jpg')
    # -----------------------------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "hsv_Tracking")
    l_s = cv2.getTrackbarPos("LS", "hsv_Tracking")
    l_v = cv2.getTrackbarPos("LV", "hsv_Tracking")
    u_h = cv2.getTrackbarPos("UH", "hsv_Tracking") 
    u_s = cv2.getTrackbarPos("US", "hsv_Tracking")
    u_v = cv2.getTrackbarPos("UV", "hsv_Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("mask", mask)
    cv2.imshow("res", result)

    # -----------------------------------------------------------------
    minVal = cv2.getTrackbarPos("lowerB", "Canny_Tracking")
    maxVal = cv2.getTrackbarPos("upperB", "Canny_Tracking")

    img_canny = cv2.Canny(img, minVal, maxVal)

    cv2.imshow("canny", img_canny)
    # -----------------------------------------------------------------
    cv2.imshow('output', img)
    
    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()
print(type(cv2.waitKey(0)))