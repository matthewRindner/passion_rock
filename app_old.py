import cv2
import random
import numpy as np


# cv2.IMREAD_COLOR OR -1  : Loads a color image. Any transparency of image will be neglected. this is the default value
# cv2.IMREAD_GRAYSCALE OR 0 : Loads image in grayscale mode
# cv2.IMREAD_UNCHANGED OR 1 : Loads image as such including alpha channel


# Create window with freedom of dimensions
cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)     
# cv2.setWindowProperty("test_img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

img = cv2.imread('assets/icb_2.jpg', 1)
# Resize image
# img = cv2.resize(img, (768, 512))      
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# #    
# cv2.imshow('output', img)

# # # Show image
# cv2.waitKey(0)
# # # Display the image infinitely until any keypress
# cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------

# rows, column, channels(color space aka rgb/bgr)
# [blue, green, red] 0 - 255

def imagePixels():
    # prints all pixels
    print(img.shape)

    # prints forst row
    print(img[0])

    # change the pixles in the first 100 rows
    for i in range(100):
        for j in range(img.shape[1]):
                img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyWindow('output')

# ------------------------------------------------------------------------------------
#   (0,0)   --> x-axis(width)
#   |
#   V
#   y-axis(height)

def drawShapes():
    width = img.shape[1]
    height = img.shape[0]

    print(width, height)
    img_shapes = cv2.line(img, (0,0), (width, height), (255, 0, 0), 10)
    img_shapes = cv2.line(img_shapes, (0, height), (width, 0), (0, 255, 0), 10)

    img_shapes = cv2.rectangle(img_shapes, (100, 100), (200, 200), (128, 128, 128), 4)
    img_shapes = cv2.circle(img_shapes, (int(width/2), int(height/2)), 50, (0, 0, 255), -1)

    # coordinates start form bottum left hand corner
    img_shapes = cv2.putText(img_shapes, 
                            'this is am example of drawing text',
                            (int(width/6), height-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('output', img_shapes)
    cv2.waitKey(0)
    cv2.destroyWindow('output')

# ------------------------------------------------------------------------------------
# RGB: Red, Green, Blue
# BGR: Blue, Green, red

# HSV: Hue, Saturation, Lightness/Brightness  

# Hue - (color (degrree on color wheel(0-360))
# Saturation - intensity of color, 0% (black) - 100% (red), more satuartion -> more bright the color 
# Lightness/Brightness - the darkness, amount of black vs white in a color


def color_colorDetection():

    # bgr_color = np.array([[[255, 0, 0]]])
    # x = cv2.cvtColor(bgr_color, cv2.COLOR_BGRHSBV)
    # x[0][0]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    light_blue = np.array([90, 50, 50]) 
    # rgb = 63,30,7
    dark_blue = np.array([130, 255, 255]) 

    # any color not in the rang of blue will be blacked out
    mask = cv2.inRange(img_hsv, light_blue, dark_blue)

    # bitwise_and will merge the src1 and src2 images using bitwise_operation
    # since we are not blending two differnent images together, we only provide the same src img
    mask_result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('output', img_hsv)
    cv2.waitKey(0)
    cv2.destroyWindow('output')

# ------------------------------------------------------------------------------------
def cornerDetection():
    img_chess = cv2.imread('assets/chessboard.png')
    img_chess = cv2.resize(img_chess, (0,0), fx=0.75, fy=0.75)
    img_chess_greyscale = cv2.cvtColor(img_chess, cv2.COLOR_BGR2GRAY)

    # convert to greyscale image
    img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi Corner Detector
    # prams: src_img, int of corner we want to return, 
    # min quality of corner btwn 0-1, min euclidian dist btwn two corners
    corners = cv2.goodFeaturesToTrack(img_greyscale, 500, 0.01, 10)
    corner = np.int0(corners)
    
    # ravel() flattens array
    # [[x,y]] -> [x,y]
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

    # connect the dots
    # for i in range(len(corners)):
    #     for j in range(i + len(corners)):
    #         corner1 = tuple(corners[i])
    #         corner2 = tuple(corners[j]) 
    #         cv2.line(img, corner1, corner2, (0, 0, 255), 1)

    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyWindow('output')


def tester():
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_hsvcolorspace = np.array([Hue range, Saturation range, Value range])
    # upper_hsvcolorspace = np.array([Hue range, Saturation range, Value range])
    print(img_hsv)
    cv2.imshow('output', img_hsv)
    cv2.waitKey(0)
    cv2.destroyWindow('output')

# ------------------------------------------------------------------------------------

def main():
    cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE) 
    img = cv2.imread('assets/icb_1.jpg', 1) 
    # imagePixels()
    # drawShapes()
    # color_colorDetection()
    cornerDetection()
    # tester()


if __name__ == "__main__":
    main()