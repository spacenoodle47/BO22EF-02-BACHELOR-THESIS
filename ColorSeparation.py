import cv2
import numpy as np

def empty(a):
    pass

def stackImages(scale,imgArray):
    """Function taken from:
    https://www.youtube.com/watch?v=WQeoO7MI0Bs&t=1114s"""
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Function based of:
    https://stackoverflow.com/questions/44720580/resize-image-to-maintain-aspect-ratio-in-python-opencv"""
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)



#create webcam object, 0 refers to the built-in webcam, 1 for external webcam
cap = cv2.VideoCapture(1)

#create TrackBars to distinguish between HSV (color)
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)


while True:
    #read the camera, will use frame variable
    (success, frame) = cap.read()

    ###BELT###
    #get the area which we are interested in, using the points above
    #frame[row, column]

    # Modified
    belt = frame[270:340, 265:535]
    frameHSV = cv2.cvtColor(belt, cv2.COLOR_BGR2HSV)

    # store HSV min/max values
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    #----------------DEFAULT------------------
    # get lower/upper bound
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # create mask - if pixels are within lowerb/upperb it will show as white, else black
    mask = cv2.inRange(frameHSV, lower, upper)

    # check if both images detects a pixel
    frameResult = cv2.bitwise_and(belt, belt, mask=mask)
    #---------------DEFAULT--------------------

    # cv2.imshow("Video", frame)
    # cv2.imshow("Belt", belt)
    # cv2.imshow("HSV", beltHSV)

    #store lower/upper bound from HSV
    red_lower_puck = np.array([0, 111, 26])
    red_upper_puck = np.array([120, 255, 255])

    white_lower_puck = np.array([0, 0, 149])
    white_upper_puck = np.array([179, 43, 255])


    #create mask with the right HSV values
    red_puck_mask = cv2.inRange(frameHSV, red_lower_puck, red_upper_puck)
    white_puck_mask = cv2.inRange(frameHSV, white_lower_puck, white_upper_puck)

    #get contour of the masks
    (cnt_red_puck, _) = cv2.findContours(red_puck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cnt_white_puck, _) = cv2.findContours(white_puck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in cnt_white_puck:
        #get area
        area = cv2.contourArea(cnt)

        #get values from the rectangle
        (x, y, w, h) = cv2.boundingRect(cnt)

        if area > 200:
            cv2.rectangle(belt, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(belt, "White", (x, y), 1, 0.8, (255, 0, 0), 1)

    for cnt in cnt_red_puck:
        #calcuate area
        area = cv2.contourArea(cnt)

        #
        (x, y, w, h) = cv2.boundingRect(cnt)

        if area > 200:
            #print("---------")
            #print(area)
            cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(belt, "Red", (x, y), 1, 0.8, (255, 0 , 0), 1)


    frameStack = stackImages(2, ([belt, frameHSV], [mask, frameResult]))
    #belt_resize = cv2.resize(belt, (350, 140))
    belt_resize = ResizeWithAspectRatio(belt, width=550)
    #cv2.imshow("Something", frameStack)
    cv2.imshow("noe", belt_resize)

    key = cv2.waitKey(1)
    if key == ord("q"):
        print("Quit")
        break

#release the camera & destroy the windows
cap.release()
cv2.destroyAllWindows()