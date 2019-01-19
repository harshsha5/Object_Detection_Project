# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
 
# load the image
image = cv2.imread(args["image"])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])'''


PINK_MIN = np.array([165, 50, 50],np.uint8)
PINK_MAX = np.array([180, 255, 255],np.uint8)

for i in range(1,33):
    #img=cv2.imread("/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/Trajectory1/PG1/img_"+str(i)+".jpeg")
    img=cv2.imread("/Users/harsh/Desktop/CMU_Sem_2/MRSD Project/Field_exp_data_set/Trajectory1/real-sense1/img_"+str(i)+".jpeg")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(img, PINK_MIN, PINK_MAX)
    cv2.imwrite('output2.jpg', frame_threshed)
    img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    output = cv2.bitwise_and(img, img, mask = frame_threshed)
    #img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    #cv2.imshow("Result", img)
    cv2.imshow("images", np.hstack([img, output]))
    cv2.waitKey(0)

    '''output = cv2.bitwise_and(img, img, mask = frame_threshed)
    img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    #frame_threshed=cv2.cvtColor(frame_threshed,cv2.COLOR_HSV2BGR)
    output=cv2.cvtColor(output,cv2.COLOR_HSV2BGR)
    # show the images
    #cv2.imshow("images", np.hstack([img, output]))
    #cv2.imshow('img',img)
    #cv2.imshow('frame_threshed',frame_threshed)
    #cv2.imshow('output',output)
    #cv2.waitKey(0)

    # convert the image to grayscale
    gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
 
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,127,255,0)
 
    # find contours in the binary image
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
    cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
    M = cv2.moments(c)
 
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the images
    cv2.imshow("Result", img)
    cv2.waitKey(0)'''
cv2.waitKey(0)


''' for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
 
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cv2.circle(output, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(output, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
        # display the image
        cv2.imshow("Image", img)
        cv2.imshow("Output", output)
        cv2.waitKey(0)'''