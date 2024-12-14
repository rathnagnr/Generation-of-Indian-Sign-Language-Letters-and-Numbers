import os
from time import sleep
import cv2

# Read the video from specified path. Note: put the relative/Absolute path corresponding to a class of video.
cam = cv2.VideoCapture("./my_video/class_0.mp4")

# Give the output folder name for this class. Note: Need to change for all classes.
root_path = './data/class_0'

# Make a folder of given name to store frames.
try:
    # creating a folder named data
    if not os.path.exists(root_path):
        os.makedirs(root_path)
# if not created then raise error
except OSError:
    print('Error: Creating directory of data')


# Use cv2 to read all the given frame till end of video.
currentframe = 0
while currentframe < 4320:     # This varies with length of videos.
    # reading from frame
    ret, frame = cam.read()
    if ret:
        # if video is still left continue creating images
        name = 'image_' + str(currentframe//2) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images.
        store_path = root_path + '/' + name
        cv2.imwrite(store_path, frame)

        # increasing counter
        currentframe += 2      # Alternative frame.
        cam.set(cv2.CAP_PROP_POS_FRAMES, currentframe)
        
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
