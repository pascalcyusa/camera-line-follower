import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls
import time

# Initialize Pi Camera
picam2 = Picamera2()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()
time.sleep(1)  # Give camera time to start up

# Define color range for line detection (adjust these values for your specific color)
lower_color = np.array([5, 200, 50])  # Lower HSV threshold
upper_color = np.array([10, 255, 70])  # Upper HSV threshold

try:
    while True:
        # Capture image from camera
        image = picam2.capture_array("main")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('img', image_rgb)
        
        # Crop the image
        crop_img = image[60:120, 0:160]
	
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(crop_img, (5, 5), 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Debug: Show HSV values
        hsv_mean = np.mean(hsv, axis=(0, 1))  # Calculate the mean HSV of the cropped region
        print(f"Mean HSV values: {hsv_mean}")  # Print the mean HSV values

        # Threshold the image to keep only the selected color
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Show HSV and Mask images for debugging
        cv2.imshow('hsv', hsv)  # Show the HSV image
        cv2.imshow('mask', mask)  # Show the thresholded mask

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Contours found: {len(contours)}")  # Debug contour detection

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)  # Get the largest contour
            M = cv2.moments(c)  # Compute moments
            
            if int(M['m00']) != 0:
                cx = int(M['m10']/M['m00'])  # Centroid x position
                cy = int(M['m01']/M['m00'])  # Centroid y position
            else:
                print("Centroid calculation error, looping to acquire new values")
                continue
            
            # Draw centroid and contours
            cv2.line(crop_img, (cx, 0), (cx, 720), (255, 0, 0), 1)  # Vertical line at centroid
            cv2.line(crop_img, (0, cy), (1280, cy), (255, 0, 0), 1)  # Horizontal line at centroid
            cv2.drawContours(crop_img, contours, -1, (0, 255, 0), 2)
            
            # Determine movement instructions
            if cx >= 120:
                print("Turn Left!")
            elif 50 < cx < 120:
                print("On Track!")
            else:
                print("Turn Right")
        else:
            print("I don't see the line")
        
        # Display the resulting frame
        cv2.imshow('frame', crop_img)
        cv2.waitKey(1)
        
except KeyboardInterrupt:
    print('All done')

