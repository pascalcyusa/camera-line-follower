import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
from libcamera import controls

# === GPIO Setup ===
GPIO.setmode(GPIO.BOARD)

# Motor control pins
ena1 = 12
in1 = 24
in2 = 26

ena2 = 32
in3 = 11
in4 = 13

# Setup motor pins
GPIO.setup(ena1, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(ena2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)

# Initialize motor PWM
motor1 = GPIO.PWM(ena1, 50)
motor2 = GPIO.PWM(ena2, 100)

# Motor speed
speed_1 = 14
speed_2 = 25

# Start motors at 0 speed
motor1.start(0)
motor2.start(0)

# === Camera Setup ===
picam2 = Picamera2()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()
time.sleep(1)  # Allow time for camera to initialize

# === Motor Control Functions ===
def move_forward():
    motor1.ChangeDutyCycle(speed_1)
    motor2.ChangeDutyCycle(speed_2)
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)

def stop():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)

def turn_right():
    motor1.ChangeDutyCycle(speed_1)
    motor2.ChangeDutyCycle(speed_2)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)

def turn_left():
    motor1.ChangeDutyCycle(speed_1)
    motor2.ChangeDutyCycle(speed_2)
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)

# === Main Loop for Line Following ===
try:
    while True:
        # Capture image
        image = picam2.capture_array("main")

        # Crop region of interest
        crop_img = image[60:120, 0:160]

        # Convert to grayscale
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Apply thresholding
        _, binary_img = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Find largest contour
            c = max(contours, key=cv2.contourArea)

            # Compute centroid
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Draw visual markers
                cv2.line(crop_img, (cx, 0), (cx, 720), (255, 0, 0), 1)
                cv2.line(crop_img, (0, cy), (1280, cy), (255, 0, 0), 1)
                cv2.drawContours(crop_img, [c], -1, (0, 255, 0), 2)

                # Determine movement
                if cx >= 120:
                    print("Turn Left!")
                    turn_left()
                elif 50 < cx < 120:
                    print("On Track!")
                    move_forward()
                elif cx <= 50:
                    print("Turn Right!")
                    turn_right()
            else:
                print("Centroid calculation error, retrying...")
                stop()

        else:
            print("No line detected, stopping")
            stop()

        # Display the processed frame
        cv2.imshow("Camera View", crop_img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("Stopping...")
    motor1.stop()
    motor2.stop()
    GPIO.cleanup()
