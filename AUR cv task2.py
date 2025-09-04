import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("D:\colourful_shapes.png")
if img is None:
    print("UNABLE TO LOAD IMAGE")
else:
    out = img.copy()

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    Lblue = np.array([90,50,70])
    Hblue = np.array([128,255,255])

    Lred1 = np.array([159,50,70])
    Hred1 = np.array([180,255,255])

    Lred2 = np.array([0,50,70])
    Hred2 = np.array([9,255,255])

    Lblack = np.array([0,0,0])
    Hblack = np.array([180,255,30])

    blue_to_black_mask = cv2.inRange(hsv,Lblue,Hblue)
    red1_to_blue_mask = cv2.inRange(hsv,Lred1,Hred1)
    red2_to_blue_mask = cv2.inRange(hsv,Lred2,Hred2)
    red_to_blue_mask = red1_to_blue_mask | red2_to_blue_mask
    black_to_red_mask = cv2.inRange(hsv,Lblack,Hblack)

    out[blue_to_black_mask > 0] = [0,0,0]
    out[red_to_blue_mask > 0] = [255,0,0]
    out[black_to_red_mask > 0] = [0,0,255]

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Processed Image")
    axes[1].axis("off")
    plt.show()