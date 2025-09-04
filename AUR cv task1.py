import cv2
import matplotlib.pyplot as plt
import numpy as np

def convolve(image, kernel):
    
    kernel = np.flip(kernel)

    imgH,imgW = np.shape(image)
    kerH,kerW = np.shape(kernel)

    padH = kerH // 2
    padW = kerW // 2

    padding = np.pad(image, pad_width=((padH,padH),(padW,padW)), 
                    mode="constant",
                    constant_values=0)
    
    out = np.zeros((imgH,imgW))

    for i in range(imgH):
        for j in range(imgW):
            img_part = padding[i:i+kerH, j:j+kerW]
            out[i,j] = np.sum(img_part * kernel) 

    out=np.clip(out,0,255) 
    return out

    
def median_filter(img):
    
    imgH,imgW = np.shape(img)
    
    padding = np.pad(img, pad_width=((1,1),(1,1)), 
                     mode="constant", 
                     constant_values=0)
    out = np.zeros((imgH,imgW))

    for i in range(imgH):
        for j in range(imgW):
            img_part = padding[i:i+3,j:j+3]
            out[i,j] = np.median(img_part)

    return out


img = cv2.imread('D:\clock.png', cv2.IMREAD_GRAYSCALE)
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(convolve(img, np.ones((5, 5)) / 25), cmap='gray')
axes[0, 1].set_title('Box Filter')
axes[0, 1].axis('off')

axes[1, 0].imshow(convolve(img, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])),
cmap='gray')
axes[1, 0].set_title('Horizontal Sobel Filter')
axes[1, 0].axis('off')

axes[1, 1].imshow(convolve(img, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])),
cmap='gray')
axes[1, 1].set_title('Vertical Sobel Filter')
axes[1, 1].axis('off')

axes[2, 0].imshow(convolve(img, np.array([[0.0751, 0.1238, 0.0751], 
                                          [0.1238, 0.2042, 0.1238], 
                                          [0.0751, 0.1238, 0.0751]])), cmap='gray')
axes[2, 0].set_title("Gaussian Filter")
axes[2, 0].axis("off")

axes[2, 1].imshow(median_filter(img), cmap='gray')
axes[2, 1].set_title("Median Filter")
axes[2, 1].axis("off")
plt.show()
