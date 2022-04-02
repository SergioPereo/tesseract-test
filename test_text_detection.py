import pytesseract
import numpy as np
import os
from pytesseract import Output
import cv2

custom_config_digits = r'--oem 3 --psm 6 outputbase digits'

current_dir = os.getcwd()

image_name = 'test4.jpeg'

img_path = os.path.join(current_dir, image_name)

processed_images_path = os.path.join(current_dir, 'processed_images')

image = cv2.imread(img_path)

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

gray = get_grayscale(image)
canny = canny(gray)

##print(pytesseract.image_to_string(image))

d = pytesseract.image_to_data(image, output_type=Output.DICT)
size = image.shape
print(size)
n_boxes = len(d['text'])
X = []
for i in range(n_boxes):
    if int(float(d['conf'][i])) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        X.append(image[y:(y+h+1), x:(x+w+1)].copy())
        #image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

for index, val in enumerate(X):
    cv2.imwrite(os.path.join(current_dir, (str(index) + image_name)), val)
    print(val.shape)
    print(pytesseract.image_to_string(val))

"""
cv2.imwrite(os.path.join(processed_images_path, image_name), image)

cv2.imshow('img', canny)
cv2.waitKey(0)
"""