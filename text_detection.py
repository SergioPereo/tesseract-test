import pytesseract
import numpy as np
import os
from pytesseract import Output
import cv2

custom_config_digits = r'--oem 3 --psm 6 outputbase digits'

current_dir = os.getcwd()

images_path = os.path.join(current_dir, 'images')


def path_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def canny(image):
    return cv2.Canny(image, 100, 200)

cropped_images_path = path_create(os.path.join(current_dir, 'cropped_images'))
processed_images_path = path_create(os.path.join(current_dir, 'processed_images'))

def img_list_save(img_list, path, filename):
    rows = []
    for index, val in enumerate(img_list):
        if val.size > 0:
            cv2.imwrite(os.path.join(path, (str(index) + "_" + filename)), val)
            image_string = pytesseract.image_to_string(val)
            image_string = image_string.replace('\n', '')
            rows.append(f'{str(index) + "_" + filename},{image_string}\n')
        else:
            print(f'Size is 0 for {str(index) + "_" + filename}')
    return rows

def img_process(image, filename, cropped_path, processed_path, cropped_csv_name, csv_name, is_gray, create_new_file):
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    boxes = len(d['text'])
    cropped_images = []
    img = image.copy()
    if is_gray:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for i in range(boxes):
        if int(float(d['conf'][i])) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cropped_images.append(img[y:(y+h+1),x:(x+w+1)].copy())
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(processed_path, filename), img)
    
    csv_rows = img_list_save(cropped_images, cropped_path, filename)

    if create_new_file:
        with open(cropped_csv_name, 'w') as f:
            for row in csv_rows:
                f.write(row)
        with open(csv_name, 'w') as f:
            image_string = pytesseract.image_to_string(image)
            image_string = image_string.replace('\n', '')
            f.write(f'{filename},{image_string}\n')
    else:
        with open(cropped_csv_name, 'a') as f:
            for row in csv_rows:
                f.write(row)
        with open(csv_name, 'a') as f:
            image_string = pytesseract.image_to_string(image)
            image_string = image_string.replace('\n', '')
            f.write(f'{filename},{image_string}\n')

for index, file in enumerate(os.listdir(images_path)):
    filename = os.fsdecode(file)
    image_path = os.path.join(images_path, filename)
    image = cv2.imread(image_path)

    cropped_normal_path = path_create(os.path.join(cropped_images_path, "normal"))
    processed_normal_path = path_create(os.path.join(processed_images_path, "normal"))
    img_process(image, filename, cropped_normal_path, processed_normal_path,"normal_cropped.csv", "normal.csv", False, index==0)

    img_grayscale = get_grayscale(image)
    cropped_grayscale_path = path_create(os.path.join(cropped_images_path, "grayscale"))
    processed_grayscale_path = path_create(os.path.join(processed_images_path, "grayscale"))
    img_process(img_grayscale, filename, cropped_grayscale_path, processed_grayscale_path, "grayscale_cropped.csv", "grayscale.csv", True, index==0)

    img_canny = canny(img_grayscale)
    cropped_canny_path = path_create(os.path.join(cropped_images_path, "canny"))
    processed_canny_path = path_create(os.path.join(processed_images_path, "canny"))
    img_process(img_canny, filename, cropped_canny_path, processed_canny_path, "canny_cropped.csv", "canny.csv", True, index==0)