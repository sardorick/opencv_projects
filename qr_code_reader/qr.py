import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

def imshow(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 15))
    plt.imshow(image)

def qr_read(img, path_to_save='qr_code_reader/qr_rect_new.jpg'):
    image = cv2.imread(img)
    img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    ret, threshold = cv2.threshold(img_gray, 200, 250, cv2.THRESH_BINARY_INV)
    contours, h = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse=True)
    # bounding rectangle
    x, y, w, h = cv2.boundingRect(sorted_contours[0])
    copy = image.copy()
    cv2.rectangle(copy, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(path_to_save, copy)
    imshow(copy)    

    
    decoder = cv2.QRCodeDetector()
    data, points, _ = decoder.detectAndDecode(image)
    if points is not None:
        print('QR Code detected!')
        print('Decoded data is', data)
        
parser = argparse.ArgumentParser(description='QR code reader')
parser.add_argument('test_data', type=str, help='Input path to test image')
args = parser.parse_args()
test_data = args.test_data
qr_read(test_data)