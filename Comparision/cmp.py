from skimage.metrics import structural_similarity as ssim
import cv2

file1 = '../images/horse.png'
file2 = '../images/psoOutput.jpg'

im1 = cv2.imread(file1, 0)
im2 = cv2.imread(file2, 0)

w, h = im2.shape
im1 = cv2.resize(im1, (h, w))

cv2.imshow('input img', im1)
cv2.waitKey()

cv2.imshow('output img', im2)
cv2.waitKey()

print('ssim value is ', ssim(im1, im2))
