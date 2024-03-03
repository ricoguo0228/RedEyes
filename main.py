import cv2
import numpy as np
from Ala import Ala

global img, ans, url_name
global point1, point2
def on_mouse(event, x, y, flags, param):
    global img, ans, point1, point2, url_name
    img1 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        cv2.circle(img1, point1, 10, (0, 255, 0), 1)
        cv2.imshow('image', img1)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(img1, point1, (x, y), (255, 0, 0), 1)
        cv2.imshow('image', img1)
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        cv2.rectangle(img1, point1, point2, (0, 0, 255), 1)
        cv2.imshow('red_eyes', img1)
        img2 = ans.astype(np.int32)
        ala = Ala(img2[:, :, 2], img2[:, :, 1], img2[:, :, 0])
        R, G, B = ala.remove_red_eyes(point1, point2)
        mid = np.zeros([R.shape[0], R.shape[1], 3])
        mid[:, :, 0] = B
        mid[:, :, 1] = G
        mid[:, :, 2] = R
        ans = mid.astype(np.uint8)
        cv2.imshow('removed_red_eyes', ans)
        cv2.imwrite('./result/'+url_name+'_fake.jpg', ans)


def main():
    global img, ans, url_name
    url_road = './resource'
    url_name = '2'
    url = url_road+'/'+url_name+'.jpg'
    img = cv2.imread(url)
    ans = img.copy()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', ans)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
