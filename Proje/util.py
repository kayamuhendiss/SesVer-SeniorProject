import cv2
import svm_train as st


# eN BÜYÜK CONTOUR ALMA
def getMaxContour(contours, minArea=200):
    maxC = None
    maxArea = minArea
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > maxArea):
            maxArea = area
            maxC = cnt
    return maxC


def getGestureImg(cnt, img, th1, model):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    imgT = img[y:y + h, x:x + w]
    imgT = cv2.bitwise_and(imgT, imgT, mask=th1[y:y + h, x:x + w])
    imgT = cv2.resize(imgT, (200, 200))
    imgTG = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
    resp = st.predict(model, imgTG)
    img = cv2.imread('DataSet/' + chr(int(resp[0]) + 64) + '_2.jpg')
    return img, chr(int(resp[0]) + 64)
