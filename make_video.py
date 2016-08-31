import cv2

location = 'output/'
img1 = cv2.imread(location+'0.jpg')
height, width, layers = img1.shape

video = cv2.cv.CreateVideoWriter(filename='video.avi',
                                 fourcc=cv2.cv.CV_FOURCC('F','M','P','4'),
                                 fps=5,
                                 frame_size=(width,height),
                                 is_color=1)

for i in range(100, 3800):
    fn = location + str(i) + '.jpg'
    print fn
    cv2.cv.WriteFrame(video, cv2.cv.LoadImage(fn))

del video

# cv2.destroyAllWindows()
# video.release()
