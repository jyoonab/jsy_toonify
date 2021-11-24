import cv2
import numpy as np
import time

from CUTGan import CUTGan

if __name__ == '__main__':

    cut_gan = CUTGan('./images\\4.png')

    cap = cv2.VideoCapture(0)
    while cv2.waitKey(33) < 0:
        ret, frame = cap.read()
        if ret:
            start_time = time.time()

            frame = cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            image_result = cut_gan.start_converting(frame)

            end_time = time.time()
            fps = np.round(1 / np.round(end_time - start_time, 3), 1)

            print(fps)
            cv2.imshow('video', image_result)

    cap.release()
    cv2.destroyAllWindows()
