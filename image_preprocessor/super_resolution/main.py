import cv2
import numpy as np
from PIL import Image
from datetime import datetime

from real_esrnet import RealESRNet

srmodel = RealESRNet('./', None, device='cuda')

test_image = Image.open('./img/1.png')
test_image_np = cv2.cvtColor(np.array(test_image), cv2.COLOR_BGR2RGB)
print("start")
start = datetime.now()
processed_image = srmodel.process(test_image_np)
print(datetime.now() - start)
processed_image = cv2.resize(processed_image, dsize=test_image_np.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)

cv2.imshow('original_image', test_image_np)
cv2.imshow('processed_image', processed_image)
cv2.waitKey(0)
