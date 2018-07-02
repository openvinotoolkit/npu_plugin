from PIL import Image
import numpy as np

w, h, c = 32, 32, 3
data = np.random.uniform(0, 255, (h, w, c)).astype(np.uint8)

img = Image.fromarray(data, 'RGB')
img.save('test.png')
