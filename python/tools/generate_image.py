from PIL import Image
import numpy as np

def gen_image(w, h, c):
    data = np.random.uniform(0, 255, (h, w, c)).astype(np.uint8)

    img = Image.fromarray(data, 'RGB')
    img.save('test.png')

def gen_data(w, h, c):
    data = np.random.uniform(-1, 1, (h, w, c)).astype(np.double16)
    np.save('test.npy', data)

w, h, c = 32, 32, 3
gen_image(w, h, c)
