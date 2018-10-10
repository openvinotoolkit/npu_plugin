from PIL import Image
import numpy as np

def gen_image(w, h, c):
    np.random.seed(19)
    data = np.random.uniform(0, 255, (h, w, c)).astype(np.uint8)
    img = Image.fromarray(data, 'RGB')
    print("Generating random png file : ", (h, w, c), "to ", 'test.png\n')
    img.save('test.png')

def gen_data(w, h, c, post_str=""):
    np.random.seed(19)
    data = np.random.uniform(0, 1, (h, w, c)).astype(np.float16)
    #data = input_image = np.random.uniform(0, 1, inputTensorShape).astype(np.float16)
    print("Generating random file : ", (h, w, c), "to ", 'test'+post_str+'.npy\n')
    np.save('test'+post_str+'.npy', data)

