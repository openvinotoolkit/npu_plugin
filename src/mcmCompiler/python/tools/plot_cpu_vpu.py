import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

alpha = 1.0

if not ((len(sys.argv) == 9) or (len(sys.argv) == 10)):
    print('usage: python3 plot_cpu_vpu.py cpu.bin vpu.bin W H C zp scale dtype [leakyrelu_alpha]')
    exit(1)

if sys.argv[8] not in ['FP16','U8']:
    print('error: invalid dtype!  Must be FP16 or U8')
    exit(1)

dtype = sys.argv[8]
if dtype not in ['FP16','U8']:
    print('error: invalid dtype!  Must be FP16 or U8')
    exit(1)

cpu_filename, vpu_filename = sys.argv[1:3]
W, H, C, zp = [int(x) for x in sys.argv[3:7]]
scale = float(sys.argv[7])

if len(sys.argv) == 10:
    alpha = float(sys.argv[9])

cpu = np.fromfile(cpu_filename, np.float32)
if alpha != 1:
    for i in range(len(cpu)):
        if cpu[i] < 0:
            cpu[i] = cpu[i] * alpha

if dtype == 'FP16':
    vpu = np.fromfile(vpu_filename, np.float16).astype(np.float32)
else:
    vpu = np.fromfile(vpu_filename, np.uint8)
    print('zp:', zp, 'scale:', scale)
    vpu = (vpu.astype(np.float32) - zp) * scale
vpu = vpu.reshape((H,W,C)).transpose((2,0,1)).flatten()
delta = vpu - cpu
print('vpu min/max:', vpu.min(), vpu.max())
print('cpu min/max:', cpu.min(), cpu.max())
vpu[:8] ; cpu[:8] ; delta.min(), delta.max()

end_idx = len(vpu)
cpu_plot, = plt.plot(cpu[:end_idx], label='cpu')
vpu_plot, = plt.plot(vpu[:end_idx], label='vpu', color='orange')
plt.legend(handles=[cpu_plot, vpu_plot])
plt.show()
