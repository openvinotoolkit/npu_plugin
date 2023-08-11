import numpy as np
import scipy
from scipy import ndimage
from scipy import special
from scipy import signal


def compare_CPU_VPU(cpu_filename, kmb_filename, N, C, H, W):

    cpu_img = np.fromfile(cpu_filename, np.float16)
    cpu_img = cpu_img.astype(np.float16)
    cpu_img = cpu_img.astype(np.float32)
    kmb_img = np.fromfile(kmb_filename, np.float32)
    if len(kmb_img) == len(cpu_img) // 2:
        kmb_img = np.fromfile(kmb_filename, np.float16)
        kmb_img = kmb_img.astype(np.float32)

    # cpu_img = scipy.special.logit(cpu_img)
    # kmb_img = scipy.special.logit(kmb_img)
    # cpu_img = 1/(1 + np.exp(-cpu_img))
    # kmb_img = 1/(1 + np.exp(-kmb_img))
    levels = 256
    input_low = -2.5439453125
    output_low = -2.5439453125
    input_high = 2.5240707397460938
    output_high = 2.5240707397460938
    # cpu_img = np.round((cpu_img - input_low) / (input_high - input_low) * (levels-1)) / (levels-1) * (output_high - output_low) + output_low
    # kmb_img = np.round((kmb_img - input_low) / (input_high - input_low) * (levels-1)) / (levels-1) * (output_high - output_low) + output_low

    kmb_img[np.isnan(kmb_img)] = 0
    kmb_img[np.isneginf(kmb_img)] = -65000
    kmb_img[np.isinf(kmb_img)] = 65000
    kmb_img[kmb_img < -65000] = -65000
    kmb_img[kmb_img > 65000] = 65000
    cpu_img[np.isnan(cpu_img)] = 0
    cpu_img[np.isneginf(cpu_img)] = -65000
    cpu_img[np.isinf(cpu_img)] = 65000
    cpu_img[cpu_img < -65000] = -65000
    cpu_img[cpu_img > 65000] = 65000
    # kmb_img = np.clip(kmb_img, -64, 64)
    # cpu_img = np.clip(cpu_img, -64, 64)

    # print("KMB: min {min} max {max}".format(min=np.min(kmb_img), max=np.max(kmb_img)))
    # print("CPU: min {min} max {max}".format(min=np.min(cpu_img), max=np.max(cpu_img)))
    v1 = ((np.inner(cpu_img, kmb_img)))
    v2 = ((np.linalg.norm(cpu_img)))
    v3 = ((np.linalg.norm(kmb_img)))
    # print(v1/(v2*v3))

    columns = 1
    rows = 1
    max_columns = 20
    for columns in range(1, max_columns+1):
        rows = int(round((columns*9.0)/18.0))
        if columns*rows >= C:
            break

    cpu_img = cpu_img.reshape((W, H, C), order='F')  # C F A
    cpu_img = np.fliplr(np.rot90(np.rot90(np.rot90(cpu_img))))

    kmb_img1 = kmb_img.reshape((W, H, C), order='F')  # C F A
    kmb_img1 = np.fliplr(np.rot90(np.rot90(np.rot90(kmb_img1))))

    kmb_img2 = kmb_img.reshape((W, H, C), order='C')  # C F A
    kmb_img2 = np.fliplr(np.rot90(np.rot90(np.rot90(kmb_img2))))

    kmb_img3 = kmb_img.reshape((H, W, C), order='F')  # C F A
    kmb_img4 = kmb_img.reshape((H, W, C), order='C')  # C F A

    d1 = np.sum(np.abs(cpu_img - kmb_img1))
    d2 = np.sum(np.abs(cpu_img - kmb_img2))
    d3 = np.sum(np.abs(cpu_img - kmb_img3))
    d4 = np.sum(np.abs(cpu_img - kmb_img4))

    # kmb_img += np.random.random(size=kmb_img.shape)*0.1 - 0.05

    # print("Diff1 cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(cpu_img-kmb_img1), max=np.max(cpu_img-kmb_img1), avg=np.average(cpu_img-kmb_img1), sa=(np.square(cpu_img-kmb_img1)).mean(axis=None)))
    # print("Diff2 cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(cpu_img-kmb_img2), max=np.max(cpu_img-kmb_img2), avg=np.average(cpu_img-kmb_img2), sa=(np.square(cpu_img-kmb_img2)).mean(axis=None)))
    # print("Diff3 cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(cpu_img-kmb_img3), max=np.max(cpu_img-kmb_img3), avg=np.average(cpu_img-kmb_img3), sa=(np.square(cpu_img-kmb_img3)).mean(axis=None)))
    # print("Diff4 cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(cpu_img-kmb_img4), max=np.max(cpu_img-kmb_img4), avg=np.average(cpu_img-kmb_img4), sa=(np.square(cpu_img-kmb_img4)).mean(axis=None)))

    if d1 <= d2 and d1 <= d3 and d1 <= d4:
        kmb_img = kmb_img1
        # print("Diff1 choosed")
    if d2 <= d1 and d2 <= d3 and d2 <= d4:
        kmb_img = kmb_img2
        # print("Diff2 choosed")
    if d3 <= d1 and d3 <= d2 and d3 <= d4:
        kmb_img = kmb_img3
        # print("Diff3 choosed")
    if d4 <= d1 and d4 <= d2 and d4 <= d3:
        kmb_img = kmb_img4
        # print("Diff4 choosed")

    sx = ndimage.sobel(cpu_img, axis=0, mode="constant")
    sy = ndimage.sobel(cpu_img, axis=1, mode="constant")
    sobel_cpu_img = np.hypot(sx, sy)
    sx = ndimage.sobel(kmb_img, axis=0, mode="constant")
    sy = ndimage.sobel(kmb_img, axis=1, mode="constant")
    sobel_kmb_img = np.hypot(sx, sy)
    # cpu_img = sobel_cpu_img
    # kmb_img = sobel_kmb_img
    sobel_diff = sobel_cpu_img - sobel_kmb_img
    diff_img = cpu_img - kmb_img
    lowest = min(np.min(cpu_img), np.min(kmb_img), 0)
    n_cpu_img = cpu_img - lowest
    n_kmb_img = kmb_img - lowest
    highest = max(np.max(n_cpu_img), np.max(n_kmb_img))
    n_cpu_img = n_cpu_img / highest
    n_kmb_img = n_kmb_img / highest

    # diff2 = diff_img*diff_img
    # print("Diff cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(diff_img), max=np.max(diff_img), avg=np.average(diff_img), sa=(np.square(diff_img)).mean(axis=None)))
    v1 = ((np.inner(cpu_img.flatten(), kmb_img.flatten())))
    v2 = ((np.linalg.norm(cpu_img.flatten())))
    v3 = ((np.linalg.norm(kmb_img.flatten())))

    return (v1/(v2*v3),
            1-(np.sum(np.abs(diff_img))/(W*H*C))/(np.max(cpu_img)-np.min(cpu_img)),
            1-np.sqrt(np.sum(diff_img*diff_img)/(W*H*C))/(np.max(cpu_img)-np.min(cpu_img)),
            np.mean(2*n_cpu_img*n_kmb_img/(n_cpu_img*n_cpu_img + n_kmb_img*n_kmb_img + 0.000001)),
            np.sum(np.minimum(n_cpu_img, n_kmb_img))/np.sum(np.maximum(n_cpu_img, n_kmb_img)),
            (W*H*C*np.sum(cpu_img*kmb_img) - np.sum(cpu_img)*np.sum(kmb_img))/(np.sqrt(W*H*C*np.sum(cpu_img*cpu_img) -
                                                                                       np.sum(cpu_img)*np.sum(cpu_img))*np.sqrt(W*H*C*np.sum(kmb_img*kmb_img) - np.sum(kmb_img)*np.sum(kmb_img))),
            scipy.signal.correlate2d(cpu_img[:, :, 0], kmb_img[:, :, 0], mode='valid')[
        0, 0] / (1*1), np.var(diff_img)
    )
