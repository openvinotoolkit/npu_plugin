import numpy as np

# Generating input image and trasposing it to ZMajor
input_channels = 3
input_width = 4
input_height = 4

# Fixing the seed to get consistent results and not have
# to change each time unfusable_scale.cpp
np.random.seed(5)

input_shape = (input_channels,input_height,input_width)
input_size = input_shape[0] * input_shape[1] * input_shape[2]

image = np.random.randint(256, size=input_size).reshape(input_shape).astype(np.uint8)
image_zmaj = image.transpose((1,2,0))

image_zmaj.tofile('unfusable_scale.in')

# NOTE: 3 values hardcoded from resnet
scales = np.array([0.0174292, 0.017507, 0.0171248])
shifts = np.array([-1.80444, -2.03571, -2.1179])

# Performing scale shift
image_proc = image.astype(np.float)
for i in range(len(scales)):
    image_proc[i] *= scales[i]
    image_proc[i] += shifts[i]

result_zmaj = image_proc.transpose((1,2,0))
result_zmaj.tofile('unfusable_scale.out')

# Computing output tensor quantization parameters

output_max = np.max(result_zmaj)
output_min = np.min(result_zmaj)
output_range = output_max - output_min
output_positive_range = output_max if output_max > 0 else 0
quantization_levels = 255
X = np.round((output_positive_range / output_range) * quantization_levels)

quant_scale_output = output_range / quantization_levels
quant_zero_point_output = 255 - X

print(str(input_width) + "x" + str(input_height) + "x" + str(input_channels) + " input image")
print("Quantization parameters for input")
print(0)
print(1.0)
print("Quantization parameters for scale weights")
print(0)
print(scales)
print("Quantization parameters for bias")
print(0)
print(scales)
print("Quantized weights")
print(scales/scales)
print("Quantized biases")
print(np.round(shifts/scales))
print("Quantization parameters for output")
print(int(quant_zero_point_output))
print(quant_scale_output)
