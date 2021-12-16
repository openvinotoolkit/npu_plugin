// RUN: vpux-translate --import-HWTEST %s

{
    "case_type": "DepthWiseConv",
    "input": {
        "shape": [
            1,
            16,
            32,
            32
        ],
        "dtype": "fp16",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 1
        }
    },
    "weight": {
        "shape": [
            16,
            1,
            4,
            4
        ],
        "dtype": "fp16",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 1
        }
    },
    "output": {
        "shape": [
            1,
            16,
            29,
            29
        ],
        "dtype": "fp16",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 1
        }
    },
    "conv_op": {
        "stride": [
            1,
            1
        ],
        "pad": [
            0,
            0,
            0,
            0
        ],
        "group": 16,
        "dilation": 1
    },
    "output_order": "nhwc",
    "activation": {
        "name": null
    }
}
