// RUN: vpux-translate --split-input-file --import-HWTEST %s | FileCheck %s

{
    "case_type": "Pipeline",
    "network": null,
    "layer_name": null,
    "input": {
        "shape": [
            1,
            256,
            16,
            16
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 63
        }
    },
    "weight": {
        "shape": [
            64,
            256,
            1,
            1
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 63
        }
    },
    "output": {
        "shape": [
            1,
            64,
            16,
            16
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 255
        }
    },
    "conv_op": {
        "stride": [
            1,
            1
        ],
        "pad": [
            0,
            0
        ],
        "group": 1,
        "dilation": 1
    },
    "activation": {
        "name": null
    }
}


// CHECK-LABEL: module @mainModule
