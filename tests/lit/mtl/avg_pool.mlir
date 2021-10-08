// RUN: vpux-translate --mlir-elide-elementsattrs-if-larger=16 --import-HWTEST %s | FileCheck %s

{
    "case_type": "AvgPool",
    "input": {
        "shape": [
            1,
            64,
            32,
            32
        ],
        "dtype": "fp16",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": -64,
            "high_range": 63
        }
    },
    "pool_op": {
        "sub_type": "avg",
        "kernel_shape": [
            2,
            2
        ],
        "stride": [
            2,
            2
        ],
        "pad": [
            0,
            0,
            0,
            0
        ]
    },
    "output": {
        "shape": [
            1,
            64,
            16,
            16
        ],
        "dtype": "fp16",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": -65536,
            "high_range": 65535
        }
    },
    "activation": {
        "name": null
    }
}

// CHECK-LABEL: module @mainModule
