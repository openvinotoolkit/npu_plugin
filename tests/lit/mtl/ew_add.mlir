// RUN: vpux-translate --mlir-elide-elementsattrs-if-larger=16 --import-HWTEST %s | FileCheck %s

{
    "case_type": "EltwiseAdd",
    "input": {
        "shape": [
            1,
            256,
            16,
            16
        ],
        "dtype": "fp16",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": -64,
            "high_range": 63
        }
    },
    "weight": {
        "shape": [
            1,
            256,
            16,
            16
        ],
        "dtype": "fp16",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": -64,
            "high_range": 63
        }
    },
    "output": {
        "shape": [
            1,
            256,
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