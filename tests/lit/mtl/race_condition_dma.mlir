// RUN: vpux-translate --split-input-file --import-HWTEST %s | FileCheck %s

{
    "case_type": "RaceConditionDMA",
    "network": null,
    "layer_name": null,
    "input": {
        "shape": [
            1,
            64,
            16,
            16
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 0.01,
            "zeropoint": 127,
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
            "scale": 0.01,
            "zeropoint": 127,
            "low_range": 0,
            "high_range": 63
        }
    }
}

// CHECK-LABEL: module @mainModule
