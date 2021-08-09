// RUN: vpux-translate --split-input-file --import-HWTEST %s | FileCheck %s

{
    "case_type": "ActivationKernelSimple",
    "kernel_filename": "testdata/HelloKernel.mvlib"
}

// CHECK-LABEL: module @mainModule
