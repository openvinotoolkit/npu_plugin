//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=false" -verify-diagnostics %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// expected-error@+1 {{CompilationMode is already defined, probably you run '--init-compiler' twice}}
module @mode attributes {VPU.compilationMode = "ReferenceSW"} {
}

// -----

// expected-error@+1 {{Architecture is already defined, probably you run '--init-compiler' twice}}
module @arch attributes {VPU.arch = "VPUX37XX"} {
}

// -----

// expected-error@+1 {{Available executor kind 'DMA_NN' was already added}}
module @executors {
    IE.ExecutorResource 2 of @DMA_NN
}

// -----

// expected-error@+1 {{Available memory kind 'CMX_NN_FragmentationAware' was already added}}
module @memory {
    IE.MemoryResource 5 bytes of @CMX_NN_FragmentationAware
}
