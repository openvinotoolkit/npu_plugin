//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=VPUX30XX --export-VPUIP -o %t %data_path_30XX%/profiling.mlir.txt
// RUN: prof_parser -b %t -m | FileCheck %s
//

//CHECK: {
//CHECK: profilingBuffer: {
//CHECK: sections: [ {
//CHECK: type: 1,
//CHECK: size: 2048
//CHECK: }, {
//CHECK: type: 2,
//CHECK: offset: 2048,
//CHECK: size: 72
//CHECK: }, {
//CHECK: type: 4,
//CHECK: offset: 2176,
//CHECK: size: 144
//CHECK: } ],
//CHECK: size: 2320
//CHECK: },
//CHECK: dmaTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_1_13/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_1_13",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_1_13/PROFTASKEND_0",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 0 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROFTASKEND_1",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 1 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]/PROFTASKEND_2",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 2 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0/PROFTASKEND_3",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1/PROFTASKEND_4",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_2/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_2",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_2/PROFTASKEND_5",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_3/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_3",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_3/PROFTASKEND_6",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 3 ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]/PROFTASKEND_7",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 4 ]
//CHECK: }, {
//CHECK: name: "dpuProfilingCMX2DDR0?_cluster_0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [ 5 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "dpuProfilingCMX2DDR0?_cluster_1",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "dpuProfilingCMX2DDR0?_cluster_2",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "dpuProfilingCMX2DDR0?_cluster_3",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0/PROFTASKEND_8",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1/PROFTASKEND_9",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_2/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_2",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_2/PROFTASKEND_10",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_3/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_3",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_3/PROFTASKEND_11",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/PROFTASKEND_12",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA/PROFTASKEND_13",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA/PROFTASKEND_14",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA/PROFTASKEND_15",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_unrolled_permuteDMA/PROFTASKEND_16",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/PROFTASKEND_17",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "dmaProfilingCMX2DDR0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 6 ]
//CHECK: } ],
//CHECK: dpuTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0",
//CHECK: taskId: 1,
//CHECK: numVariants: 20,
//CHECK: maxVariants: 20,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 1,
//CHECK: numVariants: 20,
//CHECK: maxVariants: 20,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 1,
//CHECK: numVariants: 20,
//CHECK: maxVariants: 20,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 1,
//CHECK: numVariants: 20,
//CHECK: maxVariants: 20,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_0",
//CHECK: taskId: 2,
//CHECK: numVariants: 12,
//CHECK: maxVariants: 12,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 2,
//CHECK: numVariants: 12,
//CHECK: maxVariants: 12,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 2,
//CHECK: numVariants: 12,
//CHECK: maxVariants: 12,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 2,
//CHECK: numVariants: 6,
//CHECK: maxVariants: 12,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [  ]
//CHECK: } ],
//CHECK: swTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROF_0",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 0 ],
//CHECK: taskType: "Convert"
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROF_1",
//CHECK: waitBarriers: [ 1 ],
//CHECK: updateBarriers: [ 2 ],
//CHECK: taskType: "Permute"
//CHECK: }, {
//CHECK: name: "output?PROF_2",
//CHECK: waitBarriers: [ 6 ],
//CHECK: updateBarriers: [  ],
//CHECK: taskType: "Convert"
//CHECK: } ]
//CHECK: }
