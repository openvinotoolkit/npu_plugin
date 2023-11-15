//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=VPUX37XX --export-VPUIP -o %t %data_path_37XX%/profiling.mlir.txt
// RUN: prof_parser -b %t -m | FileCheck %s

//CHECK: {
//CHECK: profilingBuffer: {
//CHECK: sections: [ {
//CHECK: type: 1,
//CHECK: size: 128
//CHECK: }, {
//CHECK: type: 3,
//CHECK: offset: 128,
//CHECK: size: 128
//CHECK: }, {
//CHECK: type: 4,
//CHECK: offset: 256,
//CHECK: size: 368
//CHECK: }, {
//CHECK: type: 5,
//CHECK: offset: 640,
//CHECK: size: 64
//CHECK: } ],
//CHECK: size: 704
//CHECK: },
//CHECK: dmaTasks: [ {
//CHECK: name: "PROFWORKPOINT_READ",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/PROFTASKEND_13",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 0 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROFTASKEND_0",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 1 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 0 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROFTASKEND_1",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 2 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0/PROFTASKEND_2",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 3 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 2 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1/PROFTASKEND_14",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 3 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]/PROFTASKEND_3",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_0/PROFTASKEND_4",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 5 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/_cluster_1/PROFTASKEND_15",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 5 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 5 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0/PROFTASKEND_5",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 6 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 5 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1/PROFTASKEND_16",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 6 ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]/PROFTASKEND_6",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 7 ]
//CHECK: }, {
//CHECK: name: "dpuProfilingCMX2DDR0?_cluster_0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "dpuProfilingCMX2DDR0?_cluster_1",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [ 8 ],
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
//CHECK: name: "pool1?t_MaxPool/_cluster_0/PROFTASKEND_7",
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
//CHECK: name: "pool1?t_MaxPool/_cluster_1/PROFTASKEND_17",
//CHECK: sourceLocale: "Register",
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
//CHECK: updateBarriers: [ 9 ]
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
//CHECK: name: "pool1?t_MaxPool/_cluster_1/PROFTASKEND_18",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 9 ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0/PROFTASKEND_9",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1/PROFTASKEND_19",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_0/PROFTASKEND_10",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 10 ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1",
//CHECK: sourceLocale: "DDR",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/_cluster_1/PROFTASKEND_20",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [ 10 ]
//CHECK: }, {
//CHECK: name: "actshaveProfilingCMX2DDR0?_cluster_0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [ 11 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "actshaveProfilingCMX2DDR0?_cluster_1",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [ 11 ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_0/PROFTASKEND_11",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_1",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_1/PROFTASKEND_21",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_0/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_0/PROFTASKEND_12",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "dmaProfilingCMX2DDR0",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_1/PROFTASKBEGIN",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_1",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "output?_cluster_1/PROFTASKEND_22",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "dmaProfilingCMX2DDR208",
//CHECK: sourceLocale: "CMX_NN",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: }, {
//CHECK: name: "PROFWORKPOINT_READ",
//CHECK: sourceLocale: "Register",
//CHECK: waitBarriers: [  ],
//CHECK: updateBarriers: [  ]
//CHECK: } ],
//CHECK: dpuTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_0",
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 3 ],
//CHECK: updateBarriers: [ 4 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0",
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 6 ],
//CHECK: updateBarriers: [ 7 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 6 ],
//CHECK: updateBarriers: [ 7 ],
//CHECK: workloadIds: [ 0 ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_0",
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 0, 1 ]
//CHECK: }, {
//CHECK: name: "pool1?t_MaxPool/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 0, 1 ]
//CHECK: } ],
//CHECK: swTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROF_0_4_0_0",
//CHECK: waitBarriers: [ 1 ],
//CHECK: updateBarriers: [ 0 ],
//CHECK: taskType: ""
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/PROF_0_4_1_1",
//CHECK: waitBarriers: [ 1 ],
//CHECK: updateBarriers: [ 0 ],
//CHECK: taskType: ""
//CHECK: }, {
//CHECK: name: "output?PROF_0_4_2_0/cluster_0",
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: taskType: ""
//CHECK: }, {
//CHECK: name: "output?PROF_0_4_2_0/cluster_1",
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: taskType: ""
//CHECK: }, {
//CHECK: name: "output?PROF_0_4_3_1/cluster_0",
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: taskType: ""
//CHECK: }, {
//CHECK: name: "output?PROF_0_4_3_1/cluster_1",
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: taskType: ""
//CHECK: } ]
//CHECK: }
