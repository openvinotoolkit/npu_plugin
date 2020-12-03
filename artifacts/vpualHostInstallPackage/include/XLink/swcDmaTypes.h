///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      swcDmaTypes.h
/// 
/// @copyright All code copyright Movidius Ltd 2012, all rights reserved
///            For License Warranty see: common/license.txt
///
/// @brief     DMAtypes
///

#ifndef __SWCDMATYPES_H___
#define __SWCDMATYPES_H___

typedef enum dma_task{
    DMA_TASK_0 = 0,
    DMA_TASK_1 = 1,
    DMA_TASK_2 = 2,
    DMA_TASK_3 = 3
}swcDmaTask_t;

/* defines used to calculate dma internal registers offsets */
/* 32 bit defines */
#define    DMA_DISABLE          0x00
#define    DMA_ENABLE           0x01
#define    DMA_DST_USE_STRIDE   0x02
#define    DMA_SRC_USE_STRIDE   0x04
#define    DMA_TX_FIFO_ADDR     0x08
#define    DMA_RX_FIFO_ADDR     0x10
#define    DMA_FORCE_SRC_AXI    0x20

typedef enum {
    NONBLOCKING = 0,
    BLOCKING = 1
}swcDmaTransfer_t;

#endif //__SWCDMATYPES_H___
