//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <mv_types.h>
#include <array>
#include <algorithm>
#include "mvSubspaces.h"

namespace subspace
{
enum {
    MAX_DIMS = 8,
};

enum Dim : int32_t { W = 0, H = 1, C = 2, N = 3, N5 = 4, N6 = 5, N7 = 6, N8 = 7};

typedef enum : uint32_t
{
    orderNHWC = 0x4213,
    orderNHCW = 0x4231,
    orderNCHW = 0x4321,
    orderHWC  = 0x213,
    orderCHW  = 0x321,
    orderWHC  = 0x123,
    orderHCW  = 0x231,
    orderWCH  = 0x132,
    orderCWH  = 0x312,
    orderNC   = 0x43,
    orderCN   = 0x34,
    orderC    = 0x3,
    orderH    = 0x2,
    orderW    = 0x1,

    orderYXZ  = orderHWC,
    orderZYX  = orderCHW,
    orderXYZ  = orderWHC,
    orderYZX  = orderHCW,
    orderXZY  = orderWCH,
    orderZXY  = orderCWH,
    orderNYXZ = orderNHWC,
    orderNYZX = orderNHCW,
    orderNZYX = orderNCHW,
    FULL_ORDER = 0x87654321
} StorageOrder;

//// NCHW - standard order: W corresponds to 0,  H to 1, C to 2 ...
//// permutation array (perm):
////      perm[i] contains:
////          - index of dimension (0..MAX_DIMS-1) in standard order on i-th place ();
////          - UNDEF if there is no dimension
////      Examples: NCHW order corresponds to perm = {0, 1, 2, 3, UNDEF, UNDEF, UNDEF, UNDEF}
////                HWC order corresponds to  perm = {2, 0, 1, UNDEF, UNDEF, UNDEF, UNDEF, UNDEF}
////
//// Indices array (mapping)
////      indices[i] contains:
////          - index (0..MAX_DIMS-1) of standard order i-th dimension in real order;
////          - UNDEF if there is no dimension
////      Examples: NCHW order corresponds to indices = {0, 1, 2, 3, UNDEF, UNDEF, UNDEF, UNDEF}
////                HWC order corresponds to  indices = {1, 2, 0, UNDEF, UNDEF, UNDEF, UNDEF, UNDEF}
////
//// Order value
////      number, i-th hexadecimal digit of which, equals (perm[i] + 1)
////      0 - there is no dimension (UNDEF)
////      1 - 8 ( 1 - 4 for NCHW) ((hexadecimal number of dimension) + 1)
typedef uint32_t t_D8StorageOrder;

// Validation conditions:
//      order should not contain one digit<>0 more than once
//      order length - amount of nonzero hexadecimal digit in order value
//      all digits on positions up to the order length should be defined
//      all digits on positions upper or equal to the order length should be UNDEF

bool isOrderValid(t_D8StorageOrder order);
bool isOrderValid(NDOrder order) = delete;

int orderToNumDims(t_D8StorageOrder order);
int orderToNumDims(NDOrder order) = delete;
int orderNDToNumDims(t_D8StorageOrder order) = delete;

int orderToPermutation(t_D8StorageOrder order, int32_t perm[]);
int orderToPermutation(NDOrder order, int32_t perm[]) = delete;
int orderToIndices(t_D8StorageOrder order, int32_t indices[]);
int orderToIndices(NDOrder order, int32_t indices[]) = delete;

NDDims orderNDToPermutation(t_D8StorageOrder order, bool& success) = delete;
NDDims orderNDToIndices(t_D8StorageOrder order, bool& success) = delete;
t_D8StorageOrder permutationToOrder(const int32_t perm[], int length = subspace::MAX_DIMS);

static inline t_D8StorageOrder maskOrder(t_D8StorageOrder fullOrder, int nOrd) {
    return static_cast<t_D8StorageOrder>(fullOrder & (0xffffffffu >> ((MAX_DIMS - nOrd) * HEX_DIGIT_BITS)));
}

NDOrder orderToNDOrder(t_D8StorageOrder order);
NDOrder orderToNDOrder(NDOrder order) = delete;

// works only up to 8D
t_D8StorageOrder NDorderToOrder(NDOrder ndOrder, bool& success);
t_D8StorageOrder NDorderToOrder(t_D8StorageOrder order, bool& success) = delete;

NDOrder extractLayoutFromShape(const long unsigned int newDims[],
    const long unsigned int newStrides[], int dimensionality, t_D8StorageOrder baseLineNDOrder, bool& success) = delete;

bool isLayoutFit(t_D8StorageOrder order, const long unsigned int lDims[],
    const long unsigned int lStrides[], int dimensionality) = delete;

//
// getDimName returns symbol-name for dimNum dimension

uint8_t getDimName(Dim dimNum);

//
// getDimNameOfOrder returns symbol-name for the dimNum-th 'memory ordered' dimension in correspondence with order value
// for example, if order = orderNHWC and dimNum = 1 the function returns 'W',
//          but if order = orderNCHW and dimNum = 1 the function returns 'H'
uint8_t getDimNameOfOrder(int32_t dimNum, uint32_t order);

}  // namespace subspace
