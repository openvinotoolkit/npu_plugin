//
// Copyright 2022 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "mvSubspaces.h"
#include <limits.h>
#include <cstring>

namespace subspace {

INLINE_ATTRIBUTE int getTotal(const int32_t subspaceDims[], int nDims)
{
    int totalSubspaces = 1;
    for(int i = 0; i < nDims; i++)
    {
        totalSubspaces *= subspaceDims[i];
    }
    return totalSubspaces;
}

INLINE_ATTRIBUTE void getCoord(int nSubspace, const int32_t dims[], int nDims, int32_t subspaceCoord[])
{
    for(int i = 0; i < nDims; ++i)
    {
        int nUpSubspace = nSubspace / dims[i];
        subspaceCoord[i] = nSubspace - nUpSubspace * dims[i];
        nSubspace = nUpSubspace;
    }
}

INLINE_ATTRIBUTE int getOffsetU8(const int32_t subspaceCoord[], const int32_t strides[], int nDims, const int8_t broadcast[])
{
    int offset = 0;
    for(int d = 0; d < nDims; ++d)
    {
        const int coord = (broadcast && broadcast[d]) ? 0 : subspaceCoord[d];
        offset += coord * strides[d];
    }
    return offset;
}

INLINE_ATTRIBUTE void getOffsetsU8(const int32_t subspaceCoord[], const int32_t strides1[], const int32_t strides2[],
        int nDims, unsigned& offset1, unsigned& offset2,
        const int8_t broadcast1[], const int8_t broadcast2[])
{
    offset1 = 0;
    offset2 = 0;
    for(int d = 0; d < nDims; ++d)
    {
        const int coord1 = (broadcast1 && broadcast1[d]) ? 0 : subspaceCoord[d];
        const int coord2 = (broadcast2 && broadcast2[d]) ? 0 : subspaceCoord[d];
        offset1 += static_cast<unsigned int>(coord1 * strides1[d]);
        offset2 += static_cast<unsigned int>(coord2 * strides2[d]);
    }
}

INLINE_ATTRIBUTE void getOffsetsU8(const int32_t subspaceCoord[], const int32_t strides1[], const int32_t strides2[],
        const int32_t strides3[], int nDims, unsigned& offset1, unsigned& offset2, unsigned& offset3,
        const int8_t broadcast1[], const int8_t broadcast2[], const int8_t broadcast3[])
{
    offset1 = 0;
    offset2 = 0;
    offset3 = 0;
    for(int d = 0; d < nDims; ++d)
    {
        const int coord1 = (broadcast1 && broadcast1[d]) ? 0 : subspaceCoord[d];
        const int coord2 = (broadcast2 && broadcast2[d]) ? 0 : subspaceCoord[d];
        const int coord3 = (broadcast3 && broadcast3[d]) ? 0 : subspaceCoord[d];
        offset1 += static_cast<unsigned int>(coord1 * strides1[d]);
        offset2 += static_cast<unsigned int>(coord2 * strides2[d]);
        offset3 += static_cast<unsigned int>(coord3 * strides3[d]);
    }
}

INLINE_ATTRIBUTE void increment1Coord(int32_t subspaceCoord[], const int32_t dims[], int nDims)
{
    for (int d = 0; d < nDims; ++d)
    {
        if (subspaceCoord[d] < dims[d] - 1) {
            subspaceCoord[d]++;
            return;
        }
        subspaceCoord[d] = 0;
    }
}

INLINE_ATTRIBUTE void incrementNCoord(int32_t subspaceCoord[], const int32_t dims[], int nDims, int inc)
{
    for(int d = 0; d < nDims; ++d)
    {
        inc += subspaceCoord[d];
        subspaceCoord[d] = inc % dims[d];
        inc -= subspaceCoord[d];
        inc /= dims[d];
    }
}

INLINE_ATTRIBUTE void incrementLine(int32_t lineCoord[], const int32_t dims[], int nDims, int axis)
{
    for(int d = 0, nAdd = 1; d < nDims && nAdd == 1 ; ++d)
    {
        if(d != axis)
        {
            lineCoord[d] = (lineCoord[d] == dims[d] - 1) ? 0 : lineCoord[d] + 1;
            nAdd = (lineCoord[d] == 0) ? 1 : 0;
        }
    }
}

INLINE_ATTRIBUTE void incrementPlane(int32_t planeCoord[], const int32_t dims[], int nDims, int axis0, int axis1)
{
    for(int d = 0, nAdd = 1; d < nDims && nAdd == 1 ; ++d)
    {
        if(d != axis0 && d != axis1)
        {
            planeCoord[d] = (planeCoord[d] == dims[d] - 1) ? 0 : planeCoord[d] + 1;
            nAdd = (planeCoord[d] == 0) ? 1 : 0;
        }
    }
}

INLINE_ATTRIBUTE int getTotalLines(const int32_t dims[], int nDims, int axis)
{
    return (dims[axis]) ? getTotal(dims, nDims) / dims[axis] : 0;
}

INLINE_ATTRIBUTE int getTotalPlanes(const int32_t dims[], int nDims, int axis0, int axis1)
{
    return (dims[axis0] * dims[axis1]) ? getTotal(dims, nDims) / (dims[axis0] * dims[axis1]) : 0;
}

INLINE_ATTRIBUTE int arrayElementExclude(int32_t a[], int el, int nEls)
{
    for(int i = el; i < nEls - 1; ++i)
    {
        a[i] = a[i + 1];
    }
    return nEls - 1;
}

INLINE_ATTRIBUTE int arraysElementExclude(int32_t a[], int32_t b[], int el, int nEls)
{
    for(int i = el; i < nEls - 1; ++i)
    {
        a[i] = a[i + 1];
        b[i] = b[i + 1];
    }
    return nEls - 1;
}

INLINE_ATTRIBUTE int getSizes(const int32_t subspaceDims[], int nDims, int32_t subspaceSizes[])
{
    int totalSubspaces = 1;
    for(int i = 0; i < nDims; i++)
    {
        subspaceSizes[i] = totalSubspaces;
        totalSubspaces *= subspaceDims[i];
    }
    return totalSubspaces;
}

INLINE_ATTRIBUTE bool isPermutationValid(const NDDims& perm) {
    if (perm.ndims() == 0) return false;
    int32_t trivial[MAX_ND_DIMS];
    for (int i = 0; i < perm.ndims(); i++) {
        trivial[i] = -1;
    }
    for (int i = 0; i < perm.ndims(); i++) {
        trivial[perm[i]] = perm[i];
    }
    for (int i = 0; i < perm.ndims(); i++) {
        if (trivial[i] != i) return false;;
    }
    return true;
}

INLINE_ATTRIBUTE bool isOrderNDValid(NDOrder ndOrder) {
    bool ret = false;
    orderNDToPermutation(ndOrder, ret);
    return ret;
}

INLINE_ATTRIBUTE int orderNDToNumDims(NDOrder ndOrder) {
    int i = 0;

    for (i = 0; i < MAX_ND_DIMS; i++) {
        int digit = static_cast<int>((ndOrder & 0xF) - 1);
        if (static_cast<unsigned>(digit) >= static_cast<unsigned>(MAX_ND_DIMS)) {
            break;
        }
        ndOrder >>= HEX_DIGIT_BITS;
    }
    return i;
}

INLINE_ATTRIBUTE NDDims orderNDToPermutation(NDOrder ndOrder, bool& success) {
    NDDims perm;

    for (int i = 0; i < MAX_ND_DIMS; i++) {
        int digit = static_cast<int>((ndOrder & 0xF));
        if (static_cast<unsigned>(digit - 1) >= static_cast<unsigned>(MAX_ND_DIMS)) {
            break;
        }
        if (!perm.push_back(digit - 1)) {
            success = false;
            return NDDims();
        }
        ndOrder >>= HEX_DIGIT_BITS;
    }
    if (isPermutationValid(perm)) {
        success = true;
        return perm;
    } else {
        success = false;
        return NDDims();
    }
}

INLINE_ATTRIBUTE NDDims orderNDToIndices(NDOrder ndOrder, bool& success) {
    NDDims indices;
    indices.resize(MAX_ND_DIMS);
    int num = MAX_ND_DIMS;
    int max = -1;
    for (int i = 0; i < MAX_ND_DIMS; i++) {
        int ind = static_cast<int>((ndOrder & 0xF));
        if (ind > max) {
            max = ind;
        }
        if (static_cast<unsigned>(ind - 1) >= static_cast<unsigned>(MAX_ND_DIMS)) {
            num = i;
            break;
        }
        LogDimIndex key(ind - 1);
        indices[key] = i;
        ndOrder >>= HEX_DIGIT_BITS;
    }
    if (num != max) {  // Illegal order value
        success = false;
        return NDDims();
    }

    indices.resize(num);

    if (isPermutationValid(indices)) {
        success = true;
        return indices;
    } else {
        success = false;
        return NDDims();
    }
}

INLINE_ATTRIBUTE NDOrder permutationToOrderND(const NDDims perm) {
    if (!isPermutationValid(perm)) {
        return 0;
    }
    uint64_t order = 0;
    int length = perm.ndims();
    length = (length < MAX_ND_DIMS) ? length : MAX_ND_DIMS;
    for (int sh = 0, i = 0; i < length; i++, sh += HEX_DIGIT_BITS) {
        order += (((static_cast<unsigned int>(perm[i]) + 1) & 0xF) << sh);
    }

    return order;
}

INLINE_ATTRIBUTE int arrayElementInclude(int32_t a[], int elementPos, int32_t value, int elementsCount, int maxDims)
{
    if (elementsCount + 1 > maxDims || elementPos > elementsCount)
    {
        return elementsCount;
    }
    for (int i = elementsCount; i >= elementPos + 1; --i)
    {
        a[i] = a[i - 1];
    }
    a[elementPos] = value;

    return elementsCount + 1;
}

INLINE_ATTRIBUTE int arraysElementInclude(int32_t a[], int32_t b[], int elementPos, int32_t value, int elementsCount, int maxDims)
{
    if (elementsCount + 1 > maxDims || elementPos > elementsCount)
    {
        return elementsCount;
    }
    for (int i = elementsCount; i >= elementPos + 1; --i)
    {
        a[i] = a[i - 1];
        b[i] = b[i - 1];
    }
    a[elementPos] = value;
    b[elementPos] = value;

    return elementsCount + 1;
}

INLINE_ATTRIBUTE bool alignPermutationSize(NDDims& baseLinePerm, int dimensionality) {
    int size = baseLinePerm.ndims();
    if (size > dimensionality) {
        for (int i = size - 1; i >= 0; i--) {
            if (baseLinePerm[i] >= (size - dimensionality)) {
                // decrease "junior" index
                baseLinePerm[i] -= (size - dimensionality);
            } else {
                // erase "elder" index
                if (!baseLinePerm.erase(i)) return false;
            }
        }
    } else if (size < dimensionality) {
        for (int i = 0; i < size; i++) {
            // increase "junior" index
            baseLinePerm[i] += (dimensionality - size);
        }
        for (int i = dimensionality - size - 1; i >= 0; i--) {
            // add new "elder" index
            if (!baseLinePerm.push_back(i)) return false;
        }
    }
    return true;
}

INLINE_ATTRIBUTE NDOrder extractLayoutFromShape(const long unsigned int newDims[],
    const long unsigned int newStrides[], int dimensionality, NDOrder baseLineOrder, bool& success) {
    if (baseLineOrder <= 0) {
        baseLineOrder = static_cast<NDOrder>(static_cast<uint64_t>(
                FULL_ND_ORDER) >> (HEX_DIGIT_BITS * (MAX_ND_DIMS - dimensionality)));
    }
    auto baseLinePerm = subspace::orderNDToPermutation(baseLineOrder, success);
    success &= alignPermutationSize(baseLinePerm, dimensionality);

    NDDims workPerm;
    success &= workPerm.resize(dimensionality);
    NDDims resultPerm;
    success &= resultPerm.resize(dimensionality);
    if (!success) {
        return 0;
    }

    for (int i = 0; i < dimensionality; i++) {
        workPerm[i] = baseLinePerm[i];
    }
    for (int j = 0; j < dimensionality; j++) {
        int indexOfMin = 0;
        unsigned int minStride = INT_MAX;
        for (int i = 0; i < dimensionality; i++) {
            if (workPerm[i] < 0) continue;
            if (newStrides[workPerm[i]] < minStride) {
                indexOfMin = i;
                minStride = newStrides[workPerm[indexOfMin]];
            } else if (newStrides[workPerm[i]] == minStride &&
                       newDims[workPerm[i]] < newDims[workPerm[indexOfMin]]) {
                indexOfMin = i;
                minStride = newStrides[workPerm[indexOfMin]];
            }
        }
        resultPerm[j] = workPerm[indexOfMin];
        workPerm[indexOfMin] = -workPerm[indexOfMin] - 1;
    }
    uint64_t newOrder = subspace::permutationToOrderND(resultPerm);

    return newOrder;
}

INLINE_ATTRIBUTE bool isLayoutFit(NDOrder ndOrder, const long unsigned int lDims[],
    const long unsigned int lStrides[], int dimensionality) {
    bool success = false;
    auto extracted = extractLayoutFromShape(lDims, lStrides, dimensionality, ndOrder, success);
    return (success && (extracted == ndOrder));
}

INLINE_ATTRIBUTE bool NDDims::erase(int i) {;
    if (_ndims <= 0) return false;  // Impossible to erase. No elements;
    _ndims = arrayElementExclude(_dims.data(), i, _ndims);
    return true;
}

} //namespace subspace

