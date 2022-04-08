//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#ifndef MV_SUBSPACES_H_
#define MV_SUBSPACES_H_
#include <mv_types.h>
#include <array>
#include <algorithm>
#include "common_types.h"

#ifndef INLINE_ATTRIBUTE
# ifdef CONFIG_ALWAYS_INLINE
#  define INLINE_ATTRIBUTE inline __attribute((always_inline))
# else
#  define INLINE_ATTRIBUTE
# endif
#endif

using namespace sw_params;

namespace subspace
{
enum {
    UNDEF = -1,
    HEX_DIGIT_BITS = 4,
};

struct LogDimIndex {
    INLINE_ATTRIBUTE LogDimIndex(int index) {
        _ind = std::min<int>(std::max(index, 0), MAX_ND_DIMS);
    }
    INLINE_ATTRIBUTE int val() const {return _ind;};
    INLINE_ATTRIBUTE LogDimIndex &operator =(const int index) {
        _ind = std::min<int>(std::max(index, 0), MAX_ND_DIMS);
        return *this;
    };
    INLINE_ATTRIBUTE bool operator<(const LogDimIndex b) const {return val() < b.val();};
    INLINE_ATTRIBUTE bool operator==(const LogDimIndex b) const {return val() == b.val();};
private:
    int32_t _ind = 0;
};

struct NDDims {
    INLINE_ATTRIBUTE int ndims() const {return _ndims;};
    INLINE_ATTRIBUTE int32_t *data(){return _dims.data();};
    INLINE_ATTRIBUTE bool resize(int newNDims) {
        if (newNDims >= 0 && newNDims <= MAX_ND_DIMS) {  // Non-negative up to 15 dimensionality is only supported
            _ndims = newNDims;
            return true;
        } else {
            return false;
        }
    }
    INLINE_ATTRIBUTE bool push_back(int32_t value) {
        if (_ndims >= MAX_ND_DIMS) return false;  // Impossible to add more than MAX_ND_DIMS == 15 elements
        _dims[_ndims++] = value;
        return true;
    }
    INLINE_ATTRIBUTE bool erase(int i);
    INLINE_ATTRIBUTE int32_t& operator[] (LogDimIndex l) {
        int i = l.val();
        return _dims[i];
    }
    INLINE_ATTRIBUTE int32_t operator[] (LogDimIndex l) const {
        int i = l.val();
        return _dims[i];
    }
    INLINE_ATTRIBUTE int32_t getElement(int i, int32_t defVal) const {
        if (i < 0 || i >= _ndims) {
            return defVal;
        } else return _dims[i];
    }
    INLINE_ATTRIBUTE int32_t getElement(LogDimIndex l, int32_t defVal) const {
        int i = l.val();
        return this->getElement(i, defVal);
    }
private:
    int _ndims = 0;
    std::array<int32_t, MAX_ND_DIMS> _dims;
};

// each element of (reduced) 'tensor' with 'subspaceDims' dimensions
// represents one 1D or 2D section of some original tensor
//
// getTotal returns common number of sections(cuts)
// subspaceDims - sizes of dimensions (in)
// nDims - dimensionality (in)
//int getTotal(const int32_t subspaceDims[], int nDims);
INLINE_ATTRIBUTE int getTotal(const int32_t subspaceDims[], int nDims);

// getCoord uses number of section to calculate coordinates of section
// nSubspace - number of section (in)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// subspaceCoord - coordinates of section (out)
INLINE_ATTRIBUTE void getCoord(int nSubspace, const int32_t dims[], int nDims, int32_t subspaceCoord[]);

// getOffsetU8 uses coordinates of the section and strides to calculate offset (in bytes)
// from beginning of original tensor to beginning of section
// subspaceCoord - coordinates of section (in)
// strides - strides (in)
// nDims - dimensionality (in)
// broadcast - broadcast flags, by dimensions (0=normal, 1=broadcasted)
// returns offset
INLINE_ATTRIBUTE int getOffsetU8(const int32_t subspaceCoord[], const int32_t strides[], int nDims, const int8_t broadcast[] = nullptr);

// getOffsetsU8 uses coordinates of the section and strides to calculate offsets (in bytes)
// from beginning of two original tensors to beginning of two corresponding sections
// subspaceCoord - coordinates of section (in)
// strides1 - strides of 1st tensor (in)
// strides2 - strides of 2nd tensor (in)
// nDims - dimensionality (in)
// broadcast1 - broadcast flags for 1st tensor, by dimensions (0=normal, 1=broadcasted)
// broadcast2 - broadcast flags for 2nd tensor, by dimensions (0=normal, 1=broadcasted)
// offset1 offset in 1st tensor (out)
// offset2 offset in 2nd tensor (out)
INLINE_ATTRIBUTE void getOffsetsU8(const int32_t subspaceCoord[], const int32_t strides1[], const int32_t strides2[],
        int nDims, unsigned& offset1, unsigned& offset2,
        const int8_t broadcast1[] = nullptr, const int8_t broadcast2[] = nullptr);

// getOffsetsU8 uses coordinates of the section and strides to calculate offsets (in bytes)
// from beginning of three original tensors to beginning of three corresponding sections
// subspaceCoord - coordinates of section (in)
// strides1 - strides of 1st tensor (in)
// strides2 - strides of 2nd tensor (in)
// strides3 - strides of 3rd tensor (in)
// nDims - dimensionality (in)
// broadcast1 - broadcast flags for 1st tensor, by dimensions (0=normal, 1=broadcasted)
// broadcast2 - broadcast flags for 2nd tensor, by dimensions (0=normal, 1=broadcasted)
// broadcast3 - broadcast flags for 3rd tensor, by dimensions (0=normal, 1=broadcasted)
// offset1 offset in 1st tensor (out)
// offset2 offset in 2nd tensor (out)
// offset3 offset in 3rd tensor (out)
INLINE_ATTRIBUTE void getOffsetsU8(const int32_t subspaceCoord[], const int32_t strides1[], const int32_t strides2[],
        const int32_t strides3[], int nDims, unsigned& offset1, unsigned& offset2, unsigned& offset3,
        const int8_t broadcast1[] = nullptr, const int8_t broadcast2[] = nullptr, const int8_t broadcast3[] = nullptr);

// increment1Coord increments current subspaceCoord by 1 element
// subspaceCoord - coordinates of section (in/out)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
INLINE_ATTRIBUTE void increment1Coord(int32_t subspaceCoord[], const int32_t dims[], int nDims);

// incrementNCoord increments current subspaceCoord by N elements
// subspaceCoord - coordinates of section (in/out)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// inc - value of the increment in elements (in)
INLINE_ATTRIBUTE void incrementNCoord(int32_t subspaceCoord[], const int32_t dims[], int nDims, int inc);

// incrementLine increments current coordinates of 1D section (line along axis coordinate) by 1
// lineCoord - full coordinate vector with line's coordinates (lineCoord[axis] is ignored) (in/out)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// axis number of coordinate along which the line goes (in)
INLINE_ATTRIBUTE void incrementLine(int32_t lineCoord[], const int32_t dims[], int nDims, int axis);

// incrementPlane increments current coordinates of 2D section (plane on axis0, axis1 coordinates) by 1
// planeCoord - full coordinate vector with plane's coordinates (, planeCoord[axis1] are ignored) (in/out)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// axis0, axis1 numbers of coordinates on which the plane is built (in)
INLINE_ATTRIBUTE void incrementPlane(int32_t planeCoord[], const int32_t dims[], int nDims, int axis0, int axis1);

// getTotalLines calculates amount of different 1D sections in tensor
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// axis number of coordinate along which the lines go (in)
// returns common amount of different 1D sections in tensor
INLINE_ATTRIBUTE int getTotalLines(const int32_t dims[], int nDims, int axis);

// getTotalPlanes calculates amount of different 2D sections in tensor
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// axis0, axis1 numbers of coordinates on which the plane is built (in)
// returns common amount of different 2D sections in tensor
INLINE_ATTRIBUTE int getTotalPlanes(const int32_t dims[], int nDims, int axis0, int axis1);

// arrayElementExclude Excludes 1 element from array
// arraysElementExclude Excludes 1 element from 2 or 3 parallel arrays
// a,b,c - target arrays (in/out)
// el - number of element to be excluded
// nEls - size of original array
// returns size of the array after excluding
INLINE_ATTRIBUTE int arrayElementExclude(int32_t a[], int el, int nEls);
INLINE_ATTRIBUTE int arraysElementExclude(int32_t a[], int32_t b[], int el, int nEls);
INLINE_ATTRIBUTE int arraysElementExclude(int32_t a[], int32_t b[], int el, int nEls);

template <typename TA0, typename TA1, typename TA2>
INLINE_ATTRIBUTE int arraysElementExclude(TA0 a[], TA1 b[], TA2 c[], int el, int nEls)
{
    for(int i = el; i < nEls - 1; ++i)
    {
        a[i] = a[i + 1];
        b[i] = b[i + 1];
        c[i] = c[i + 1];
    }
    return nEls - 1;
}

// arrayElementInclude Includes 1 element to array
// arraysElementInclude Includes 1 element to 2 parallel arrays
// a,b - target arrays (in/out)
// elementPos - number of element to be included
// value - element value to be included
// elementsCount - size of original array
// returns size of the array after including
INLINE_ATTRIBUTE int arrayElementInclude(int32_t a[], int elementPos, int32_t value, int elementsCount, int maxDims = MAX_ND_DIMS);
INLINE_ATTRIBUTE int arraysElementInclude(int32_t a[], int32_t b[], int elementPos, int32_t value, int elementsCount, int maxDims = MAX_ND_DIMS);

// getSizes calculates sizes (in elements) of included subtensors of smaller dimensionality,
// subspaceDims - sizes of dimensions (in)
// nDims - dimensionality (in)
// subspaceSizes - sizes of included subtensors (out)
// returns common number of elements
INLINE_ATTRIBUTE int getSizes(const int32_t subspaceDims[], int nDims, int32_t subspaceSizes[]);

// permutation array (perm):
//      perm[i] contains:
//          - index of logical dimension (0..MAX_ND_DIMS-1) on the i-th memory ordered place (from inner to outer)
//            (which logical dimension are on the i-th memory ordered place)
//      Examples: NCHW order corresponds to perm = {3, 2, 1, 0}
//                HWC order corresponds to  perm = {0, 2, 1}
//
// Indices array (mapping)
//      indices[i] contains:
//          - index of memory ordered dimension (0..MAX_ND_DIMS-1) on the i-th logical ordered place
//            (where the i-th logical dimension 'go' in memory ordered dimensions)
//      Examples: NCHW order corresponds to indices = {3, 2, 1, 0}
//                HWC order corresponds to  indices = {0, 2, 1}
//
// Order value
//      number, i-th hexadecimal digit of which, equals (perm[i] + 1)
//      0 - there is no dimension
//      1 - 15 (1 - F) ( 1 - 4 for NCHW) ((hexadecimal number of dimension) + 1)

// Validation conditions:
//      order should not contain one digit<>0 more than once
//      order length - amount of nonzero hexadecimal digit in order value
//      all digits on positions up to the order length should be defined
//      all digits from 1 up to order length should be presented in the order value
//      all digits on positions upper or equal to the order length should be 0

template<class T>
INLINE_ATTRIBUTE static void permuteArray(const T src_set[], const int32_t permutation[], T dst_set[], int set_lng) {
    for (int i = 0; i < set_lng; i ++) {
        dst_set[i] = src_set[permutation[i]];
    }
}

INLINE_ATTRIBUTE int orderNDToNumDims(NDOrder ndOrder);

INLINE_ATTRIBUTE NDDims orderNDToPermutation(NDOrder ndOrder, bool& success);
INLINE_ATTRIBUTE NDDims orderNDToIndices(NDOrder ndOrder, bool& success);
INLINE_ATTRIBUTE NDOrder permutationToOrderND(const NDDims perm);

// alignPermutationSize makes the length of permutation vector equal to dimensionality
// by removing or adding "elder" dimensions (with minimal contents)
// depending on length > dimensionality or length < dimensionality correspondingly
INLINE_ATTRIBUTE bool alignPermutationSize(NDDims& baseLinePerm, int dimensionality);

// extractLayoutFromShape calculates layout value on the base of dimensions and strides arrays
// baseLineOrder is used as template in corresponding of which the dimension index is selected
// in the cases of ambiguity
// i.e. if the dimension == 1 then subsequent stride will be equal to current and it is impossible)
INLINE_ATTRIBUTE NDOrder extractLayoutFromShape(const long unsigned int newDims[],
    const long unsigned int newStrides[], int dimensionality, NDOrder baseLineNDOrder, bool& success);

INLINE_ATTRIBUTE bool isLayoutFit(NDOrder ndOrder, const long unsigned int lDims[],
    const long unsigned int lStrides[], int dimensionality);

INLINE_ATTRIBUTE bool isPermutationValid(const NDDims& perm);
INLINE_ATTRIBUTE bool isOrderNDValid(NDOrder ndOrder);

}  // namespace subspace
#ifdef CONFIG_ALWAYS_INLINE
#include "../src/mvSubspaces.cpp"
#endif

#endif  // MV_SUBSPACES_H_
