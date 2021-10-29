// {% copyright %}

#pragma once
#include <mv_types.h>
#include <array>
#include <algorithm>
#include "common_types.h"

#define ALWAYS_INLINE

using namespace sw_params;

namespace subspace
{
enum {
    UNDEF = -1,
    HEX_DIGIT_BITS = 4,
};

struct LogDimIndex {
    inline __attribute((always_inline)) LogDimIndex(int index) {
        _ind = std::min<int>(std::max(index, 0), MAX_ND_DIMS);
    }
    inline __attribute((always_inline)) int val() const {return _ind;};
    inline __attribute((always_inline)) LogDimIndex &operator =(const int index) {
        _ind = std::min<int>(std::max(index, 0), MAX_ND_DIMS);
        return *this;
    };
    inline __attribute((always_inline)) bool operator<(const LogDimIndex b) const {return val() < b.val();};
    inline __attribute((always_inline)) bool operator==(const LogDimIndex b) const {return val() == b.val();};
private:
    int32_t _ind = 0;
};

// arrayElementExclude Excludes 1 element from array
// arraysElementExclude Excludes 1 element from 2 or 3 parallel arrays
// a,b,c - target arrays (in/out)
// el - number of element to be excluded
// nEls - size of original array
// returns size of the array after excluding
#ifndef ALWAYS_INLINE
int arrayElementExclude(int32_t a[], int el, int nEls);
int arraysElementExclude(int32_t a[], int32_t b[], int el, int nEls);
#else
inline int __attribute((always_inline)) arrayElementExclude(int32_t a[], int el, int nEls)
{
    for(int i = el; i < nEls - 1; ++i)
    {
        a[i] = a[i + 1];
    }
    return nEls - 1;
}

inline int __attribute((always_inline)) arraysElementExclude(int32_t a[], int32_t b[], int el, int nEls)
{
    for(int i = el; i < nEls - 1; ++i)
    {
        a[i] = a[i + 1];
        b[i] = b[i + 1];
    }
    return nEls - 1;
}
#endif

//template <typename TA0, typename TA1, typename TA2>
//int arraysElementExclude(TA0 a[], TA1 b[], TA2 c[], int el, int nEls)
//{
//    for(int i = el; i < nEls - 1; ++i)
//    {
//        a[i] = a[i + 1];
//        b[i] = b[i + 1];
//        c[i] = c[i + 1];
//    }
//    return nEls - 1;
//}

struct NDDims {
    inline __attribute((always_inline)) int ndims() const {return _ndims;};
    inline __attribute((always_inline)) int32_t *data(){return _dims.data();};
    inline __attribute((always_inline)) bool resize(int newNDims) {
        if (newNDims >= 0 && newNDims <= MAX_ND_DIMS) {  // Non-negative up to 15 dimensionality is only supported
            _ndims = newNDims;
            return true;
        } else {
            return false;
        }
    }
    inline __attribute((always_inline)) bool push_back(int32_t value) {
        if (_ndims >= MAX_ND_DIMS) return false;  // Impossible to add more than MAX_ND_DIMS == 15 elements
        _dims[_ndims++] = value;
        return true;
    }
    inline __attribute((always_inline)) bool erase(int i) {
        if (_ndims <= 0) return false;  // Impossible to erase. No elements;
        _ndims = arrayElementExclude(_dims.data(), i, _ndims);
        return true;
    }
    inline __attribute((always_inline)) int32_t& operator[] (LogDimIndex l) {
        int i = l.val();
        return _dims[i];
    }
    inline __attribute((always_inline)) int32_t operator[] (LogDimIndex l) const {
        int i = l.val();
        return _dims[i];
    }
    inline __attribute((always_inline)) int32_t getElement(int i, int32_t defVal) const {
        if (i < 0 || i >= _ndims) {
            return defVal;
        } else return _dims[i];
    }
    inline __attribute((always_inline)) int32_t getElement(LogDimIndex l, int32_t defVal) const {
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
#ifndef ALWAYS_INLINE
int getTotal(const int32_t subspaceDims[], int nDims);
#else
inline int __attribute((always_inline)) getTotal(const int32_t subspaceDims[], int nDims)
{
    int totalSubspaces = 1;
    for(int i = 0; i < nDims; i++)
    {
        totalSubspaces *= subspaceDims[i];
    }
    return totalSubspaces;
}
#endif
// getCoord uses number of section to calculate coordinates of section
// nSubspace - number of section (in)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// subspaceCoord - coordinates of section (out)
#ifndef ALWAYS_INLINE
void getCoord(int nSubspace, const int32_t dims[], int nDims, int32_t subspaceCoord[]);
#else
inline void __attribute((always_inline)) getCoord(int nSubspace, const int32_t dims[], int nDims, int32_t subspaceCoord[])
{
    for(int i = 0; i < nDims; ++i)
    {
        int nUpSubspace = nSubspace / dims[i];
        subspaceCoord[i] = nSubspace - nUpSubspace * dims[i];
        nSubspace = nUpSubspace;
    }
}
#endif

// getOffsetU8 uses coordinates of the section and strides to calculate offset (in bytes)
// from beginning of original tensor to beginning of section
// subspaceCoord - coordinates of section (in)
// strides - strides (in)
// nDims - dimensionality (in)
// broadcast - broadcast flags, by dimensions (0=normal, 1=broadcasted)
// returns offset
#ifndef ALWAYS_INLINE
int getOffsetU8(const int32_t subspaceCoord[], const int32_t strides[], int nDims, const int8_t broadcast[] = nullptr);
#else
inline __attribute((always_inline)) int getOffsetU8(const int32_t subspaceCoord[], const int32_t strides[], int nDims, const int8_t broadcast[] = nullptr)
{
    int offset = 0;
    for(int d = 0; d < nDims; ++d)
    {
        const int coord = (broadcast && broadcast[d]) ? 0 : subspaceCoord[d];
        offset += coord * strides[d];
    }
    return offset;
}

#endif

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
#ifndef ALWAYS_INLINE
void getOffsetsU8(const int32_t subspaceCoord[], const int32_t strides1[], const int32_t strides2[],
        int nDims, unsigned& offset1, unsigned& offset2,
        const int8_t broadcast1[] = nullptr, const int8_t broadcast2[] = nullptr);
#else
inline void __attribute((always_inline)) getOffsetsU8(const int32_t subspaceCoord[], const int32_t strides1[], const int32_t strides2[],
        int nDims, unsigned& offset1, unsigned& offset2,
        const int8_t broadcast1[] = nullptr, const int8_t broadcast2[] = nullptr)
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
#endif

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
#ifndef ALWAYS_INLINE
void getOffsetsU8(const int32_t subspaceCoord[], const int32_t strides1[], const int32_t strides2[],
        const int32_t strides3[], int nDims, unsigned& offset1, unsigned& offset2, unsigned& offset3,
        const int8_t broadcast1[] = nullptr, const int8_t broadcast2[] = nullptr, const int8_t broadcast3[] = nullptr);
#else
inline void __attribute((always_inline)) getOffsetsU8(const int32_t subspaceCoord[], const int32_t strides1[], const int32_t strides2[],
        const int32_t strides3[], int nDims, unsigned& offset1, unsigned& offset2, unsigned& offset3,
        const int8_t broadcast1[] = nullptr, const int8_t broadcast2[] = nullptr, const int8_t broadcast3[] = nullptr)
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
#endif

// increment1Coord increments current subspaceCoord by 1 element
// subspaceCoord - coordinates of section (in/out)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
#ifndef ALWAYS_INLINE
void increment1Coord(int32_t subspaceCoord[], const int32_t dims[], int nDims);
#else
inline void __attribute((always_inline)) increment1Coord(int32_t subspaceCoord[], const int32_t dims[], int nDims)
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
#endif

// incrementNCoord increments current subspaceCoord by N elements
// subspaceCoord - coordinates of section (in/out)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// inc - value of the increment in elements (in)
#ifndef ALWAYS_INLINE
void incrementNCoord(int32_t subspaceCoord[], const int32_t dims[], int nDims, int inc);
#else
inline void __attribute((always_inline)) incrementNCoord(int32_t subspaceCoord[], const int32_t dims[], int nDims, int inc)
{
    for(int d = 0; d < nDims; ++d)
    {
        inc += subspaceCoord[d];
        subspaceCoord[d] = inc % dims[d];
        inc -= subspaceCoord[d];
        inc /= dims[d];
    }
}
#endif

// incrementLine increments current coordinates of 1D section (line along axis coordinate) by 1
// lineCoord - full coordinate vector with line's coordinates (lineCoord[axis] is ignored) (in/out)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// axis number of coordinate along which the line goes (in)
#ifndef ALWAYS_INLINE
void incrementLine(int32_t lineCoord[], const int32_t dims[], int nDims, int axis);
#else
inline __attribute((always_inline)) void incrementLine(int32_t lineCoord[], const int32_t dims[], int nDims, int axis)
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
#endif

// incrementPlane increments current coordinates of 2D section (plane on axis0, axis1 coordinates) by 1
// planeCoord - full coordinate vector with plane's coordinates (, planeCoord[axis1] are ignored) (in/out)
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// axis0, axis1 numbers of coordinates on which the plane is built (in)
#ifndef ALWAYS_INLINE
void incrementPlane(int32_t planeCoord[], const int32_t dims[], int nDims, int axis0, int axis1);
#else
inline __attribute((always_inline)) void incrementPlane(int32_t planeCoord[], const int32_t dims[], int nDims, int axis0, int axis1)
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
#endif

// getTotalLines calculates amount of different 1D sections in tensor
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// axis number of coordinate along which the lines go (in)
// returns common amount of different 1D sections in tensor
#ifndef ALWAYS_INLINE
int getTotalLines(const int32_t dims[], int nDims, int axis);
#else
inline __attribute((always_inline)) int getTotalLines(const int32_t dims[], int nDims, int axis)
{
    return (dims[axis]) ? getTotal(dims, nDims) / dims[axis] : 0;
}
#endif

// getTotalPlanes calculates amount of different 2D sections in tensor
// dims - sizes of dimensions (in)
// nDims - dimensionality (in)
// axis0, axis1 numbers of coordinates on which the plane is built (in)
// returns common amount of different 2D sections in tensor
#ifndef ALWAYS_INLINE
int getTotalPlanes(const int32_t dims[], int nDims, int axis0, int axis1);
#else
inline __attribute((always_inline)) int getTotalPlanes(const int32_t dims[], int nDims, int axis0, int axis1)
{
    return (dims[axis0] * dims[axis1]) ? getTotal(dims, nDims) / (dims[axis0] * dims[axis1]) : 0;
}
#endif

// arrayElementInclude Includes 1 element to array
// arraysElementInclude Includes 1 element to 2 parallel arrays
// a,b - target arrays (in/out)
// elementPos - number of element to be included
// value - element value to be included
// elementsCount - size of original array
// returns size of the array after including
#ifndef ALWAYS_INLINE
int arrayElementInclude(int32_t a[], int elementPos, int32_t value, int elementsCount, int maxDims = MAX_ND_DIMS);
int arraysElementInclude(int32_t a[], int32_t b[], int elementPos, int32_t value, int elementsCount, int maxDims = MAX_ND_DIMS);
#else
inline int __attribute((always_inline)) arrayElementInclude(int32_t a[], int elementPos, int32_t value, int elementsCount, int maxDims = MAX_ND_DIMS);
inline int __attribute((always_inline)) arrayElementInclude(int32_t a[], int elementPos, int32_t value, int elementsCount, int maxDims)
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

inline __attribute((always_inline)) int arraysElementInclude(int32_t a[], int32_t b[], int elementPos, int32_t value, int elementsCount, int maxDims = MAX_ND_DIMS)
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
#endif

// getSizes calculates sizes (in elements) of included subtensors of smaller dimensionality,
// subspaceDims - sizes of dimensions (in)
// nDims - dimensionality (in)
// subspaceSizes - sizes of included subtensors (out)
// returns common number of elements
#ifndef ALWAYS_INLINE
int getSizes(const int32_t subspaceDims[], int nDims, int32_t subspaceSizes[]);
#else
inline __attribute((always_inline)) int getSizes(const int32_t subspaceDims[], int nDims, int32_t subspaceSizes[])
{
    int totalSubspaces = 1;
    for(int i = 0; i < nDims; i++)
    {
        subspaceSizes[i] = totalSubspaces;
        totalSubspaces *= subspaceDims[i];
    }
    return totalSubspaces;
}
#endif

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

#ifndef ALWAYS_INLINE
int orderNDToNumDims(NDOrder ndOrder);
NDDims orderNDToPermutation(NDOrder ndOrder, bool& success);
NDDims orderNDToIndices(NDOrder ndOrder, bool& success);
NDOrder permutationToOrderND(const NDDims perm);
int orderNDToNumDims(NDOrder ndOrder);
NDDims orderNDToPermutation(NDOrder ndOrder, bool& success);
NDDims orderNDToIndices(NDOrder ndOrder, bool& success);
NDOrder permutationToOrderND(const NDDims perm);
bool isLayoutFit(NDOrder ndOrder, const long unsigned int lDims[],
                 const long unsigned int lStrides[], int dimensionality);

bool isPermutationValid(const NDDims& perm);
bool isOrderNDValid(NDOrder ndOrder);
#else
inline __attribute((always_inline)) int orderNDToNumDims(NDOrder ndOrder);
inline __attribute((always_inline)) NDDims orderNDToPermutation(NDOrder ndOrder, bool& success);
inline __attribute((always_inline)) NDDims orderNDToIndices(NDOrder ndOrder, bool& success);
inline __attribute((always_inline)) NDOrder permutationToOrderND(const NDDims perm);
inline __attribute((always_inline)) bool isLayoutFit(NDOrder ndOrder, const long unsigned int lDims[],
                 const long unsigned int lStrides[], int dimensionality);

inline __attribute((always_inline)) bool isPermutationValid(const NDDims& perm);
inline __attribute((always_inline)) bool isOrderNDValid(NDOrder ndOrder);
inline __attribute((always_inline)) int orderNDToNumDims(NDOrder ndOrder)
{
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
template<class T>
static inline __attribute((always_inline)) void permuteArray(const T src_set[], const int32_t permutation[], T dst_set[], int set_lng) {
    for (int i = 0; i < set_lng; i ++) {
        dst_set[i] = src_set[permutation[i]];
    }
}

inline __attribute((always_inline)) NDDims orderNDToPermutation(NDOrder ndOrder, bool& success)
{
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
inline __attribute((always_inline)) NDDims orderNDToIndices(NDOrder ndOrder, bool& success)
{
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
inline __attribute((always_inline)) NDOrder permutationToOrderND(const NDDims perm)
{
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
#endif

// alignPermutationSize makes the length of permutation vector equal to dimensionality
// by removing or adding "elder" dimensions (with minimal contents)
// depending on length > dimensionality or length < dimensionality correspondingly
#ifndef ALWAYS_INLINE
bool alignPermutationSize(NDDims& baseLinePerm, int dimensionality);
#else
inline __attribute((always_inline)) bool alignPermutationSize(NDDims& baseLinePerm, int dimensionality)
{
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
#endif

// extractLayoutFromShape calculates layout value on the base of dimensions and strides arrays
// baseLineOrder is used as template in corresponding of which the dimension index is selected
// in the cases of ambiguity
// i.e. if the dimension == 1 then subsequent stride will be equal to current and it is impossible)
#ifndef ALWAYS_INLINE
NDOrder extractLayoutFromShape(const long unsigned int newDims[],
                               const long unsigned int newStrides[], int dimensionality, NDOrder baseLineNDOrder, bool& success);
#else
inline __attribute((always_inline)) NDOrder extractLayoutFromShape(const long unsigned int newDims[],
    const long unsigned int newStrides[], int dimensionality, NDOrder baseLineNDOrder, bool& success)
{
    if (baseLineNDOrder <= 0) {
        baseLineNDOrder = static_cast<NDOrder>(static_cast<uint64_t>(
                                                     FULL_ND_ORDER) >> (HEX_DIGIT_BITS * (MAX_ND_DIMS - dimensionality)));
    }
    auto baseLinePerm = subspace::orderNDToPermutation(baseLineNDOrder, success);
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
#endif

#ifndef ALWAYS_INLINE
bool isLayoutFit(NDOrder ndOrder, const long unsigned int lDims[],
                 const long unsigned int lStrides[], int dimensionality);

bool isPermutationValid(const NDDims& perm);
bool isOrderNDValid(NDOrder ndOrder);
#else
inline __attribute((always_inline)) bool isLayoutFit(NDOrder ndOrder, const long unsigned int lDims[],
    const long unsigned int lStrides[], int dimensionality)
{
    bool success = false;
    auto extracted = extractLayoutFromShape(lDims, lStrides, dimensionality, ndOrder, success);
    return (success && (extracted == ndOrder));
}

inline __attribute((always_inline)) bool isPermutationValid(const NDDims& perm)
{
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

inline __attribute((always_inline)) bool isOrderNDValid(NDOrder ndOrder)
{
    bool ret = false;
    orderNDToPermutation(ndOrder, ret);
    return ret;
}
#endif
}  // namespace subspace
