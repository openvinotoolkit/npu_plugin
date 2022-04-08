// {% copyright %}

#include "mvSubspaces8d.h"
#include <limits.h>
#include <cstring>
#include <stdio.h>

namespace subspace {

bool isOrderValid(t_D8StorageOrder order) {
    uint32_t ord = order;
    if (!ord) return false;  // zero order
    bool ind[subspace::MAX_DIMS];

    for (int i = 0; i < subspace::MAX_DIMS; i++) {
        ind[i] = true;
    }

    int lng = 0;
    for (int i = 0; i < subspace::MAX_DIMS; i++) {
        int digit = static_cast<int>((ord & 0xF) - 1);
        if (digit != UNDEF) {
            if (digit >= subspace::MAX_DIMS) return false;
            if (!ind[digit]) return false;  // dimension is used more than once
            ind[digit] = false;  // dimension have been used
            lng = i + 1;
        } else {
            break;
        }

        ord >>= HEX_DIGIT_BITS;
    }

    ord = order >> (HEX_DIGIT_BITS * lng);
    for (int i = lng; i < subspace::MAX_DIMS; i++) {
        int digit = static_cast<int>((ord & 0xF) - 1);
        if (digit != UNDEF) return false;  // all digits on positions upper or equal to the order length should be UNDEF
        ord >>= HEX_DIGIT_BITS;
    }

    return true;
}

int orderToNumDims(t_D8StorageOrder order) {
    int i = 0;

    for (i = 0; i < subspace::MAX_DIMS; i++) {
        int digit = static_cast<int>((order & 0xF) - 1);
        if ((unsigned)digit >= (unsigned)subspace::MAX_DIMS) {
            break;
        }
        order >>= HEX_DIGIT_BITS;
    }
    return i;
}

int orderToPermutation(t_D8StorageOrder order, int32_t perm[]) {
    int lng = 0;

    for (int i = 0; i < subspace::MAX_DIMS; i++) {
        perm[i] = UNDEF;
    }
    for (int i = 0; i < subspace::MAX_DIMS; i++) {
        int digit = static_cast<int>((order & 0xF) - 1);
        if ((unsigned)digit >= (unsigned)subspace::MAX_DIMS) {
            break;
        }
        perm[i] = digit;
        lng = i + 1;
        order >>= HEX_DIGIT_BITS;
    }
    return lng;
}

int orderToIndices(t_D8StorageOrder order, int32_t indices[]) {
    int lng = 0;

    for (int i = 0; i < subspace::MAX_DIMS; i++) {
        indices[i] = UNDEF;
    }
    for (int i = 0; i < subspace::MAX_DIMS; i++) {
        int ind = static_cast<int>((order & 0xF) - 1);
        if ((unsigned)ind >= (unsigned)subspace::MAX_DIMS)
            break;
        indices[ind] = i;
        lng = i + 1;
        order >>= HEX_DIGIT_BITS;
    }
    return lng;
}

t_D8StorageOrder permutationToOrder(const int32_t perm[], int length) {
    uint32_t order = 0;
    length = (length < subspace::MAX_DIMS) ? length : subspace::MAX_DIMS;
    for (int sh = 0, i = 0; i < length; i++, sh += HEX_DIGIT_BITS) {
        order += (((static_cast<unsigned int>(perm[i]) + 1) & 0xF) << sh);
    }

    return order;
}

NDOrder orderToNDOrder(uint32_t order) {
    int32_t perm[subspace::MAX_DIMS];
    std::memset(perm, 0x0, subspace::MAX_DIMS * sizeof(perm[0]));
    int ndims = subspace::orderToPermutation(order, perm);
    // Enumerate elements of permutation ascending
    for (int i = 0; i < ndims; i++) {
        auto minPtr = std::min_element(perm, perm + ndims);
        // add a big constant (MAX_ND_DIMS) to work with each element once
        *minPtr = MAX_ND_DIMS + i;
    }
    auto maxPtr = std::max_element(perm, perm + ndims);
    int32_t maxDimNum = *maxPtr;
    // Inverse enumeration. The 'big constant' is eliminated automatically
    for (int i = 0; i < ndims; i++) {
        perm[i] = maxDimNum - perm[i];
    }

    uint32_t newOrder = permutationToOrder(perm, ndims);
    return static_cast<NDOrder>(newOrder);
}

t_D8StorageOrder NDorderToOrder(NDOrder ndOrder, bool& success) {
    NDDims perm = orderNDToPermutation(ndOrder, success);
    if (success && perm.ndims() <= MAX_DIMS) {
        uint32_t oldOrder = 0;
        int oldOrdSize = std::min(static_cast<uint32_t>(perm.ndims()), static_cast<uint32_t>(subspace::MAX_DIMS));
        for (int i = 0; i < oldOrdSize; i++) {
             oldOrder += (oldOrdSize - perm[i]) << (HEX_DIGIT_BITS * i);
        }
        return oldOrder;
    } else {
        success = false;
        return 0;
    }
}

uint8_t getDimName(Dim dimNum) {
    const static char dimName[] = {'W', 'H', 'C', 'N', '5', '6', '7', '8'};
    return dimName[dimNum];
}

uint8_t getDimNameOfOrder(int32_t dimNum, uint32_t order) {
    int32_t perm[subspace::MAX_DIMS];
    int numDims = orderToPermutation(order, perm);
    return (dimNum < 0 || dimNum >= numDims) ? '-' : getDimName(static_cast<Dim>(perm[dimNum]));
}

} //namespace subspace

