/*
 * {% copyright %}
 */

namespace vpux {

struct __attribute__((packed)) MemRefData {
    uint32_t dataAddr;  // Can't use pointers, since they have platform-dependent size.
                        // Will be located in WIN_F.

    uint32_t isStatic;  // Boolean flag to indicate static shape vs dynamic shape.

    uint32_t numDims;
    uint32_t dimsAddr;      // Pointer to the buffer with dimensions (int32_t[]).
    uint32_t stridesAddr;   // Pointer to the buffer with strides in bits (int64_t[]).
                           // Will be located in WIN_E (static case) or in WIN_F (dynamic case).
                           // The kernel should infer output dims/strides and write them only in dynamic case.

    uint32_t dataType;      // An enum, which should be aligned between kernels and the compiler.
    uint64_t dimsOrder;     // Packed permutation array.
};


struct __attribute__((packed)) BaseParams {
    MemRefData input;
    MemRefData output;
};

}  // namespace vpux