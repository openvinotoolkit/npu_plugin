/*
 * {% copyright %}
 */
#pragma once

#include "nn_hw_resources.h"
#include <nn_memory.h>
#include <array>
#include <limits>
#include <algorithm>

namespace nn {
namespace common_runtime {
struct ConstBuffer {
    // 64 bit address capable buffer representation
    ConstBuffer(uint64_t addr = 0, unsigned int size = 0)
        : addr_(addr)
        , size_(size) {}

    ConstBuffer(const void *ptr, unsigned int size = 0)
        : addr_(reinterpret_cast<uint32_t>(ptr))
        , size_(size) {}

    inline void assign(const void *ptr) { addr_ = static_cast<uint64_t>(reinterpret_cast<uint32_t>(ptr)); }

    inline void assign(uint64_t addr) { addr_ = addr; }

    inline uint64_t addr() const { return addr_; }

    uint32_t addr32() const;

    inline unsigned int size() const { return size_; }

    inline operator uint64_t() const { return addr(); }

    ConstBuffer add(unsigned int offset, bool *overflow) const;

    uint64_t operator+(unsigned int offset) const = delete;

private:
    uint64_t addr_;
    unsigned int size_;
};

// This is a copy to make code readability easier, if the buffer should be modifiable or not
// Since we are using integer addresses now instead of pointers the compiler can't help us
typedef struct ConstBuffer Buffer;

struct ClusterMapper {
    typedef std::pair<unsigned char, unsigned char> Range;

    struct Config {
        Config(unsigned int mask)
            : mask_(mask) {}

        bool valid() const;
        unsigned int index() const;
        Range range() const;
        operator unsigned int() const { return mask_; }

    private:
        unsigned int mask_;
    };

    ClusterMapper() {}

    unsigned char config_count(unsigned int clusters) const;
    Config config(unsigned int clusters, unsigned int i) const;
};

struct BarrierMapper {
    explicit BarrierMapper(unsigned char config);

    unsigned char map_by_id(unsigned char b) const;
    unsigned long long map_by_mask(unsigned long long mask) const;

private:
    std::array<unsigned char, TOTAL_PHYSICAL_BARRIERS> mappings_;
};

struct NNCmxAddressSolver {
    NNCmxAddressSolver(const std::array<Buffer, MAX_SLICES> &workspace_buffers, uint32_t activeSliceMask);
    Buffer solve(unsigned char relativeMask) const;

private:
    std::array<Buffer, MAX_SLICES> workspace_buffers_;
    uint32_t activeSliceMask_;

    unsigned int toAbsoluteMask(unsigned char relativeMask) const;
    unsigned int toBroadcastMask(unsigned int absoluteMask) const;
};

struct NN_CACHE_ALIGNED CommonRelocationData {
    nn::memory::cache_aligned_vector<ConstBuffer> blob_buffers_;
    nn::memory::cache_aligned_vector<ConstBuffer> kernel_buffers_;
    nn::memory::cache_aligned_vector<ConstBuffer> inputs_;
    nn::memory::cache_aligned_vector<Buffer> outputs_;
    nn::memory::cache_aligned_vector<Buffer> profiling_outputs_;
    Buffer ddr_heap_; // garbage-initialized
    Buffer ddr_bss_;  // zero-initialized

    CommonRelocationData(const nn::memory::cache_aligned_vector<ConstBuffer> &blob_buffers,
                         const nn::memory::cache_aligned_vector<ConstBuffer> &kernel_buffers, unsigned int input_count,
                         unsigned int output_count, unsigned int profiling_output_count)
        : blob_buffers_(blob_buffers)
        , kernel_buffers_(kernel_buffers)
        , inputs_(input_count)
        , outputs_(output_count)
        , profiling_outputs_(profiling_output_count)
        , ddr_heap_()
        , ddr_bss_() {}

    void print() const;
};

struct NNRelocationData {
    const CommonRelocationData &crd_;
    NNCmxAddressSolver nn_solver_;

    NNRelocationData(const CommonRelocationData &crd, const std::array<Buffer, MAX_SLICES> &workspace_buffers,
                     uint32_t activeSliceMask)
        : crd_(crd)
        , nn_solver_(workspace_buffers, activeSliceMask) {}

    // For JIT case where we don't need NNCMX data
    NNRelocationData(const CommonRelocationData &crd)
        : crd_(crd)
        , nn_solver_({Buffer(0xDEADBEEF, 0x0), Buffer(0xDEADBEEF, 0x0)}, 0) {}

    unsigned int getFistSlice();
};

struct SimplifiedTensorLayout {
    enum {
        STRIDING_LEVELS = 2,
    };

    SimplifiedTensorLayout();
    bool load(unsigned int dims, const unsigned char *order, const float *strides, const unsigned int *sizes);
    void print() const;

    inline unsigned int line_stride() const { return line_stride_; }
    inline unsigned int line_length() const { return line_length_; }
    inline unsigned int plane_stride() const { return plane_stride_; }
    inline unsigned int plane_length() const { return plane_length_; }
    inline unsigned int plane_count() const {
        return plane_length_ ? (total_length_ / plane_length_ / (line_length_ ? line_length_ : 1)) : 1;
    }
    inline unsigned int total_length() const { return total_length_; }

private:
    unsigned int line_stride_;
    unsigned int line_length_;
    unsigned int plane_stride_;
    unsigned int plane_length_;
    unsigned int total_length_;
};

struct RelativeAddress {
    enum class Location : unsigned char {
        None,
        Blob,
        BlobKernels,
        Input,
        Output,
        Heap,
        Bss,
        NnCmx,
        Absolute, // Used when data_offset is an absolute address
        KernelsBuffer,
        MAC_Accumulators,
        ProfilingOutput
    };

    enum class Class : unsigned char {
        Base,
        Data,
        SparsityMap,
        SparsityTable,
    };

    RelativeAddress()
        : location_(Location::None)
        , index_(0)
        , data_offset_(0)
        , sparsity_map_offset_(0)
        , sparsity_table_offset_(0)
        , referenced_data_size_(0) {}

    explicit RelativeAddress(Location location, unsigned int index = 0, unsigned int data_offset = 0,
                             unsigned int sparsity_map_offset = 0, unsigned int sparsity_table_offset = 0,
                             unsigned int referenced_data_size = 0)
        : location_(location)
        , index_(index)
        , data_offset_(data_offset)
        , sparsity_map_offset_(sparsity_map_offset)
        , sparsity_table_offset_(sparsity_table_offset)
        , referenced_data_size_(referenced_data_size) {}

    Buffer resolve(const NNRelocationData &rd, Class c = Class::Data, bool *overflow = nullptr) const;
    Buffer resolve32(const NNRelocationData &rd, Class c = Class::Data, bool *overflow = nullptr) const;

    unsigned int offset(Class c) const;
    Location location() const { return location_; }
    unsigned int index() const { return index_; }

    bool isValid() const { return location_ != Location::None; }
    void print(const char *name = "") const;
    void set_index(unsigned int index) { index_ = index; }

    static uint32_t to_dpu_multicast(uint32_t addr, unsigned int &offset1, unsigned int &offset2,
                                     unsigned int &offset3);
    static uint32_t to_dpu_multicast_base(uint32_t addr);

private:
    Location location_;
    unsigned int index_;
    unsigned int data_offset_;
    unsigned int sparsity_map_offset_;
    unsigned int sparsity_table_offset_;
    unsigned int referenced_data_size_;
};
} // namespace common_runtime
} // namespace nn
