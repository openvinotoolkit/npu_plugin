/*
* {% copyright %}
*/
#ifndef NN_RELOCATION_H_
#define NN_RELOCATION_H_

#include "nn_resources.h"
#include <nn_memory.h>
#include <array>
#include <limits>
#include <algorithm>
#include "common_types.h"

namespace nn
{
    namespace inference_runtime
    {
    using namespace sw_params;

        struct ConstBuffer
        {
            // 64 bit address capable buffer representation
            ConstBuffer(uint64_t addr = 0, unsigned int size = 0) :
                addr_(addr),
                size_(size)
            {
            }

            ConstBuffer(const void *ptr, unsigned int size = 0) :
                 addr_(reinterpret_cast<uint32_t>(ptr)),
                 size_(size)
            {
            }

            inline void assign(const void *ptr)
            {
                addr_ = static_cast<uint64_t>(reinterpret_cast<uint32_t>(ptr));
            }

            inline void assign(uint64_t addr)
            {
                addr_ = addr;
            }

            inline uint64_t addr() const
            {
                return addr_;
            }

            uint32_t addr32() const;

            inline unsigned int size() const
            {
                return size_;
            }

            inline operator uint64_t() const
            {
                return addr();
            }

            ConstBuffer add(unsigned int offset, bool *overflow) const;

            uint64_t operator +(unsigned int offset) const = delete;

        private:
            uint64_t addr_;
            unsigned int size_;
        };

        // This is a copy to make code readability easier, if the buffer should be modifiable or not
        // Since we are using integer addresses now instead of pointers the compiler can't help us
        typedef struct ConstBuffer Buffer;

        struct NN_CACHE_ALIGNED UPARelocationData
        {
            nn::memory::cache_aligned_vector<ConstBuffer> blob_buffers_;
            nn::memory::cache_aligned_vector<ConstBuffer> inputs_;
            nn::memory::cache_aligned_vector<Buffer> outputs_;
            Buffer ddr_heap_; // garbage-initialized
            Buffer ddr_bss_; // zero-initialized
            Buffer upa_cmx_;

            UPARelocationData(const nn::memory::cache_aligned_vector<ConstBuffer> &blob_buffers, unsigned int input_count, unsigned int output_count) :
                blob_buffers_(blob_buffers),
                inputs_(input_count),
                outputs_(output_count),
                ddr_heap_(),
                ddr_bss_(),
                upa_cmx_()
            {
            }

            void print() const;
        };

        struct ClusterMapper
        {
            typedef std::pair<unsigned char, unsigned char> Range;

            struct Config
            {
                Config(unsigned int mask) :
                    mask_(mask)
                {
                }

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

        struct BarrierMapper
        {
            explicit BarrierMapper(unsigned char config);

            unsigned char map_by_id(unsigned char b) const;
            unsigned long long map_by_mask(unsigned long long mask) const;

        private:
            std::array<unsigned char, TOTAL_PHYSICAL_BARRIERS> mappings_;
        };

        struct NNCmxTranslationTable
        {
            std::array<char, 16> mappings_;

            static NNCmxTranslationTable fromAbsoluteMask(unsigned int sliceMask);

            NNCmxTranslationTable() :
                mappings_()
            {
                std::fill(mappings_.begin(), mappings_.end(), -1);
            }

            void print() const;
        };

        struct NNCmxAddressSolver
        {
            NNCmxAddressSolver(const Buffer &nn_cmx_base, NNCmxTranslationTable table);
            Buffer solve(unsigned char relativeMask) const;

        private:
            Buffer nn_cmx_base_;
            NNCmxTranslationTable table_;

            unsigned char toAbsoluteMask(unsigned char relativeMask) const;
            unsigned int toBroadcastMask(unsigned int absoluteMask) const;
        };

        struct NNRelocationData
        {
            const UPARelocationData &upa_rd_;
            NNCmxAddressSolver nn_solver_;

            NNRelocationData(const UPARelocationData &upa_rd, const Buffer &nn_cmx, NNCmxTranslationTable table) :
                upa_rd_(upa_rd),
                nn_solver_(nn_cmx, table)
            {
            }
        };

        struct SimplifiedTensorLayout
        {
            enum
            {
                STRIDING_LEVELS = 2,
            };

            SimplifiedTensorLayout();
            bool load(unsigned int dims, const unsigned char *order, const float *strides, const unsigned int *sizes);
            void print() const;

            inline unsigned int line_stride() const { return line_stride_; }
            inline unsigned int line_length() const { return line_length_; }
            inline unsigned int plane_stride() const { return plane_stride_; }
            inline unsigned int plane_length() const { return plane_length_; }
            inline unsigned int plane_count() const { return plane_length_ ? (total_length_ / plane_length_ / (line_length_ ? line_length_ : 1)) : 1; }
            inline unsigned int total_length() const { return total_length_; }

        private:
            unsigned int line_stride_;
            unsigned int line_length_;
            unsigned int plane_stride_;
            unsigned int plane_length_;
            unsigned int total_length_;
        };

        struct RelativeAddress
        {
            enum class Class : unsigned char
            {
                Base,
                Data,
                SparsityMap,
                SparsityTable,
            };

            RelativeAddress() :
                location_(Location::NONE),
                index_(0),
                data_offset_(0),
                sparsity_map_offset_(0),
                sparsity_table_offset_(0)
            {
            }

            explicit RelativeAddress(
                Location location,
                unsigned short index = 0,
                unsigned int data_offset = 0,
                unsigned int sparsity_map_offset = 0,
                unsigned int sparsity_table_offset = 0) :
                location_(location),
                index_(index),
                data_offset_(data_offset),
                sparsity_map_offset_(sparsity_map_offset),
                sparsity_table_offset_(sparsity_table_offset)
            {
            }

            Buffer resolve(const NNRelocationData &rd, Class c = Class::Data, bool *overflow = nullptr) const;
            Buffer resolve32(const NNRelocationData &rd, Class c = Class::Data, bool *overflow = nullptr) const;

            unsigned int offset(Class c) const;
            Location location() const { return location_; }
            unsigned short index() const { return index_; }

            bool isValid() const { return location_ != Location::NONE; }
            void print() const;
            void set_index(unsigned short index) { index_ = index; }

            static uint32_t to_dpu_multicast(uint32_t addr, unsigned int &offset1, unsigned int &offset2, unsigned int &offset3);
            static uint32_t to_dpu_multicast_base(uint32_t addr);

        private:
            Location location_;
            unsigned short index_;
            unsigned int data_offset_;
            unsigned int sparsity_map_offset_;
            unsigned int sparsity_table_offset_;
        };
    }
}

#endif /* NN_RELOCATION_H_ */
