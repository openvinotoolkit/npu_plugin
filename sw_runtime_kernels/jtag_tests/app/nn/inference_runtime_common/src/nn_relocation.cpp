/*
* {% copyright %}
*/
#include "nn_relocation.h"
#include "nn_resources.h"
#include <nn_log.h>
#include <assert.h>
#include <limits.h>

namespace
{
    using namespace nn::inference_runtime;

    #if !defined(CONFIG_TARGET_SOC_3600) && !defined(CONFIG_TARGET_SOC_3710) && !defined(CONFIG_TARGET_SOC_3720)
             const unsigned int ADDRESS_MASK = ~0x00f00000u;
    #else
             const unsigned int ADDRESS_MASK = ~0x00e00000u;
    #endif // CONFIG_SOC_3600

    #if defined(CONFIG_TARGET_SOC_3600) || defined(CONFIG_TARGET_SOC_3710) || defined(CONFIG_TARGET_SOC_3720)
    enum
    {
        CLUSTER_CONFIG_COUNT = 2,
    };

    const unsigned char CLUSTER_SETS[][CLUSTER_CONFIG_COUNT] =
    {
        { 0b0001, 0b0010 },
        { 0b0011,      0 },
    };

    const unsigned char CLUSTER_CONFIGS[MAX_SLICES] =
    {
        2, 1,
    };

    // This table back-references CLUSTER_SETS and stores at what index a particular configuration can be found
    const char CLUSTER_CONFIG_INDEX[] =
    {
        -1,  0,
         1,  2,
    };

    const ClusterMapper::Range WIDEST_RANGE[] =
    {
        { 0, 0 }, // 0b0000
        { 0, 1 }, // 0b0001
        { 1, 1 }, // 0b0010
        { 0, 2 }, // 0b0011
    };
    #else
    enum
    {
        CLUSTER_CONFIG_COUNT = 4,
    };

    const unsigned char CLUSTER_SETS[][CLUSTER_CONFIG_COUNT] =
    {
        { 0b0001, 0b0010, 0b0100, 0b1000 },
        { 0b0011, 0b0110, 0b1100,      0 },
        { 0b0111, 0b1110,      0,      0 },
        { 0b1111,      0,      0,      0 },
    };

    const unsigned char CLUSTER_CONFIGS[MAX_SLICES] =
    {
        4, 3, 2, 1,
    };

    // This table back-references CLUSTER_SETS and stores at what index a particular configuration can be found
    const char CLUSTER_CONFIG_INDEX[] =
    {
        13,  0,  1,  4,
         2, -1,  5,  8,
         3, -1, -1, -1,
         6, -1,  9, 12,
    };

    const ClusterMapper::Range WIDEST_RANGE[] =
    {
        { 0, 0 }, // 0b0000
        { 0, 1 }, // 0b0001
        { 1, 1 }, // 0b0010
        { 0, 2 }, // 0b0011
        { 2, 1 }, // 0b0100
        { 0, 0 }, // 0b0101 // not supported - HW
        { 1, 2 }, // 0b0110
        { 0, 3 }, // 0b0111
        { 3, 1 }, // 0b1000
        { 0, 0 }, // 0b1001 // not supported - SW
        { 0, 0 }, // 0b1010 // not supported - HW
        { 0, 0 }, // 0b1011 // not supported - SW
        { 2, 2 }, // 0b1100
        { 0, 0 }, // 0b1101 // not supported - SW
        { 1, 3 }, // 0b1110
        { 0, 4 }, // 0b1111
    };
    #endif
}

namespace nn
{
    namespace inference_runtime
    {
        ConstBuffer ConstBuffer::add(unsigned int offset, bool *overflow) const
        {
            if (size() <= offset)
            {
                nnLog(MVLOG_ERROR, "Address 0x%llx out of range [0x%llx - 0x%llx)", addr() + offset, addr(), addr() + size());

                if (overflow)
                    *overflow = true;
            }

            return ConstBuffer(addr() + offset, size() - offset);
        }

        uint32_t ConstBuffer::addr32() const
        {
            #ifdef CONFIG_NN_CHECK_ADDRESSES
            if ((addr_ >> 32) != 0)
                nnLog(MVLOG_WARN, "addr32: Truncating address 0x%llx greater than 32 bits\n", addr_);
            #endif

            return static_cast<uint32_t>(addr_);
        }

        void UPARelocationData::print() const
        {
            nnLog(MVLOG_INFO, "%u blob buffer(s)", blob_buffers_.size());
            for (unsigned int b = 0; b < blob_buffers_.size(); ++b)
                nnLog(MVLOG_INFO, "\tBlob %u @ 0x%llx", b, blob_buffers_[b].addr());
            nnLog(MVLOG_INFO, "%u input buffer(s)", inputs_.size());
            for (unsigned int i = 0; i < inputs_.size(); ++i)
                nnLog(MVLOG_INFO, "\tInput %u @ 0x%llx", i, inputs_[i].addr());
            nnLog(MVLOG_INFO, "%u output buffer(s)", outputs_.size());
            for (unsigned int o = 0; o < outputs_.size(); ++o)
                nnLog(MVLOG_INFO, "\tOutput %u @ 0x%llx", o, outputs_[o].addr());
            nnLog(MVLOG_INFO, "DDR Heap @ 0x%llx", ddr_heap_.addr());
            nnLog(MVLOG_INFO, "DDR BSS @ 0x%llx", ddr_bss_.addr());
            nnLog(MVLOG_INFO, "UPA CMX @ 0x%llx", upa_cmx_.addr());
        }

        bool ClusterMapper::Config::valid() const
        {
            return range().second > 0;
        }

        unsigned int ClusterMapper::Config::index() const
        {
            char index = mask_ < sizeof(CLUSTER_CONFIG_INDEX) ? CLUSTER_CONFIG_INDEX[mask_] : 0;
            assert(index >= 0 && "Looking for an invalid configuration");
            return static_cast<unsigned int>(index % CLUSTER_CONFIG_COUNT);
        }

        ClusterMapper::Range ClusterMapper::Config::range() const
        {
            assert(mask_ < sizeof(WIDEST_RANGE) / sizeof(WIDEST_RANGE[0]) && "Requesting range for invalid cluster config");
            return WIDEST_RANGE[mask_];
        }

        unsigned char ClusterMapper::config_count(unsigned int clusters) const
        {
            return clusters ? CLUSTER_CONFIGS[clusters - 1] : 0;
        }

        ClusterMapper::Config ClusterMapper::config(unsigned int clusters, unsigned int i) const
        {
            return clusters ? CLUSTER_SETS[clusters - 1][i] : 0;
        }

        BarrierMapper::BarrierMapper(unsigned char config) :
            mappings_()
        {
            const unsigned char shift = static_cast<unsigned char>(config * BARRIERS_PER_GROUP);

            // Shift ids and wrap around on overflow
            for (unsigned int i = 0; i < mappings_.size(); ++i)
                mappings_[i] = (i + shift) % TOTAL_USED_BARRIERS;
        }

        unsigned char BarrierMapper::map_by_id(unsigned char b) const
        {
            assert(b < mappings_.size() && "Requested mappign for an unknown barrier");
            return mappings_[b];
        }

        unsigned long long BarrierMapper::map_by_mask(unsigned long long mask) const
        {
            unsigned long long result = 0;

            for (unsigned char b = 0; mask > 0; mask >>= 1, ++b)
                if (mask & 1)
                    result |= (1ull << map_by_id(b));

            return result;
        }

        NNCmxTranslationTable NNCmxTranslationTable::fromAbsoluteMask(unsigned int sliceMask)
        {
            NNCmxTranslationTable table;
            unsigned int relativeSlices = 0;

            for (unsigned int absoluteSlice = 0; absoluteSlice < MAX_SLICES; ++absoluteSlice)
            {
                if (sliceMask & (1u << absoluteSlice))
                {
                    assert((1u << relativeSlices) < sizeof(table.mappings_) / sizeof(table.mappings_[0]) && "Too many slice mappings");
                    table.mappings_[1u << relativeSlices] = static_cast<char>(1u << absoluteSlice);
                    ++relativeSlices;
                }
            }

            // Single relative slice masks have absolute masks assigned.
            // Now combine them for the rest of the relative masks.

            for (unsigned int i = 1; i < (1u << relativeSlices); ++i)
            {
                if (table.mappings_[i] == -1)
                {
                    int mapping = 0;

                    for (unsigned int j = 0; j < relativeSlices; ++j)
                        if ((1u << j) & i)
                            mapping |= static_cast<int>(table.mappings_[1u << j]);

                    if (mapping > 0)
                        table.mappings_[i] = static_cast<char>(mapping);
                }
            }

            return table;
        }

        void NNCmxTranslationTable::print() const
        {
            nnLog(MVLOG_INFO, "TT: %d, %d, %d, %d", mappings_[0], mappings_[1], mappings_[2], mappings_[3]);
        }

        NNCmxAddressSolver::NNCmxAddressSolver(const Buffer &nn_cmx_base, NNCmxTranslationTable table) :
            nn_cmx_base_(nn_cmx_base),
            table_(table)
        {
        }

        Buffer NNCmxAddressSolver::solve(unsigned char relativeMask) const
        {
            assert(0 < relativeMask && "Requested to solve the void. There has to be at least one slice in the RelativeMask");

            unsigned char absoluteMask = toAbsoluteMask(relativeMask);
            unsigned int broadcastMask = toBroadcastMask(absoluteMask) << 20;
            return Buffer((nn_cmx_base_.addr() & ADDRESS_MASK) | broadcastMask, nn_cmx_base_.size());
        }

        unsigned char NNCmxAddressSolver::toAbsoluteMask(unsigned char relativeMask) const
        {
            char absoluteMask = 0;

            assert(relativeMask < sizeof(table_.mappings_) && "RelativeMask out of bounds");
            absoluteMask = table_.mappings_[relativeMask];

            assert(absoluteMask != -1 && "RelativeMask requires more physical slices than allocated. Check for correct resource requirements information in the header.");

            return (unsigned char)absoluteMask;
        }

        unsigned int NNCmxAddressSolver::toBroadcastMask(unsigned int absoluteMask) const
        {
            /*
              * From 14_10_NCE_CMX.odt
              ******************************
              * 12 - 0b1100 - 0x4
              *  6 - 0b0110 - 0x5
              *  3 - 0b0011 - 0x6
              *  9 - 0b1001 - 0x7
              * 14 - 0b1110 - 0x8
              *  7 - 0b0111 - 0x9
              * 11 - 0b1011 - 0xa
              * 13 - 0b1101 - 0xb
              * 15 - 0b1111 - 0xc, 0xd, 0xe
              *
              *  5 - 0x0101 - ???
              * 10 - 0x1010 - ???
              *
              *  1 - 0x0001 - 0
              *  2 - 0x0010 - 1
              *  4 - 0x0100 - 2
              *  8 - 0x1000 - 3
              *
              *  0 - 0x0000 - 0
            */

            static const char broadcast_masks[] =
            {
                0x0, 0x0, 0x1, 0x6,
                0x2, -1, 0x5, 0x9,
                0x3, 0x7, -1, 0xa,
                0x4, 0xb, 0x8, 0xc,
            };

            assert(0 < absoluteMask && "AbsoluteMask doesn't contain any slices");
            assert(absoluteMask < sizeof(broadcast_masks) / sizeof(broadcast_masks[0]) && "AbsoluteMask too large");
            char broadcastMask = broadcast_masks[absoluteMask];

            assert(0 <= broadcastMask && "The HW doesn't support broadcasting to the chosen slices");
            return static_cast<unsigned int>(broadcastMask);
        }

        SimplifiedTensorLayout::SimplifiedTensorLayout() :
            line_stride_(0),
            line_length_(0),
            plane_stride_(0),
            plane_length_(0),
            total_length_(0)
        {
        }

        bool SimplifiedTensorLayout::load(unsigned int dims, const unsigned char *order, const float *strides, const unsigned int *sizes)
        {
            unsigned int line_stride_in_bits = 0;
            unsigned int plane_stride_in_bits = 0;
            unsigned int *rt_dims[SimplifiedTensorLayout::STRIDING_LEVELS] = { &line_length_, &plane_length_ };
            unsigned int *rt_strides[SimplifiedTensorLayout::STRIDING_LEVELS] = { &line_stride_in_bits, &plane_stride_in_bits };

            auto bit_strides = [&](unsigned int i) -> unsigned int { return static_cast<unsigned int>(strides[i] * CHAR_BIT); };

            unsigned int previous_size = 1;
            unsigned int previous_stride = bit_strides(0);
            unsigned int total_length_in_bits = bit_strides(0);

            for (unsigned int dim = 0, level = 0; dim < dims; ++dim)
            {
                total_length_in_bits *= sizes[order[dim]];

                if (previous_size * previous_stride < bit_strides(1 + order[dim]) &&
                    sizes[order[dim]] > 1)
                {
                    if (level >= SimplifiedTensorLayout::STRIDING_LEVELS)
                    {
                        nnLog(MVLOG_ERROR, "Max striding levels exceeded");
                        return false;
                    }
                    *rt_strides[level] = bit_strides(1 + order[dim]);
                    *rt_dims[level] = (previous_size * previous_stride) / (level ? *rt_strides[level - 1] : CHAR_BIT);
                    ++level;
                }

                previous_size = sizes[order[dim]];
                previous_stride = bit_strides(1 + order[dim]);
            }

            line_stride_  = line_stride_in_bits / CHAR_BIT;
            plane_stride_ = plane_stride_in_bits / CHAR_BIT;
            total_length_ = total_length_in_bits / CHAR_BIT;

            return true;
        }

        void SimplifiedTensorLayout::print() const
        {
            nnLog(MVLOG_INFO, "STL: line %u / %u, plane %u / %u, total %u / -",
                line_length_, line_stride_, plane_length_, plane_stride_, total_length_);
        }

        Buffer RelativeAddress::resolve32(const NNRelocationData &rd, Class c, bool *overflow) const
        {
            Buffer buffer = resolve(rd, c, overflow);

            #ifdef CONFIG_NN_CHECK_ADDRESSES
            if ((buffer.addr() >> 32) != 0)
                nnLog(MVLOG_WARN, "resolve32: Truncating address 0x%llx greater than 32 bits\n", addr);
            #endif

            return Buffer(buffer.addr32(), buffer.size());
        }

        Buffer RelativeAddress::resolve(const NNRelocationData &rd, Class c, bool *overflow) const
        {
            switch (location_)
            {
//                case Location::Blob:
//                    return rd.upa_rd_.blob_buffers_[index_].add(offset(c), overflow);
//
//                case Location::Input:
//                    return rd.upa_rd_.inputs_[index_].add(offset(c), overflow);
//
//                case Location::Output:
//                    return rd.upa_rd_.outputs_[index_].add(offset(c), overflow);
//
//                case Location::Heap:
//                    return rd.upa_rd_.ddr_heap_.add(offset(c), overflow);
//
                case Location::DDR:
                    return rd.upa_rd_.ddr_bss_.add(offset(c), overflow);

                case Location::UPA_CMX:
                    return rd.upa_rd_.upa_cmx_.add(offset(c), overflow);

                case Location::NN_CMX:
                    return rd.nn_solver_.solve(static_cast<unsigned char>(index_)).add(offset(c), overflow);

//                case Location::Absolute:
//                    return Buffer(offset(c), std::numeric_limits<unsigned int>::max() - offset(c) + 1);

                default:
                    nnLog(MVLOG_WARN, "Unexpected relative location: %d\n", location_);
                    print();
                    return Buffer();
            }
        }

        void RelativeAddress::print() const
        {
            nnLog(MVLOG_WARN, "RelativeAddress: L: %u, I: %u, DO: %u = 0x%x, SM: %u = 0x%x, ST: %u = 0x%x",
                static_cast<unsigned int>(location_), index_, data_offset_, data_offset_, sparsity_map_offset_, sparsity_map_offset_, sparsity_table_offset_, sparsity_table_offset_);
        }

        unsigned int RelativeAddress::offset(Class c) const
        {
            switch (c)
            {
                case Class::Data:
                    return data_offset_;

                case Class::SparsityMap:
                    return sparsity_map_offset_;

                case Class::SparsityTable:
                    return sparsity_table_offset_;

                case Class::Base:
                default:
                    return 0;
            }
        }

        uint32_t RelativeAddress::to_dpu_multicast(uint32_t addr, unsigned int &offset1, unsigned int &offset2, unsigned int &offset3)
        {
            const unsigned int bare_ptr = addr & ADDRESS_MASK;
            const unsigned int broadcast_mask = (addr & ~ADDRESS_MASK) >> 20;

            static const unsigned short multicast_masks[] =
            {
                0x0000, 0x0001, 0x0002, 0x0003,
                0x0012, 0x0011, 0x0010, 0x0030,
                0x0211, 0x0210, 0x0310, 0x0320,
                0x3210, 0x3210, 0x3210, 0x3210,
            };

            assert(broadcast_mask < sizeof(multicast_masks) / sizeof(multicast_masks[0]) && "Broadcast mask out of range");
            const unsigned short multicast_mask = multicast_masks[broadcast_mask];

            assert(multicast_mask != 0xffff && "Got an invalid multicast mask");

            unsigned int base_mask = (multicast_mask & 0xf) << 20;
            offset1 *= (multicast_mask >> 4) & 0xf;
            offset2 *= (multicast_mask >> 8) & 0xf;
            offset3 *= (multicast_mask >> 12) & 0xf;

            return bare_ptr | base_mask;
        }

        uint32_t RelativeAddress::to_dpu_multicast_base(uint32_t addr)
        {
            unsigned int offset1, offset2, offset3;
            return to_dpu_multicast(addr, offset1, offset2, offset3);
        }
    }
}
