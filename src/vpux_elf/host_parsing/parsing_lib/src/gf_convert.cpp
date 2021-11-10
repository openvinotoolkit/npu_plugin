#include <gf_convert.h>
#include <stdio.h>
#include <pipePrintInit.h>

using nn::common_runtime::RelativeAddress;

namespace parsing_lib {
template <typename T>
void convert_vector(const flatbuffers::Vector<T> *fbVec, std::vector<T> &refVec) {
    if (!fbVec) {
        refVec.resize(0);
        return;
    }

    refVec.resize(fbVec->size());
    for (uint i = 0; i < fbVec->size(); i++)
        refVec[i] = fbVec->Get(i);
}

template <typename S, typename D>
void convert_enum(S src, D &dst) {
    dst = static_cast<D>(static_cast<typename std::underlying_type<S>::type>(src));
}

RelativeAddress::Location convertMemoryLocation(MemoryLocation ml) {
    switch (ml) {
        case MemoryLocation::ProgrammableInput:
            return RelativeAddress::Location::Input;
        case MemoryLocation::ProgrammableOutput:
            return RelativeAddress::Location::Output;
        case MemoryLocation::VPU_DDR_Heap:
            return RelativeAddress::Location::Heap;
        case MemoryLocation::GraphFile:
            return RelativeAddress::Location::Blob;
        case MemoryLocation::GFEmbeddedKernel:
            return RelativeAddress::Location::BlobKernels;
        case MemoryLocation::VPU_CMX_NN:
            return RelativeAddress::Location::NnCmx;
        case MemoryLocation::VPU_DDR_BSS:
            return RelativeAddress::Location::Bss;
        case MemoryLocation::AbsoluteAddr:
            return RelativeAddress::Location::Absolute;
        case MemoryLocation::KernelsBuffer:
            return RelativeAddress::Location::KernelsBuffer;
        case MemoryLocation::MAC_Accumulators:
            return RelativeAddress::Location::MAC_Accumulators;
        default:
            return RelativeAddress::Location::None;
    }
}

bool convertRelativeAddress(const TensorReference &tr, RelativeAddress &ra) {
    RelativeAddress::Location location = convertMemoryLocation(tr.locale);
    unsigned short index = 0;
    unsigned int data_offset = 0;
    unsigned int sparsity_map_offset = 0;
    unsigned int sparsity_table_offset = 0;

    if (location == RelativeAddress::Location::NnCmx) {
        for (auto bit : tr.locale_index) {
            if (bit >= 4) {
                printf("ERROR: Broadcast destination too large at locale %d, locale_index %d\n", static_cast<int>(location), bit);
                return false;
            }

            index |= (1 << bit);
        }

        if (index == 0) {
            printf("TR: %p ^%s^ %d %d %d %d\n", &tr, tr.name.c_str(), tr.data.data_index, tr.locale, tr.locale_index.size());
            printf("WARN: locale_index not set for NN CMX. Defaulting to 0\n");
            index |= (1 << 0);
        }
    } else {
        if (tr.locale_index.size() == 0) {
            printf("ERROR: Locale %d is expected to have %d locale index\n", static_cast<int>(location), tr.locale_index.size());
            return false;
        }
        index = tr.locale_index[0];
    }

    data_offset = tr.data.data_index;

    const unsigned long long INVALID = 999999999999999999ull;

    if (tr.data.sparsity_index != INVALID)
        sparsity_map_offset = tr.data.sparsity_index;

    if (tr.data.storage_element_index != INVALID)
        sparsity_table_offset = tr.data.storage_element_index;

    ra = RelativeAddress(location, index, data_offset, sparsity_map_offset, sparsity_table_offset);

    return true;
}

#define COPY_FIELD(field) pl.field = gf->field()
#define COPY_NESTED_FIELD(struct, field) pl.struct.field = gf->struct()->field()
#define COPY_ENUM(field) convert_enum(gf->field(), pl.field)
#define COPY_VECTOR(field) convert_vector(gf->field(), pl.field)

void convertTensorReference(const MVCNN::TensorReference *gf, parsing_lib::TensorReference &pl) {
    pl.name = std::string(gf->name()->c_str());
    printf("TR: %p %s\n", &pl, pl.name.c_str());
    COPY_VECTOR(dimensions);
    COPY_VECTOR(strides);
    COPY_FIELD(leading_offset);
    COPY_FIELD(trailing_offset);
    COPY_NESTED_FIELD(data, data_index);
    COPY_NESTED_FIELD(data, sparsity_index);
    COPY_NESTED_FIELD(data, storage_element_index);
    COPY_NESTED_FIELD(data, storage_element_size);
    COPY_ENUM(locale);
    COPY_VECTOR(locale_index);
    COPY_ENUM(data_dtype);
    COPY_VECTOR(quant_zero);
    COPY_VECTOR(quant_mult);
    COPY_VECTOR(quant_shift);
    COPY_FIELD(quant_post_shift_right);
    COPY_FIELD(order);
    COPY_FIELD(swizzling_key);
    COPY_VECTOR(base_ptrs);
}

void convertTensorReference(const MVCNN::TensorReference *gfTensor, parsing_lib::Optional<parsing_lib::TensorReference> &ref) {
    if (gfTensor)
        convertTensorReference(gfTensor, ref.ref());
    else
        printf("TENSOR MISSING!\n");
}

void convertNNDmaTask(const MVCNN::NNDMATask *gf, parsing_lib::DMATask &pl) {
    convertTensorReference(gf->src(), pl.src);
    convertTensorReference(gf->dst(), pl.dst);

    COPY_FIELD(compression);
    COPY_FIELD(set_crit);
    COPY_FIELD(set_ord);
}

void convertInvariant(const MVCNN::NCEInvariantFields *gf, parsing_lib::Invariant &pl)
{
    COPY_ENUM(dpu_task_type);
    // PPETask
    COPY_ENUM(mpe_frequent_mode);
    COPY_FIELD(kernelH);
    COPY_FIELD(kernelW);
    COPY_FIELD(kernel_strideH);
    COPY_FIELD(kernel_strideW);
    COPY_FIELD(kernel_padLeft);
    COPY_FIELD(kernel_padRight);
    COPY_FIELD(kernel_padTop);
    COPY_FIELD(kernel_padBottom);

    convertTensorReference(gf->parent_input_tensor(), pl.parent_input_tensor);
    convertTensorReference(gf->parent_output_tensor(), pl.parent_output_tensor);
    convertTensorReference(gf->parent_weights_tensor(), pl.parent_weights_tensor);
    convertTensorReference(gf->input_data(), pl.input_data);
    convertTensorReference(gf->output_data(), pl.output_data);
    convertTensorReference(gf->weights_data(), pl.weights_data);
    convertTensorReference(gf->weights_table(), pl.weights_table);
    convertTensorReference(gf->activation_window(), pl.activation_window);

    COPY_FIELD(activation_window_channel_length);
    COPY_FIELD(odu_offset);
    COPY_FIELD(out_channel_offset);
    COPY_FIELD(is_segmented);
    COPY_FIELD(is_continued);
    COPY_FIELD(is_superdense);
    COPY_VECTOR(segment_height);
    COPY_ENUM(odu_permutation);
}

void convertVariant(const MVCNN::NCEVariantFields *gf, parsing_lib::Variant &pl)
{
    convert_vector(gf->associated_barriers()->wait_barriers(), pl.associated_barriers.wait_barriers);
    convert_vector(gf->associated_barriers()->update_barriers(), pl.associated_barriers.update_barriers);
    convert_vector(gf->associated_barriers()->virtual_wait_barriers(), pl.associated_barriers.virtual_wait_barriers);
    convert_vector(gf->associated_barriers()->virtual_update_barriers(), pl.associated_barriers.virtual_update_barriers);

    COPY_ENUM(mpe_mode);
    COPY_FIELD(padLeft);
    COPY_FIELD(padRight);
    COPY_FIELD(padTop);
    COPY_FIELD(padBottom);

    COPY_FIELD(workload_start_X);
    COPY_FIELD(workload_start_Y);
    COPY_FIELD(workload_start_Z);
    COPY_FIELD(workload_end_X);
    COPY_FIELD(workload_end_Y);
    COPY_FIELD(workload_end_Z);

    convertTensorReference(gf->profiling_data(), pl.profiling_data);
}

void convertNCE2Task(const MVCNN::NCE2Task *gf, parsing_lib::NCE2Task &pl)
{
    convertInvariant(gf->invariant(), pl.invariant);
    if (!gf->variant()) {
        pl.variant.resize(0);
        return;
    }

    pl.variant.resize(gf->variant()->size());
    for (uint i = 0; i < gf->variant()->size(); i++)
        convertVariant(gf->variant()->Get(i), pl.variant[i]);
}
} // namespace parsing_lib
