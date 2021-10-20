#include <data_types.h>
#include <utils.h>
#include <algorithm>
#include <tuple>

constexpr uint32_t CHAR_BIT = 8;

namespace parsing_lib {

int decode_storage_order(const TensorReference &t, unsigned char *order) {
    const auto &dims = t.dimensions;
    const auto &strides = t.strides;
    const unsigned int S = dims.size();

    // printf("STRIDES %d: ", strides.size());
    // for (float f : strides)
    //     printf("%f ", f);
    // printf("\n");
    // // printf("DIMS: %d %d %d\n", dims[0], dims[1], dims[2]);
    // printf("DIMS %d: ", dims.size());
    // for (int d : dims)
    //     printf("%d ", d);
    // printf("\n");

    if (t.strides.size() != S + 1) {
        printf("ERROR: "
                "Got %lu strides for a tensor with %u dimensions. Expecting correlated strides and dimensions "
                "vectors.\n",
                strides.size(), S);
        return -1;
    }

    for (unsigned int i = 0; i < S; ++i)
        order[i] = i;

    std::sort(&order[0], &order[0] + S, [&](int lhs, int rhs) {
        return std::make_tuple(strides[lhs + 1], dims[lhs], lhs) <
                std::make_tuple(strides[rhs + 1], dims[rhs], rhs);
    });

    return S;
}

bool decode_simplified_layout(const TensorReference &t, SimplifiedTensorLayout &stl) {
    const unsigned int DIMENSIONS = t.dimensions.size();
    unsigned char order[DIMENSIONS];
    if (decode_storage_order(t, order) < 1)
        return false;

    return stl.load(DIMENSIONS, order, t.strides.data(),
                    reinterpret_cast<const unsigned int *>(t.dimensions.data()));
}

bool SimplifiedTensorLayout::load(unsigned int dims, const unsigned char *order, const float *strides,
                                  const unsigned int *sizes) {
    unsigned int line_stride_in_bits = 0;
    unsigned int plane_stride_in_bits = 0;
    unsigned int *rt_dims[SimplifiedTensorLayout::STRIDING_LEVELS] = {&line_length_, &plane_length_};
    unsigned int *rt_strides[SimplifiedTensorLayout::STRIDING_LEVELS] = {&line_stride_in_bits, &plane_stride_in_bits};

    auto bit_strides = [&](unsigned int i) -> unsigned int { return static_cast<unsigned int>(strides[i] * CHAR_BIT); };

    // printf("ST %f %f %f %f\n", strides[0], strides[1], strides[2], strides[3]);
    // printf("DIM %d %d %d %d\n", sizes[0], sizes[1], sizes[2], sizes[3]);
    // printf("ORDER: %d %d %d\n", order[0], order[1], order[2]);

    unsigned int previous_size = 1;
    unsigned int previous_stride = bit_strides(0);
    unsigned int total_length_in_bits = bit_strides(0);

    // printf("TLB: %d\n", total_length_in_bits);

    for (unsigned int dim = 0, level = 0; dim < dims; ++dim) {
        total_length_in_bits *= sizes[order[dim]];
        // printf("TLB: %d\n", total_length_in_bits);

        if (previous_size * previous_stride < bit_strides(1 + order[dim]) && sizes[order[dim]] > 1) {
            if (level >= SimplifiedTensorLayout::STRIDING_LEVELS) {
                printf("ERROR: Max striding levels exceeded\n");
                return false;
            }
            *rt_strides[level] = bit_strides(1 + order[dim]);
            *rt_dims[level] = (previous_size * previous_stride) / (level ? *rt_strides[level - 1] : CHAR_BIT);
            ++level;
        }

        previous_size = sizes[order[dim]];
        previous_stride = bit_strides(1 + order[dim]);
    }

    line_stride_ = line_stride_in_bits / CHAR_BIT;
    plane_stride_ = plane_stride_in_bits / CHAR_BIT;
    total_length_ = total_length_in_bits / CHAR_BIT;

    // printf("LS: %d PS: %d TL: %d\n", line_stride_, plane_stride_, total_length_);

    return true;
}
}
