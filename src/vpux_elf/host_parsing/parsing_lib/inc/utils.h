#pragma once
#include <nce_2p7_hw.h>

namespace parsing_lib {
uint8_t getIDUDTypeSizeBits(host_parsing::InputTensorDType dtype);
uint8_t getODUDTypeSizeBits(host_parsing::OutputTensorDType dtype);
// Convert between schema DType representation (random order) and HW which is the exact values
// registers need
host_parsing::InputTensorDType convertInputDtype(DType dtype);
host_parsing::OutputTensorDType convertOutputDtype(DType dtype);

struct SimplifiedTensorLayout {
    SimplifiedTensorLayout() : line_stride_(0), line_length_(0), plane_stride_(0), plane_length_(0), total_length_(0) {}
    enum {
        STRIDING_LEVELS = 2,
    };

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

bool decode_simplified_layout(const TensorReference &t, SimplifiedTensorLayout &stl);
}
