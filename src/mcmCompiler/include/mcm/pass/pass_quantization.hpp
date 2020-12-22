#ifndef QUANTIZATION_UTILS_HPP
#define QUANTIZATION_UTILS_HPP

#include "include/mcm/computation/model/base_op_model.hpp"

namespace mv
{

    enum class Precision {
        Default,
        U8,
        I8,
        FP16,
        I32,
        FP32,
    };

    DType getDType(Precision p);
    int64_t calculateZeroPoint(
        float low,
        float high,
        DType dtype,
        int levels = 256);
    double calculateScales(
        float low,
        float high,
        int levels = 256);
    void calcZeroPointAndScalePerTensor(
        double floatMax,
        double floatMin,
        double& quantScale,
        int64_t& quantZp,
        DType dtype,
        int64_t levels = 256);
    void calcZeroPointAndScalePerChannel(
        std::vector<double> &floatMax,
        std::vector<double> &floatMin,
        std::vector<double> &quantScale,
        std::vector<int64_t> &quantZp,
        DType dtype,
        int64_t levels = 256);
    void updateInfMinMaxPerTensor(mv::Data::TensorIterator tensor);
    void updateInfMinMaxPerChannel(mv::Data::TensorIterator tensor);

    //NOTE: workaround. merge_in_one is true for activations and false for weights
    QuantizationParams extractQuantParams(
        Data::OpListIterator fqOp,
        bool merge_in_one,
        bool extract_input_params = false);

    QuantizationParams extractQuantParamsI(
        Data::OpListIterator fqOp,
        bool merge_in_one);

    QuantizationParams extractQuantParamsO(
        Data::OpListIterator fqOp,
        bool merge_in_one);

}

#endif
