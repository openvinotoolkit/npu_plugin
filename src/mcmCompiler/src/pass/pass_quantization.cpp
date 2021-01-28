#include "mcm/pass/pass_quantization.hpp"

#include <cmath>
#include <map>
#include "include/mcm/tensor/dtype/dtype.hpp"

mv::DType mv::getDType(mv::Precision p) {
    static const std::map<mv::Precision, mv::DType> types {
            {mv::Precision::Default, mv::DType("Default")},
            {mv::Precision::U8, mv::DType("UInt8")},
            {mv::Precision::I8, mv::DType("Int8")},
            {mv::Precision::FP16, mv::DType("Float16")},
            {mv::Precision::I32, mv::DType("Int32")},
            {mv::Precision::FP32, mv::DType("Float32")},
    };

    return types.at(p);
}

// Using double low high values instead of float seems
// to produce a high enough variance, which impacts
// quantized networks accuracy, some have slightly better accuracy
// while the most of them have slightly worse accuracy.
// vehicle_license_plate_detection_barrier is a regression example
// With double arguments, input layer ZP is assigned as
// 127 instead of 128
int64_t mv::calculateZeroPoint(
    float low,
    float high,
    int levels,
    mv::DType dtype)
{
    if ((low > 0.f) || (high < 0.f) || (low == high))
        throw std::runtime_error("Unsupported FQ low/high");
    if (levels > 256)
        throw std::runtime_error("Unsupported FQ levels");

    int64_t zeroPoint = 0;

    if (dtype == getDType(Precision::U8)) {
        float x = -static_cast<float>(levels - 1.0f) * low / (high - low);
        zeroPoint = static_cast<int64_t>(std::round(x));
    }
    if (dtype == getDType(Precision::I8)) {
        float x = -static_cast<float>(levels - 1.0f) * ((high + low) * 0.5f) / (high - low);
        zeroPoint = static_cast<int64_t>(std::round(x));
    }

    return zeroPoint;
}

// Using double low high values instead of float seems
// to produce a high enough variance, which impacts
// quantized networks accuracy, some have slightly better accuracy
// while the most of them have slightly worse accuracy.
double mv::calculateScale(float low, float high, int levels) {
    if (low == high)
        throw std::runtime_error("Unsupported FQ low/high");
    if (levels > 256)
        throw std::runtime_error("Unsupported FQ levels");

    return static_cast<double>((high - low) / (levels - 1));
}

void mv::calcZeroPointAndScalePerTensor(
    double floatMax,
    double floatMin,
    int levels,
    mv::DType dtype,
    double& quantScale,
    int64_t& quantZp)
{
    quantScale = calculateScale(floatMin, floatMax, levels);
    quantZp = calculateZeroPoint(floatMin, floatMax, levels, dtype);
}

void calcZeroPointsAndScalesPerChannel(
    std::vector<double> &floatMax,
    std::vector<double> &floatMin,
    int levels,
    mv::DType dtype,
    std::vector<double> &quantScale,
    std::vector<int64_t> &quantZp)
{
    std::transform(floatMax.cbegin(), floatMax.cend(),
        floatMin.cbegin(), quantScale.begin(),
        [levels](const double &max,const double &min)
        {return mv::calculateScale(min, max, levels);});

    std::transform(floatMax.cbegin(), floatMax.cend(),
        floatMin.cbegin(), quantZp.begin(),
        [dtype, levels](const double &max, const double &min)
        {return mv::calculateZeroPoint(min, max, levels, dtype);});
}

void mv::updateInfMinMaxPerTensor(mv::Data::TensorIterator tensor)
{
    auto& tensorQuantization = tensor->get<mv::QuantizationParams>("quantParams");

    //Note: if input Tensor has min, max of infs...we need to compute them
    if (tensorQuantization.infinitelimits())
    {
        //Quantization equation Real = scale(Quantized - zeroPoint)
        double maximumFloat = tensorQuantization.getScale()[0] * (255 - tensorQuantization.getZeroPoint()[0]);
        double minimumFloat = -tensorQuantization.getZeroPoint()[0] * tensorQuantization.getScale()[0];
        if (minimumFloat == -0)
            minimumFloat = 0;

        mv::QuantizationParams newTensorQuantization(tensorQuantization.getZeroPoint(),
                                                    tensorQuantization.getScale(),{minimumFloat},{maximumFloat});
        tensor->setQuantParams(newTensorQuantization);
    }
}

void mv::updateInfMinMaxPerChannel(mv::Data::TensorIterator tensor)
{
    auto& tensorQuantization = tensor->get<mv::QuantizationParams>("quantParams");

    //Note: Do not care if populated or unpopulated....batch = 1
    if (tensorQuantization.infinitelimits())
    {
        std::vector <double> maximums, minimums;
        double maximumFloat, minimumFloat;
        for (uint32_t channel = 0; channel < tensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]; channel++)
        {
            //Quantization equation Real = scale(Quantized - zeroPoint)
            maximumFloat = tensorQuantization.getScale()[channel] * (255 - tensorQuantization.getZeroPoint()[0]);
            minimumFloat = -tensorQuantization.getZeroPoint()[0] * tensorQuantization.getScale()[channel];
            if (minimumFloat == -0)
                minimumFloat = 0;
            maximums.push_back(maximumFloat);
            minimums.push_back(minimumFloat);
        }
        mv::QuantizationParams newTensorQuantization(tensorQuantization.getZeroPoint(),
                                                    tensorQuantization.getScale(),minimums, maximums);
        tensor->setQuantParams(newTensorQuantization);
    }
}

//NOTE: workaround. merge_in_one is true for activations and false for weights
mv::QuantizationParams mv::extractQuantParams(mv::Data::OpListIterator fqOp, bool merge_in_one, bool extract_input_params) {
    if (fqOp->getOpType() != "FakeQuantize")
        throw std::runtime_error("extractQuantParams works only with FQ layers");

    auto inputs = fqOp->getInputTensor();
    auto attrs = fqOp->getAttrs();

    auto levels = fqOp->get<unsigned>("levels");

    std::vector<double> min_range;
    std::vector<double> max_range;

    if (extract_input_params) {
        min_range = fqOp->getInputTensor(1)->getDoubleData();
        max_range = fqOp->getInputTensor(2)->getDoubleData();
    } else {
        min_range = fqOp->getInputTensor(3)->getDoubleData();
        max_range = fqOp->getInputTensor(4)->getDoubleData();
    }

    if (min_range.size() != max_range.size() || min_range.empty())
        throw std::runtime_error("Unsupported FQ low/high");

    std::vector<int64_t> zero_points;
    std::vector<double> scales;
    std::vector<double> min;
    std::vector<double> max;
    if (merge_in_one) {
        float output_min_value = *std::min_element(min_range.begin(), min_range.end());
        float output_max_value = *std::max_element(max_range.begin(), max_range.end());

        zero_points.push_back(calculateZeroPoint(output_min_value, output_max_value, levels, getDType(Precision::U8)));
        scales.push_back(calculateScale(output_min_value, output_max_value, levels));
        min.push_back(output_min_value);
        max.push_back(output_max_value);
    } else {
        for (size_t i = 0; i < min_range.size(); ++i) {
            float min_value = min_range[i];
            float max_value = max_range[i];

            zero_points.push_back(calculateZeroPoint(min_value, max_value, levels, getDType(Precision::U8)));
            scales.push_back(calculateScale(min_value, max_value, levels));
            min.push_back(min_value);
            max.push_back(max_value);
        }
    }

    return mv::QuantizationParams{zero_points, scales, min, max};
}

mv::QuantizationParams mv::extractQuantParamsI(mv::Data::OpListIterator fqOp, bool merge_in_one) {
    return extractQuantParams(fqOp, merge_in_one, true);
}

mv::QuantizationParams mv::extractQuantParamsO(mv::Data::OpListIterator fqOp, bool merge_in_one) {
    return extractQuantParams(fqOp, merge_in_one, false);
}
