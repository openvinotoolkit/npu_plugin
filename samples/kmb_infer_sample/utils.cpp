//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "utils.hpp"

#include <map>
#include <limits>
#include <string>
#include <algorithm>
#include <vector>
#include <cfloat>

#include <graph_tools.hpp>

std::vector<std::string> readLabelsFromFile(const std::string& labelFileName) {
    std::vector<std::string> labels;

    std::ifstream inputFile;
    inputFile.open(labelFileName, std::ios::in);
    if (inputFile.is_open()) {
        std::string strLine;
        while (std::getline(inputFile, strLine)) {
            trim(strLine);
            labels.push_back(strLine);
        }
    }
    return labels;
}

void get_common_dims(const IE::Blob::Ptr blob,
                     size_t &dimx,
                     size_t &dimy,
                     size_t &dimz,
                     size_t &dimn) {
    IE::SizeVector dims = blob->getTensorDesc().getDims();
    dimn = 1;
    if (dims.size() == 2) {
        dimz = 1;
        dimy = dims[0];
        dimx = dims[1];
    } else if (dims.size() == 3) {
        dimx = dims[2];
        dimy = dims[1];
        dimz = dims[0];
    } else if (dims.size() == 4) {
        dimx = dims[3];
        dimy = dims[2];
        dimz = dims[1];

        if (dims[0] != 1) {
            dimn = dims[0];
        }
    }
}

using LayerProcessor = std::function<IE::Blob::Ptr(const IE::CNNLayerPtr, IE::Blob::Ptr)>;

IE::Blob::Ptr pooling(const IE::CNNLayerPtr layer, IE::Blob::Ptr src) {
    auto poolLayer = std::dynamic_pointer_cast<IE::PoolingLayer>(layer);

    if (poolLayer == nullptr) {
        THROW_IE_EXCEPTION << "PoolLayer is nullptr.";
    }

    size_t KW = poolLayer->_kernel_x;
    size_t KH = poolLayer->_kernel_y;

    size_t SH = poolLayer->_stride_y;
    size_t SW = poolLayer->_stride_x;

    int PH = poolLayer->_padding_y;
    int PW = poolLayer->_padding_x;

    bool isAvgPooling = (poolLayer->_type == IE::PoolingLayer::PoolType::AVG);
    bool excludePad = poolLayer->_exclude_pad;

    size_t IW, IH, _;
    get_common_dims(src, IW, IH, _, _);

    auto tensorDesc = layer->outData[0]->getTensorDesc();
    tensorDesc.setPrecision(IE::Precision::FP32);
    auto dst = IE::make_shared_blob<float>(tensorDesc);
    dst->allocate();

    size_t OW, OH, OC;
    get_common_dims(dst, OW, OH, OC, _);

    const auto *src_data = src->cbuffer().as<const float *>();
    auto *dst_data = dst->buffer().as<float *>();

    for (size_t c = 0; c < OC; c++) {
        for (size_t oh = 0; oh < OH; oh++) {
            for (size_t ow = 0; ow < OW; ow++) {
                size_t oidx = c * OH * OW + oh * OW + ow;
                float out_ref = isAvgPooling ? static_cast<float>(0) : -FLT_MAX;

                for (uint32_t kh = 0; kh < KH; kh++) {
                    for (uint32_t kw = 0; kw < KW; kw++) {
                        int32_t iw = ow * SW - PW + kw;
                        int32_t ih = oh * SH - PH + kh;
                        if (iw < 0 || static_cast<size_t>(iw) >= IW || ih < 0
                            || static_cast<size_t>(ih) >= IH)
                            continue;
                        uint32_t iidx = c * IH * IW + ih * IW + iw;

                        float d = src_data[iidx];
                        out_ref = isAvgPooling ? out_ref + d : std::max(out_ref, d);
                    }
                }

                if (isAvgPooling) {
                    int w_beg = ow * SW - PW;
                    int w_end = w_beg + KW;
                    int h_beg = oh * SH - PH;
                    int h_end = h_beg + KH;

                    w_beg = excludePad ? std::max<int>(w_beg, 0) : std::max<int>(w_beg, -PW);
                    h_beg = excludePad ? std::max<int>(h_beg, 0) : std::max<int>(h_beg, -PH);

                    w_end = excludePad ? std::min<int>(w_end, IW) : std::min<int>(w_end, IW + PW);
                    h_end = excludePad ? std::min<int>(h_end, IH) : std::min<int>(h_end, IH + PH);

                    out_ref /= (h_end - h_beg) * (w_end - w_beg);
                }

                dst_data[oidx] = out_ref;
            }
        }
    }

    return dst;
}

IE::Blob::Ptr scaleShift(const IE::CNNLayerPtr layer, IE::Blob::Ptr src) {
    auto scaleLayer = std::dynamic_pointer_cast<IE::ScaleShiftLayer>(layer);

    auto tensorDesc = layer->outData[0]->getTensorDesc();
    tensorDesc.setPrecision(IE::Precision::FP32);
    auto dst = IE::make_shared_blob<float>(tensorDesc);
    dst->allocate();

    auto& in_size = src->getTensorDesc().getDims();
    IE::Layout layout = src->getTensorDesc().getLayout();

    if (scaleLayer == nullptr) {
        THROW_IE_EXCEPTION << "ScaleLayer is nullptr.";
    }

    auto* weights = scaleLayer->_weights->cbuffer().as<float*>();
    const float* bias_data = nullptr;
    if (scaleLayer->_biases) {
        bias_data = weights + in_size[in_size.size() - 3];
    }

    float* dst_data = dst->buffer();
    const float* src_data = src->buffer();

    size_t N1, C1, H1, W1;
    get_common_dims(dst, W1, H1, C1, N1);

    for (size_t n = 0; n < N1; n++) {
        for (size_t c = 0; c < C1; c++) {
            float val = 0.0f;
            if (bias_data)
                val = bias_data[c];
            for (size_t h = 0; h < H1; h++) {
                for (size_t w = 0; w < W1; w++) {
                    size_t iidx = layout == IE::Layout::NCHW ?
                                  w + h * W1 + c * W1 * H1 + n * W1 * H1 * C1 :
                                  c + w * C1 + h * C1 * W1 + n * W1 * H1 * C1;
                    dst_data[iidx] = val + src_data[iidx] * weights[c];
                }
            }
        }
    }

    return dst;
}

IE::Blob::Ptr relu(const IE::CNNLayerPtr layer, IE::Blob::Ptr src) {
    auto reluLayer = std::dynamic_pointer_cast<IE::ReLULayer>(layer);

    auto tensorDesc = layer->outData[0]->getTensorDesc();
    tensorDesc.setPrecision(IE::Precision::FP32);
    auto dst = IE::make_shared_blob<float>(tensorDesc);
    dst->allocate();

    if (reluLayer == nullptr) {
        THROW_IE_EXCEPTION << "ReluLayer is nullptr.";
    }

    float negative_slope = reluLayer->negative_slope;

    float *srcData = src->buffer();
    float *dstData = dst->buffer();
    size_t count = src->size();
    for (size_t indx = 0; indx < count; ++indx) {
        float inpt = srcData[indx];
        float val = std::max(inpt, 0.0f) + negative_slope * std::min(inpt, 0.0f);
        dstData[indx] = val;
    }

    return dst;
}

int getOffset(const IE::SizeVector& coordinates, const IE::SizeVector& strides) {
    size_t offset = 0;
    for (size_t i = 0; i < coordinates.size(); ++i) {
        offset += coordinates[i] * strides[i];
    }
    return offset;
}

void incrementCoordinates(IE::SizeVector& coordinates, const IE::SizeVector& dims) {
    for (size_t d = 0, nAdd = 1; d < coordinates.size() && nAdd == 1 ; ++d) {
        coordinates[d] = (coordinates[d] == dims[d] - 1) ? 0 : coordinates[d] + 1;
        nAdd = (coordinates[d] == 0) ? 1 : 0;
    }
}

IE::Blob::Ptr softmax(const IE::CNNLayerPtr layer, IE::Blob::Ptr src) {
    auto softmaxLayer = std::dynamic_pointer_cast<IE::SoftMaxLayer>(layer);
    auto dst = IE::make_shared_blob<float>(layer->outData[0]->getTensorDesc());
    dst->allocate();

    if (softmaxLayer == nullptr) {
        THROW_IE_EXCEPTION << "SoftmaxLayer is nullptr.";
    }

    int axis = softmaxLayer->axis;

    IE::SizeVector tensorSizes = src->getTensorDesc().getDims();
    std::reverse(tensorSizes.begin(), tensorSizes.end());

    IE::SizeVector tensorStrides(tensorSizes.size());
    axis = tensorSizes.size() - 1 - axis;

    float *srcData = src->buffer();
    float *dstData = dst->buffer();

    size_t totalElements = 1;
    size_t totalLines = 1;

    for (size_t i = 0; i < tensorSizes.size(); ++i) {
        tensorStrides[i] = totalElements;
        totalElements *= tensorSizes[i];
    }
    size_t axisSize = tensorSizes[axis];
    size_t axisStride = tensorStrides[axis];
    tensorSizes.erase(tensorSizes.begin() + axis);
    tensorStrides.erase(tensorStrides.begin() + axis);
    totalLines = totalElements / axisSize;

    std::vector<float> temp(axisSize);

    IE::SizeVector tensorCoordinates(tensorSizes.size());

    const float *srcLine;
    float *dstLine;

    for (size_t nLine = 0; nLine < totalLines; ++nLine) {
        int offset = getOffset(tensorCoordinates, tensorStrides);

        srcLine = srcData + offset;
        dstLine = dstData + offset;
        float largest = std::numeric_limits<float>::lowest();
        for (size_t i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            float val = srcLine[ind];
            largest = std::max(val, largest);
        }
        float sum = 0.0f;
        for (size_t i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            float val = srcLine[ind];
            temp[i2] = std::exp(val - largest);
            sum += temp[i2];
        }
        for (size_t i2 = 0; i2 < axisSize; ++i2) {
            int ind = i2 * axisStride;
            dstLine[ind] = temp[i2] / sum;
        }
        incrementCoordinates(tensorCoordinates, tensorSizes);
    }

    return dst;
}

void transformBolb(IE::Blob::Ptr blob, IE::Precision target_precision) {
    float* data = blob->buffer();
    float min_value = 0.f;
    float max_value = 0.f;

    if (target_precision == IE::Precision::U8) {
        min_value = static_cast<float>(std::numeric_limits<uint8_t>::min());
        max_value = static_cast<float>(std::numeric_limits<uint8_t>::max());
    } else if (target_precision == IE::Precision::I8) {
        min_value = static_cast<float>(std::numeric_limits<int8_t>::min());
        max_value = static_cast<float>(std::numeric_limits<int8_t>::max());
    }

    for (size_t i = 0; i < blob->size(); ++i) {
        data[i] = std::max(min_value, std::min(max_value, data[i]));
    }
}

IE::Blob::Ptr convolution(const IE::CNNLayerPtr layer, IE::Blob::Ptr src) {
    auto convLayer = std::dynamic_pointer_cast<IE::ConvolutionLayer>(layer);

    auto tensorDesc = layer->outData[0]->getTensorDesc();
    tensorDesc.setPrecision(IE::Precision::FP32);
    auto dst = IE::make_shared_blob<float>(tensorDesc);
    dst->allocate();

    if (convLayer == nullptr) {
        THROW_IE_EXCEPTION << "ConvolutionLayer is nullptr.";
    }

    const float *bias_data = nullptr;
    if (convLayer->_biases) {
        bias_data = convLayer->_biases->buffer();
    }
    const float *weights_data = convLayer->_weights->buffer();

    size_t KW = convLayer->_kernel_x;
    size_t KH = convLayer->_kernel_y;
    size_t KD = convLayer->_kernel.size() > IE::Z_AXIS ? convLayer->_kernel[IE::Z_AXIS] : 1lu;

    size_t SW = convLayer->_stride_x;
    size_t SH = convLayer->_stride_y;
    size_t SD = convLayer->_stride.size() > IE::Z_AXIS ? convLayer->_stride[IE::Z_AXIS] : 0lu;

    size_t DW = convLayer->_dilation_x;
    size_t DH = convLayer->_dilation_y;
    size_t DD = convLayer->_dilation.size() > IE::Z_AXIS ? convLayer->_dilation[IE::Z_AXIS] : 0lu;

    size_t PW = convLayer->_padding_x;
    size_t PH = convLayer->_padding_y;
    size_t PD = convLayer->_padding.size() > IE::Z_AXIS ? convLayer->_padding[IE::Z_AXIS] : 0lu;

    size_t GC = convLayer->_group;

    auto src_dims = src->getTensorDesc().getDims();
    size_t IC = src_dims[1];
    size_t ID = (src_dims.size() == 5lu) ? src_dims[2] : 1lu;
    size_t IH = src_dims.at(src_dims.size() - 2);
    size_t IW = src_dims.back();

    auto dst_dims = dst->getTensorDesc().getDims();
    size_t OW = dst_dims.back();
    size_t OH = dst_dims.at(dst_dims.size() - 2);
    size_t OD = (dst_dims.size() == 5lu) ? dst_dims[2] : 1lu;
    size_t OC = dst_dims[1];

    const auto *src_data = src->cbuffer().as<const float *>();
    auto *dst_data = dst->buffer().as<float *>();

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t od = 0; od < OD; od++) {
                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        size_t oidx = g * OC / GC * OD * OH * OW
                                      + oc * OD * OH * OW
                                      + od * OH * OW
                                      + oh * OW
                                      + ow;
                        if (bias_data)
                            dst_data[oidx] = bias_data[g * OC / GC + oc];

                        for (size_t ic = 0; ic < IC / GC; ic++) {
                            for (size_t kd = 0; kd < KD; kd++) {
                                for (size_t kh = 0; kh < KH; kh++) {
                                    for (size_t kw = 0; kw < KW; kw++) {
                                        int32_t iw = ow * SW - PW + kw * DW;
                                        int32_t ih = oh * SH - PH + kh * DH;
                                        int32_t id = od * SD - PD + kd * DD;
                                        if (iw < 0 || iw >= (int32_t) IW ||
                                            ih < 0 || ih >= (int32_t) IH ||
                                            id < 0 || id >= (int32_t) ID)
                                            continue;
                                        size_t iidx = g * IC / GC * ID * IH * IW
                                                      + ic * ID * IH * IW
                                                      + id * IH * IW
                                                      + ih * IW
                                                      + iw;
                                        size_t widx = g * OC / GC * IC / GC * KD * KH * KW
                                                      + oc * IC / GC * KD * KH * KW
                                                      + ic * KD * KH * KW
                                                      + kd * KH * KW
                                                      + kh * KW
                                                      + kw;

                                        dst_data[oidx] += src_data[iidx] * weights_data[widx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (convLayer->blobs["oi-scale"]) {
        const float *oScale = convLayer->blobs["oi-scale"]->buffer().as<const float *>();
        size_t N, C, H, W;
        get_common_dims(dst, W, H, C, N);

        if (layer->outData[0]->getTensorDesc().getPrecision() != IE::Precision::FP32) {
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t idx = c * H * W
                                         + h * W
                                         + w;

                            dst_data[idx] = std::floor(oScale[c] * dst_data[idx]);
                        }
                    }
                }
            }
        }
    }

    return dst;
}

IE::Blob::Ptr eltwise(const IE::CNNLayerPtr layer, IE::Blob::Ptr left, IE::Blob::Ptr right) {
    auto eltwiseLayer = std::dynamic_pointer_cast<IE::EltwiseLayer>(layer);

    auto dst = IE::make_shared_blob<float>(left->getTensorDesc());
    dst->allocate();

    float* data = dst->buffer();
    const float* left_data = left->cbuffer().as<const float*>();
    const float* right_data = right->cbuffer().as<const float*>();;

    if (eltwiseLayer == nullptr) {
        THROW_IE_EXCEPTION << "EltwiseLayer is nullptr.";
    }

    if (eltwiseLayer->_operation == IE::EltwiseLayer::eOperation::Sum) {
        for (size_t i = 0; i < dst->size(); ++i) {
            data[i] = left_data[i] + right_data[i];
        }
    }

    return dst;
}

IE::Blob::Ptr input(const IE::CNNLayerPtr layer, IE::Blob::Ptr src) {
    return src;
}

IE::Blob::Ptr processNetwork(const IE::CNNNetwork& network, const IE::Blob::Ptr src) {
    static std::map<std::string, LayerProcessor> layerProcessors {
        { "Pooling" , pooling },
        { "ScaleShift", scaleShift},
        { "ReLU", relu},
        { "SoftMax", softmax},
        { "Convolution", convolution}
    };

    auto layers = CNNNetSortTopologicallyEx(network, InferenceEngine::details::default_order);
    std::map<std::string, IE::Blob::Ptr> outputBlobs;

    if (layers.empty())
        return src;

    for (auto& layer : layers) {
        if (layer->type == "Input") {
            outputBlobs[layer->name] = input(layer, src);
        } else if (layerProcessors.count(layer->type)) {
            outputBlobs[layer->name] = layerProcessors[layer->type](layer, outputBlobs[layer->insData[0].lock()->getName()]);
        } else if (layer->type == "Eltwise") {  // Make all comparisons caseless
            outputBlobs[layer->name] = eltwise(layer, outputBlobs[layer->insData[0].lock()->getName()], outputBlobs[layer->insData[1].lock()->getName()]);
        } else if (!layer->insData.empty()) {
            outputBlobs[layer->name] = outputBlobs[layer->insData[0].lock()->getName()];
        }
    }

    return src;
}

IE::CNNNetwork getNetwork(const std::string& model_path) {
    IE::CNNNetReader reader;
    reader.ReadNetwork(model_path);
    reader.ReadWeights(fileNameNoExt(model_path) + ".bin");
    return reader.getNetwork();
}

IE::Blob::Ptr preprocessUncompiledLayers(const std::string &layersPath, const std::string& data) {
    auto net = getNetwork(layersPath);

    IE::Blob::Ptr inputblob;
    if (net.getInputsInfo().empty()) {
        THROW_IE_EXCEPTION << "net.getInputsInfo() is empty!";
    }
    inputblob = IE::make_shared_blob<float>(net.getInputsInfo().begin()->second->getTensorDesc());
    auto blobData = inputblob->buffer().as<IE::PrecisionTrait<IE::Precision::U8>::value_type*>();
    std::copy(data.begin(), data.end(), blobData);

    return processNetwork(net, inputblob);
}

IE::Blob::Ptr postprocessUncompiledLayers(const std::string &layersPath, IE::Blob::Ptr src) {
    return processNetwork(getNetwork(layersPath), src);
}
