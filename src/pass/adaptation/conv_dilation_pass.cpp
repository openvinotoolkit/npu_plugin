#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/tiling.hpp"
#include <algorithm>

#define SCALE_RANGE_0_1 0.00392156862745098
#define ZP_RANGE_0_1 0

//NOTE: There are 2 passes implemented for dilation convolution. The one solution is based to the idea that we push
//zero points inside the weight tensor of the convolution in order to simulate dilated conv and this pass is used by emulator.
//This idea might lead to really big kernel sizes so in order to implement dilation in kmb we use the storage element to slice/concat.
static void convDilationUsingWeightsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void convDilationUsingStorageElementFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvDilationUsingStorageElement)
            .setFunc(convDilationUsingStorageElementFcn)
            .setDescription(
                "This pass dilates a kernel using new method with SEPS");
    }
    namespace pass
    {
        MV_REGISTER_PASS(ConvDilationUsingWeights)
            .setFunc(convDilationUsingWeightsFcn)
            .setDescription(
                "This pass dilates a kernel using new method with SEPS");
    }
}

mv::Data::TensorIterator createDilatedConvSubConv(mv::OpModel om, mv::Data::OpListIterator opIt, mv::Data::TensorIterator sourceTensor,
                                                    std::array<unsigned short, 4> padding, std::string name, mv::Shape newShape, size_t subConvIdx)
{
    mv::Data::TensorIterator subConv;
    //TODO handle stride != 1
    auto stride = opIt->get<std::array<unsigned short, 2>>("stride");

    mv::Data::TensorIterator sliceInput = om.slice(sourceTensor,
                                 {0, 0, 0, 0},
                                 newShape,
                                 sourceTensor->get<mv::QuantizationParams>("quantParams"),
                                 opIt->getName() + "_dilatedSlice_" + std::to_string(subConvIdx));


    sliceInput->set<bool>("dilatedSlice", true);
    auto sliceInputOp = om.getSourceOp(sliceInput);

    sliceInputOp->set<bool>("dilatedSlice", true);

    if (opIt->getOpType() == "Conv")
    {
        subConv = om.conv(sliceInput,
                opIt->getInputTensor(1),
                stride,
                padding,
                1,
                opIt->get<unsigned>("group"),
                opIt->get<mv::DType>("dType"),
                opIt->get<mv::QuantizationParams>("quantParams"),
                name);
    }
    else
    {
        subConv = om.depthwiseConv(sliceInput,
                opIt->getInputTensor(1),
                stride,
                padding,
                1,
                opIt->get<mv::DType>("dType"),
                opIt->get<mv::QuantizationParams>("quantParams"),
                name);
    }

    auto subConvOp = om.getSourceOp(subConv);
    subConvOp->set<bool>("DilatedSubConv", true);
    subConvOp->set<unsigned>("originalDilationFactor", opIt->get<unsigned>("dilationFactor"));
    sliceInput->set<mv::Shape>("originalShape", sourceTensor->getShape());
    subConv->set<mv::Shape>("originalShape", sourceTensor->getShape());
    subConvOp->set<mv::Shape>("originalShape", sourceTensor->getShape());
    subConvOp->set<unsigned>("subConvIndex", subConvIdx);
    if(opIt->hasAttr("opId"))
    {
        unsigned currentOpId = opIt->get<unsigned>("opId");
        subConvOp->set<unsigned>("opId", currentOpId);
        sliceInputOp->set<unsigned>("opId", currentOpId);
    }


    return subConv;
}

std::array<unsigned short, 4> calcNewPadding(mv::Data::OpListIterator opIt, size_t newWidth, size_t newHeight)
{
    auto oldPadding = opIt->get<std::array<unsigned short, 4>>("padding");
    auto oldStride = opIt->get<std::array<unsigned short, 2>>("stride");
    auto kernelShape = opIt->getInputTensor(1)->getShape();

    if (oldPadding[0] == 0 && oldPadding[1] == 0 && oldPadding[2] == 0 && oldPadding[3] == 0)
    {
        //Valid padding
        return oldPadding;
    }
    else
    {

        //same padding
        //case of stride 1
        //p = ((f-1)*s -n +f )/2
        //unsigned int p1 = ((newWidth - 1) * oldStride[0] - kernelShape[mv::KERNEL_WIDTH] + newWidth)/2;
        //unsigned int p2 = ((newHeight - 1) * oldStride[1] - kernelShape[mv::KERNEL_HEIGHT] + newHeight)/2;

        /*
        Formuala for padding taken from here: https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow

        For 'SAME' option output dimensions and padding options are computed as:
        out_height 	= ceil(float(in_height) / float(strides[1]))
        out_width 	= ceil(float(in_width) / float(strides[2]))
        pad_along_height 	= max((out_height - 1) * strides[1] + filter_height - in_height, 0)
        pad_along_width 	= max((out_width - 1) * strides[2] + filter_width - in_width, 0)
        pad_top 	= pad_along_height // 2
        pad_bottom 	= pad_along_height - pad_top
        pad_left 	= pad_along_width // 2
        pad_right 	= pad_along_width - pad_left
        */

        unsigned int out_height = ceil(newHeight / oldStride[0]);
        unsigned int out_width = ceil(newWidth / oldStride[1]);
        auto pad_along_height 	= std::max(int((out_height - 1) * oldStride[0] + kernelShape[mv::KERNEL_HEIGHT] - newHeight), 0);
        auto pad_along_width 	= std::max(int((out_width - 1) * oldStride[1] + kernelShape[mv::KERNEL_WIDTH] - newWidth), 0);
        unsigned int pad_top 	= pad_along_height / 2;
        unsigned int pad_bottom 	= pad_along_height - pad_top;
        unsigned int pad_left 	= pad_along_width / 2;
        unsigned int pad_right 	= pad_along_width - pad_left;

        std::array<unsigned short, 4> padding = {pad_left, pad_right, pad_top, pad_bottom};
        return padding;
    }
}

void convDilationUsingStorageElementFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto returnedParams = model.getGlobalConfigParams();
    double CMX = returnedParams->get<unsigned>("cmx");

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        if (opType == "Conv" || opType == "DepthwiseConv")
        {
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");

            if (dilationFactor > 1)
            {
                auto nextOp = findSinkLayers(dm, opIt->getOutputTensor(0))[0];
                auto nonDilatedKernel = opIt->getInputTensor(1);
                auto nonDilatedKernelShape = nonDilatedKernel->getShape();
                auto inputTensor = opIt->getInputTensor(0);
                auto outputTensor = opIt->getOutputTensor(0);
                auto outputShape = outputTensor->getShape();
                auto outputTensorMemory = outputShape.totalSize() * std::ceil(outputTensor->getDType().getSizeInBits()/8.0);
                auto name = opIt->getName();

                auto originalShape = inputTensor->getShape();
                std::vector<mv::Data::TensorIterator> subConvs;

                size_t width = originalShape[mv::IO_WIDTH_DIMENSION];
                size_t height = originalShape[mv::IO_HEIGHT_DIMENSION];
                size_t sliceWidth = std::ceil(((double)width)/dilationFactor);
                size_t sliceHeight = std::ceil(((double)height)/dilationFactor);
                //size_t sliceWidth = originalShape[mv::IO_WIDTH_DIMENSION]/dilationFactor;
                //size_t sliceHeight = originalShape[mv::IO_HEIGHT_DIMENSION]/dilationFactor;
                std::array<unsigned short, 4> padding = calcNewPadding(opIt, sliceWidth, sliceHeight);

                //Create sub dilated convs
                size_t subConvIdx = 0;
                uint64_t leadingOffset = 0;
                mv::Shape newShape({sliceWidth, sliceHeight, nonDilatedKernelShape[mv::KERNEL_INPUT_CHANNELS], 1});
                for (size_t i = 0; i < dilationFactor; i++)
                {
                    mv::Shape subConvShape = newShape;
                    if (height%dilationFactor != 0 && i >= (height%dilationFactor))
                    {
                        subConvShape[mv::IO_HEIGHT_DIMENSION] = newShape[mv::IO_HEIGHT_DIMENSION] - 1;
                    }
                    for (size_t j = 0; j < dilationFactor; j++)
                    {
                        if (width%dilationFactor != 0 && j >= (width%dilationFactor))
                        {
                            subConvShape[mv::IO_WIDTH_DIMENSION] = newShape[mv::IO_WIDTH_DIMENSION] - 1;
                        }
                        subConvs.push_back(createDilatedConvSubConv(om, opIt, inputTensor, padding,
                            name + "_DilatedSubConv" + std::to_string(i)+"_"+std::to_string(j),
                            subConvShape, subConvIdx++));
                        subConvs[subConvs.size()-1]->set<uint64_t>("leadingOffset", leadingOffset);
                        leadingOffset += subConvs[subConvs.size()-1]->getShape().totalSize();
                    }

                }

                // reconnect children to subgraph
                std::vector<mv::Data::OpListIterator> opsToLink;
                std::vector<std::size_t> inputSlots;
                for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
                {
                    opsToLink.push_back(sinkFlow.sink());
                    inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
                }

                auto dtype = opIt->get<mv::DType>("dType");
                auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
                auto opId = opIt->get<unsigned>("opId");

                om.removeOp(opIt);
                //NOTE: if the output tensor can fit on one moment on cmx, then it can be concatenated
                //on cmx by the use of the storage element of the next operation. If it can not then
                //the idea is that we will spill the sub dilation convs to ddr and we will bring them
                //back to a re-order z-major convolution which will unfold and provide the correct tensor
                mv::Data::TensorIterator concatIt;
                bool needSparse2SparseOp = false;
                bool hackCauseConcatsLocationsAreBad = false;
                std::vector<mv::Data::TensorIterator> subConvsPerColumn;
                std::vector<mv::Data::TensorIterator> firstLevelConcats;

                if (outputTensorMemory < CMX)
                {
                    for (size_t i = 0; i < dilationFactor; i++)
                    {
                        for (size_t j = 0; j < dilationFactor; j++)
                        {
                            subConvsPerColumn.push_back(subConvs[i*dilationFactor + j]);
                        }
                        concatIt = om.implicitConcat(subConvsPerColumn, "W", quantParams,
                                        name + std::to_string(i) + "DDR_WIDTH_join");
                        om.getSourceOp(concatIt)->set<unsigned>("opId", opId);
                        om.getSourceOp(concatIt)->set<bool>("dilatedWidthConcat", true);
                        om.getSourceOp(concatIt)->set<size_t>("lineofConcatHeight", i);
                        om.getSourceOp(concatIt)->set<unsigned>("dilationFactor", dilationFactor);
                        firstLevelConcats.push_back(concatIt);
                        subConvsPerColumn.clear();
                    }
                    hackCauseConcatsLocationsAreBad = true;
                    concatIt = om.implicitConcat(firstLevelConcats, "H", quantParams, name + "DDR_HEIGHT_join");
                    om.getSourceOp(concatIt)->set<unsigned>("opId", opId);
                    om.getSourceOp(concatIt)->set<bool>("joinSimulation", true);
                    om.getSourceOp(concatIt)->set<size_t>("dilationSubConvs", dilationFactor * dilationFactor);
//                    for (unsigned j = 0; j < opsToLink.size(); ++j)
//                    {
//                        opsToLink[j]->setInputTensor(concatIt, inputSlots[j], false);
//                        om.defineFlow(concatIt, opsToLink[j], inputSlots[j]);
//                    }
                }
                else
                {
                    // Specify that next layer requires sparse input
                    // At least for now until we have a way to convert a tensor with storage elements into a dense one
                    // Assuming that this will be done after SSD-512
                    if (nextOp->isSparsityConsumer())
                        nextOp->set<bool>("forcedToHaveActivationSparsityDueToDilatedConv", true);
                    else
                        needSparse2SparseOp = true;
                    concatIt = om.implicitJoin(subConvs,
                        "HW",
                        dtype,
                        quantParams,
                        name + "dilatedjoin");
                    for (unsigned j = 0; j < opsToLink.size(); ++j)
                    {
                        opsToLink[j]->setInputTensor(concatIt, inputSlots[j], false);
                        om.defineFlow(concatIt, opsToLink[j], inputSlots[j]);
                    }
                }
                om.getSourceOp(concatIt)->set<unsigned>("opId", opId);


                //NOTE: for now i will place a neutral z-major convolution just for re-order
                //but under chat with runtime we can make it work without computations, with by-passing
                if (needSparse2SparseOp || hackCauseConcatsLocationsAreBad)
                {
                    mv::Shape weightsShape({1, 1, outputShape[mv::IO_CHANNEL_DIMENSION], outputShape[mv::IO_CHANNEL_DIMENSION]});
                    std::vector<int64_t> weightsData(weightsShape.totalSize());
                    for (unsigned k = 0; k < weightsShape[mv::KERNEL_OUTPUT_CHANNELS]; ++k)
                    {
                        for (unsigned c = 0; c < weightsShape[mv::KERNEL_INPUT_CHANNELS]; ++c)
                        {
                            for (unsigned h = 0; h < weightsShape[mv::KERNEL_HEIGHT]; ++h)
                            {
                                for (unsigned w = 0; w < weightsShape[mv::KERNEL_WIDTH]; ++w)
                                {
                                    const size_t idx = (k * weightsShape[mv::KERNEL_INPUT_CHANNELS] * weightsShape[mv::KERNEL_WIDTH] * weightsShape[mv::KERNEL_HEIGHT]) +
                                                       (c * weightsShape[mv::KERNEL_WIDTH] * weightsShape[mv::KERNEL_HEIGHT]) +
                                                       (h * weightsShape[mv::KERNEL_WIDTH]) +
                                                        w;
                                    if (c == k)
                                        weightsData[idx] = 255;
                                    else
                                        weightsData[idx] = 0;
                                }
                            }
                        }
                    }

                    auto sparse2SparseWeights = om.constantInt(
                        weightsData,
                        weightsShape,
                        nonDilatedKernel->get<mv::DType>("dType"),
                        mv::Order("NHWC"),
                        {{ZP_RANGE_0_1},{SCALE_RANGE_0_1},{0.0},{1.0}},
                        inputTensor->getName() + "_DilatedSparse2SparseWeights");

                    auto sparse2SparseConv = om.conv(concatIt,
                            sparse2SparseWeights,
                            {1, 1},
                            {0, 0, 0, 0},
                            1,
                            1,
                            nonDilatedKernel->get<mv::DType>("dType"),
                            concatIt->get<mv::QuantizationParams>("quantParams"),
                            inputTensor->getName() + "_DilatedSparse2SparseConv");

                    auto sparse2SparseConvOp = om.getSourceOp(sparse2SparseConv);
                    auto sparse2SparseWeightsOp = om.getSourceOp(sparse2SparseWeights);
                    sparse2SparseConvOp->set<unsigned>("opId", om.getSourceOp(concatIt)->get<unsigned>("opId"));
                    sparse2SparseWeightsOp->set<unsigned>("opId", om.getSourceOp(concatIt)->get<unsigned>("opId"));

                    for (unsigned j = 0; j < opsToLink.size(); ++j)
                    {
                        opsToLink[j]->setInputTensor(sparse2SparseConv, inputSlots[j], false);
                        om.defineFlow(sparse2SparseConv, opsToLink[j], inputSlots[j]);
                    }
                    if (needSparse2SparseOp)
                        sparse2SparseConvOp->set<bool>("forcedToHaveActivationSparsityDueToDilatedConv", true);
                }
            }
        }
    }
}

void convDilationUsingWeightsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    mv::OpModel om(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "Conv" || opIt->getOpType() == "DepthwiseConv")
        {
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");

            if (dilationFactor > 1)
            {

                /*Get the kernel attributes*/
                auto nonDilatedKernel = opIt->getInputTensor(1);
                auto nonDilatedKernelWidth = nonDilatedKernel->getShape()[KERNEL_WIDTH];
                auto nonDilatedKernelHeight = nonDilatedKernel->getShape()[KERNEL_HEIGHT];
                auto nonDilatedKernelInputChannels = nonDilatedKernel->getShape()[KERNEL_INPUT_CHANNELS];
                auto nonDilatedKernelOutpuChannels = nonDilatedKernel->getShape()[KERNEL_OUTPUT_CHANNELS];


                /** Calculate dilated kernel shape
                  *
                  * dilatedWidth = kw + (kw - 1)(df - 1)
                  * dilatedHeight = kh + (kh - 1)(df - 1)
                  */
                mv::Shape dilatedKernelShape = mv::Shape({nonDilatedKernelWidth + (nonDilatedKernelWidth - 1) * (dilationFactor - 1),
                                                          nonDilatedKernelHeight + (nonDilatedKernelHeight - 1) * (dilationFactor - 1),
                                                          nonDilatedKernelInputChannels, nonDilatedKernelOutpuChannels});
                auto nonDilatedKernelOp = opIt.rightmostParent();
                unsigned currentOpId = nonDilatedKernelOp->get<unsigned>("opId");
                auto quantParams = nonDilatedKernelOp->get<mv::QuantizationParams>("quantParams");
                /*Populate dilated tensor with zeros*/

                /*Create Dilated Kernel Tensor*/

                //build the dilated kernel with zero points corresponding to each channel - KMB does not support different zp per channel
                std::vector<int64_t> defaultData(dilatedKernelShape.totalSize(), quantParams.getZeroPoint(0));
                mv::Tensor dilatedKernel("dilatedKernel", dilatedKernelShape, nonDilatedKernel->getDType(), mv::Order(mv::Order::getRowMajorID(dilatedKernelShape.ndims())), defaultData);

                for (unsigned oc = 0; oc < nonDilatedKernelOutpuChannels; ++oc)
                    for (unsigned ic = 0; ic < nonDilatedKernelInputChannels; ++ic)
                        for (unsigned kcolumn = 0; kcolumn < nonDilatedKernelHeight; ++kcolumn)
                            for (unsigned krow = 0; krow < nonDilatedKernelWidth; ++krow)
                                /*Copy non-dilated weights into the dilated kernel*/
                                if (krow != 0 || kcolumn != 0)
                                    dilatedKernel.at({krow + (dilationFactor - 1) * krow, kcolumn + (dilationFactor - 1) * kcolumn, ic, oc}) = nonDilatedKernel->at({krow, kcolumn, ic, oc});
                                else
                                    dilatedKernel.at({krow, kcolumn, ic, oc}) = nonDilatedKernel->at({krow, kcolumn, ic, oc});

                auto dilatedKernelOp = om.constantDataElement(
                    dilatedKernel.getData(),
                    dilatedKernelShape,
                    dilatedKernel.getDType(),
                    dilatedKernel.getOrder(),
                    quantParams,
                    nonDilatedKernelOp->getName() + "_Dilated");

                om.removeOp(nonDilatedKernelOp);
                om.defineFlow(dilatedKernelOp, opIt, 1);
                opIt->set<std::array<unsigned short, 2>>("kSize", {dilatedKernelShape[KERNEL_WIDTH], dilatedKernelShape[KERNEL_HEIGHT]} );
                opIt->setInputTensor(dilatedKernelOp, 1, false);
                opIt->set<unsigned>("dilationFactor", 1);
                auto DilatedKernelOpFetched = opIt.rightmostParent();
                DilatedKernelOpFetched->set<unsigned>("opId", currentOpId);
            }

        }

    }
}
