#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/tiling.hpp"
#include <algorithm>

static void convDilationNewFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void convDilationOldFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvolutionDilation_New)
            .setFunc(convDilationNewFcn)
            .setDescription(
                "This pass dilates a kernel using new method with SEPS");
    }

    namespace pass
    {
        MV_REGISTER_PASS(ConvolutionDilation_Old)
            .setFunc(convDilationOldFcn)
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

    //TODO handle last slice in case of originalShape[mv::IO_WIDTH_DIMENSION]%dilationFactor !=0
    mv::Data::TensorIterator sliceInput = om.slice(sourceTensor,
                                 {0, 0, 0, 0},
                                 newShape,
                                 sourceTensor->get<mv::QuantizationParams>("quantParams"),
                                 opIt->getName() + "_dilatedSlice_" + std::to_string(subConvIdx));

    //sliceInput->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
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
void convDilationNewFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    mv::OpModel om(model);
    mv::DataModel dm(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        if (opType == "Conv" || opType == "DepthwiseConv")
        {
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");

            if (dilationFactor > 1)
            {
                // Specify that next layer requires sparse input
                // At least for now until we have a way to convert a tensor with storage elements into a dense one
                // Assuming that this will be done after SSD-512 
                opIt.leftmostChild()->set<bool>("forcedToHaveActivationSparsityDueToDilatedConv", true);

                auto nonDilatedKernel = opIt->getInputTensor(1);
                auto nonDilatedKernelShape = nonDilatedKernel->getShape();
                auto inputTensor = opIt->getInputTensor(0);
                auto parentOpIt = om.getSourceOp(inputTensor);
                auto name = opIt->getName();

                auto originalShape = inputTensor->getShape();
                auto numberOfSubConvs = dilationFactor * dilationFactor;
                std::vector<mv::Data::TensorIterator> subConvs;

                size_t sliceWidth = originalShape[mv::IO_WIDTH_DIMENSION]/dilationFactor;
                size_t sliceHeight = originalShape[mv::IO_HEIGHT_DIMENSION]/dilationFactor;
                //TODO handle last slice in case of originalShape[mv::IO_WIDTH_DIMENSION]%dilationFactor !=0
                std::array<unsigned short, 4> padding = calcNewPadding(opIt, sliceWidth, sliceHeight);

                //Create sub dilated convs
                size_t subConvIdx = 0;
                uint64_t leadingOffset = 0;
                for (size_t i = 0; i < dilationFactor; i++)
                {
                    mv::Shape newShape({sliceWidth, sliceHeight, nonDilatedKernelShape[mv::KERNEL_OUTPUT_CHANNELS], 1});
                    for (size_t j = 0; j < dilationFactor; j++)
                    {
                        subConvs.push_back(createDilatedConvSubConv(om, opIt, inputTensor, padding,
                            name + "_DilatedSubConv" + std::to_string(i)+"_"+std::to_string(j),
                            newShape, subConvIdx++));
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

                auto join = om.implicitJoin(subConvs,
                        "HW",
                        dtype,
                        quantParams,
                        name + "dialtedjoin");
                om.getSourceOp(join)->set<unsigned>("opId", opId);
                for (unsigned j = 0; j < opsToLink.size(); ++j)
                {
                    opsToLink[j]->setInputTensor(join, inputSlots[j], false);
                    om.defineFlow(join, opsToLink[j], inputSlots[j]);
                }
                //TODO add StorageElement & sparsity MAp for following Op
            }
        }
    }
}

void convDilationOldFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    mv::OpModel om(model);
    mv::DataModel dm(model);

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
                auto nonDilatedKernelShape = nonDilatedKernel->getShape();


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
