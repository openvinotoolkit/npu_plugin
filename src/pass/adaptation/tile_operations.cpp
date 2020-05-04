#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <cmath>

const size_t MAX_LIMIT_KERNEL = 11;
const size_t NUMBER_OF_PARTITIONS = 4;
const size_t GROUP_DILATION = 1;

static void tileOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void replaceKernel(mv::Data::OpListIterator opIt, std::size_t newKernelSize, mv::OpModel om);
void partitionOperation(mv::Data::OpListIterator opIt, std::size_t oldKernelSize, std::size_t partitions, mv::ComputationModel& model, mv::Data::OpListIterator nextOpIt);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(TileOps)
        .setFunc(tileOpsFcn)
        .setDescription(
            "Unfortunately HW supports only operations till 11 kernel size, \
                so replace/tile bigger kernels to smaller."
        );
    }
}

void tileOpsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model,
                       mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto convOps = om.getOps("Conv");
    for (auto conv:convOps)
    {
        //NOTE: Suppose they are symmetrical for now
        auto kernel_size = conv->getInputTensor(1)->getShape()[mv::KERNEL_WIDTH];
        if (kernel_size > MAX_LIMIT_KERNEL)
        {
            auto nextOp = mv::findSinkLayers(dm, conv->getOutputTensor(0))[0];
            //NOTE: The idea here is that we need 4 equal partitions
            //NOTE: That idea is lifted later but stays here just in case
//            bool canSplitToFour = (kernel_size%NUMBER_OF_EQUAL_PARTITIONS==0);
//            if (!canSplitToFour)
//            {
//                while ((kernel_size%NUMBER_OF_EQUAL_PARTITIONS != 0))
//                    kernel_size = kernel_size + 1;
//                replaceKernel(conv, kernel_size, om);
//            }
            partitionOperation(conv, kernel_size, NUMBER_OF_PARTITIONS, model, nextOp);
        }
    }
}

void replaceKernel(mv::Data::OpListIterator opIt, std::size_t newKernelSize, mv::OpModel om)
{
    mv::Data::TensorIterator weightsTensor =  opIt->getInputTensor(1);
    auto kernelShape = opIt->getInputTensor(1)->getShape();
    std::vector<int64_t> weightsDataFull(kernelShape[mv::KERNEL_INPUT_CHANNELS] *
           kernelShape[mv::KERNEL_OUTPUT_CHANNELS] * newKernelSize * newKernelSize);
    weightsDataFull.clear();
    for (size_t k = 0; k < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; k++)
    {
        std::vector<int64_t> weightsDataChannel(kernelShape[mv::KERNEL_INPUT_CHANNELS] *
                 newKernelSize * newKernelSize, weightsTensor->get<mv::QuantizationParams>("quantParams").getZeroPoint()[k]);
        weightsDataFull.insert( weightsDataFull.begin() + k * weightsDataChannel.size(), weightsDataChannel.begin(), weightsDataChannel.end() );
    }
    for (size_t k = 0; k < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; k++)
    {
        for (size_t c = 0; c < kernelShape[mv::KERNEL_INPUT_CHANNELS]; c++)
        {
            for (size_t h = 0; h < kernelShape[mv::KERNEL_HEIGHT]; h++)
            {
                for (size_t w = 0; w < kernelShape[mv::KERNEL_WIDTH]; w++)
                {
                    auto currWeight = (int64_t)weightsTensor->at({w,h,c,k});
                    const size_t idx = (k * kernelShape[mv::KERNEL_INPUT_CHANNELS] * kernelShape[mv::KERNEL_WIDTH] * kernelShape[mv::KERNEL_HEIGHT]) +
                                       (c * kernelShape[mv::KERNEL_WIDTH] * kernelShape[mv::KERNEL_HEIGHT]) +
                                       (h * kernelShape[mv::KERNEL_WIDTH]) +
                                        w;
                    weightsDataFull[idx] = currWeight;
                }
            }
        }
    }
    auto weights = om.constantInt(weightsDataFull,
                        {newKernelSize, newKernelSize,
                        kernelShape[mv::KERNEL_INPUT_CHANNELS], kernelShape[mv::KERNEL_OUTPUT_CHANNELS]},
                        mv::DType("UInt8"),
                        weightsTensor->getOrder(),
                        weightsTensor->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "newWeights");
    om.removeOp(om.getSourceOp(opIt->getInputTensor(1)));
    opIt->setInputTensor(weights, 1, false);
    om.defineFlow(weights, opIt, 1);
    om.getSourceOp(opIt->getInputTensor(1))->set<unsigned>("opId", opIt->get<unsigned>("opId"));
}

void partitionOperation(mv::Data::OpListIterator opIt, std::size_t oldKernelSize, std::size_t partitions, mv::ComputationModel &model, mv::Data::OpListIterator nextOpIt)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto inputTensor = opIt->getInputTensor(0);
    auto weightTensor = opIt->getInputTensor(1);
    auto initialStride = opIt->get<std::array<unsigned short, 2>>("stride");
    auto initialPadding = opIt->get<std::array<unsigned short, 4>>("padding");
    std::array<unsigned short, 4> padding = initialPadding;
    unsigned initialOpId = opIt->get<unsigned>("opId");

    mv::Shape beginInputShape, branchInputSize, beginWeightShape, branchWeightSize;
    std::size_t branchWidth, branchHeight;
    mv::Data::TensorIterator placeAdd0, placeAdd1, placeAdd2, conv, bias, newBias;
    std::vector <mv::Data::TensorIterator> convs;
    std::size_t partitionedKenelSize = oldKernelSize;
    bool hasBias = opIt->hasAttr("bias");
    if (hasBias)
        bias =  dm.getTensor(opIt->get<std::string>("bias"));

    for (std::size_t branchId = 0; branchId < partitions; branchId++)
    {
        if (branchId == 0)
        {
            beginInputShape = {0,0,0,0};
            partitionedKenelSize = MAX_LIMIT_KERNEL;
            branchWidth = inputTensor->getShape()[mv::IO_WIDTH_DIMENSION] - (oldKernelSize - partitionedKenelSize);
            branchHeight = inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION] - (oldKernelSize - partitionedKenelSize);
            beginWeightShape = {0,0,0,0};
            branchWeightSize = {partitionedKenelSize, partitionedKenelSize,
                weightTensor->getShape()[mv::KERNEL_INPUT_CHANNELS],weightTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]};
            branchInputSize = {branchWidth, branchHeight,
                inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION],1};
            padding = {initialPadding[0], 0, initialPadding[2], 0};
         }
        else if (branchId == 1)
        {
            beginInputShape = {partitionedKenelSize, 0,0,0};
            branchWidth = inputTensor->getShape()[mv::IO_WIDTH_DIMENSION] - partitionedKenelSize;
            branchHeight = inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION] - (oldKernelSize - partitionedKenelSize);
            beginWeightShape = {partitionedKenelSize,0,0,0};
            branchWeightSize = {oldKernelSize - partitionedKenelSize, partitionedKenelSize,
                weightTensor->getShape()[mv::KERNEL_INPUT_CHANNELS], weightTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]};
            branchInputSize = {branchWidth, branchHeight,
                inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION],1};
            padding = {0, initialPadding[1], initialPadding[2], 0};
        }
        else if (branchId == 2)
        {
            beginInputShape = {0, partitionedKenelSize,0,0};
            branchWidth = inputTensor->getShape()[mv::IO_WIDTH_DIMENSION] - (oldKernelSize - partitionedKenelSize);
            branchHeight = inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]- partitionedKenelSize;
            beginWeightShape = {0,partitionedKenelSize,0,0};
            branchWeightSize = {partitionedKenelSize, oldKernelSize - partitionedKenelSize,
                weightTensor->getShape()[mv::KERNEL_INPUT_CHANNELS],weightTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]};
            branchInputSize = {branchWidth, branchHeight,
                inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION],1};
            padding = {initialPadding[0], 0, 0, initialPadding[3]};
        }
        else
        {
            beginInputShape = {partitionedKenelSize, partitionedKenelSize,0,0};
            branchWidth = inputTensor->getShape()[mv::IO_WIDTH_DIMENSION] - partitionedKenelSize;
            branchHeight = inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]- partitionedKenelSize;
            beginWeightShape = {partitionedKenelSize, partitionedKenelSize,0,0};
            branchWeightSize = {oldKernelSize - partitionedKenelSize, oldKernelSize - partitionedKenelSize,
                weightTensor->getShape()[mv::KERNEL_INPUT_CHANNELS],weightTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]};
            branchInputSize = {branchWidth, branchHeight,
                inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION],1};
            padding = {0, initialPadding[1], 0, initialPadding[3]};
        }

        auto sliceInput = om.slice(inputTensor,
                                   beginInputShape,
                                   branchInputSize,
                                   inputTensor->get<mv::QuantizationParams>("quantParams"),
                                   opIt->getName() + "_slice_Input" + std::to_string(branchId));

        auto sliceWeight = om.slice(weightTensor,
                                   beginWeightShape,
                                   branchWeightSize,
                                   weightTensor->get<mv::QuantizationParams>("quantParams"),
                                   opIt->getName() + "_slice_Weight" + std::to_string(branchId));

        conv = om.conv(sliceInput,
                            sliceWeight,
                            initialStride,
                            padding,
                            GROUP_DILATION,
                            GROUP_DILATION,
                            opIt->get<mv::DType>("dType"),
                            opIt->get<mv::QuantizationParams>("quantParams"),
                            opIt->getName() + std::to_string(branchId));
        convs.push_back(conv);
        auto convOp = om.getSourceOp(conv);

        if (hasBias)
        {
            std::string biasName = mv::createBiasName(convOp->getName() + "_bias");
            newBias = dm.defineTensor(mv::Tensor(biasName, bias->getShape(),
                                         opIt->get<mv::DType>("dType"), bias->getOrder(), bias->getData(), bias->get<mv::QuantizationParams>("quantParams")));
            om.addAttr(convOp, "bias", biasName);
        }

        auto sliceInputOp = om.getSourceOp(sliceInput);
        auto sliceWeightOp = om.getSourceOp(sliceWeight);
        convOp->set<unsigned>("opId", initialOpId);
        sliceInputOp->set<unsigned>("opId", initialOpId);
        sliceWeightOp->set<unsigned>("opId", initialOpId);

    }
    placeAdd0 = om.eltwise({convs[0], convs[1]}, "And",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "AND_Partition0");
    placeAdd1 = om.eltwise({placeAdd0, convs[2]}, "And",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "AND_Partition1");
    placeAdd2 = om.eltwise({placeAdd1, convs[3]}, "And",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "AND_Partition2");
    nextOpIt->setInputTensor(placeAdd2, 0, false );
    auto placeAdd0Op = om.getSourceOp(placeAdd0);
    auto placeAdd1Op = om.getSourceOp(placeAdd1);
    auto placeAdd2Op = om.getSourceOp(placeAdd2);
    placeAdd0Op->set<unsigned>("opId", initialOpId);
    placeAdd1Op->set<unsigned>("opId", initialOpId);
    placeAdd2Op->set<unsigned>("opId", initialOpId);
    om.defineFlow(placeAdd2, nextOpIt, 0);
    om.removeOp(opIt);
}
