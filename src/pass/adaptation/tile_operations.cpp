#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <cmath>

const size_t MAX_LIMIT_KERNEL = 11;
const size_t MID_LIMIT_KERNEL_H = 5;
const size_t MID_LIMIT_KERNEL_W = 5;
const size_t NUMBER_OF_PARTITIONS = 9;
const size_t GROUP_DILATION = 1;

static void tileOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void replaceKernel(mv::Data::OpListIterator opIt, std::size_t newKernelSize, mv::OpModel om);
void partitionOperation(mv::Data::OpListIterator opIt, std::size_t oldKernelSize, std::size_t partitions, mv::ComputationModel& model, mv::Data::OpListIterator nextOpIt);
void padInputTensor(mv::Data::OpListIterator opIt, mv::ComputationModel &model);

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
            // Check if padding is too large for the partition. If so, pre pad the input and 
            // remove padding from original conv before the partition operation
            // auto originalPadding = conv->get<std::array<unsigned short, 4>>("padding");
            // if((originalPadding[0] > std::floor(MID_LIMIT_KERNEL_W/2)) or (originalPadding[2] > std::floor(MID_LIMIT_KERNEL_H/2)))
            // {
                padInputTensor(conv, model); //TODO renable handling of padding where it's small enough!
            // }
            

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

void padInputTensor(mv::Data::OpListIterator opIt, mv::ComputationModel &model)
{
    mv::OpModel om(model);

    auto inputTensor = opIt->getInputTensor(0);
    auto opId = opIt->get<unsigned>("opId");
    auto inputTensorQPs =  inputTensor->get<mv::QuantizationParams>("quantParams");
    auto inputTensorShape = inputTensor->getShape();
    auto zeroPoint = inputTensorQPs.getZeroPoint();
    auto originalPadding = opIt->get<std::array<unsigned short, 4>>("padding");
    auto otherDimSize = inputTensorShape[mv::IO_CHANNEL_DIMENSION] * inputTensorShape[mv::IO_BATCH_DIMENSION];

    // Create top/bottom padding, width of original tensor, height of t/b padding
    auto topSize = inputTensorShape[mv::IO_WIDTH_DIMENSION] * originalPadding[2];
    auto bottomSize = inputTensorShape[mv::IO_WIDTH_DIMENSION] * originalPadding[3];

    std::vector<int64_t> topData(topSize * otherDimSize, zeroPoint[0]);
    auto topPad = om.constantInt(topData, 
                                {inputTensorShape[mv::IO_WIDTH_DIMENSION], originalPadding[2], inputTensorShape[mv::IO_CHANNEL_DIMENSION], inputTensorShape[mv::IO_BATCH_DIMENSION]}, 
                                mv::DType("UInt8"), 
                                inputTensor->getOrder(), 
                                inputTensorQPs);
    om.getSourceOp(topPad)->set<unsigned>("opId", opId);
    topPad->set<bool>("is_pad", true);

    std::vector<int64_t> bottomData(bottomSize * otherDimSize, zeroPoint[0]);
    auto bottomPad = om.constantInt(bottomData, 
                            {inputTensorShape[mv::IO_WIDTH_DIMENSION], originalPadding[3], inputTensorShape[mv::IO_CHANNEL_DIMENSION], inputTensorShape[mv::IO_BATCH_DIMENSION]}, 
                            mv::DType("UInt8"), 
                            inputTensor->getOrder(), 
                            inputTensorQPs);
    om.getSourceOp(bottomPad)->set<unsigned>("opId", opId);
    bottomPad->set<bool>("is_pad", true);

    auto newHeight = inputTensorShape[mv::IO_HEIGHT_DIMENSION] + originalPadding[2] + originalPadding[3];
    // Create left/right padding, height of (original tensor+ t/b padding), width of l/r padding
    auto leftSize = newHeight * originalPadding[0];
    auto rightSize = newHeight * originalPadding[1];

    std::vector<int64_t> leftData(leftSize * otherDimSize, zeroPoint[0]);
    auto leftPad = om.constantInt(leftData, 
                                {originalPadding[0], newHeight, inputTensorShape[mv::IO_CHANNEL_DIMENSION], inputTensorShape[mv::IO_BATCH_DIMENSION]}, 
                                mv::DType("UInt8"), 
                                inputTensor->getOrder(), 
                                inputTensorQPs);
    leftPad->set<bool>("is_pad", true);

    om.getSourceOp(leftPad)->set<unsigned>("opId", opId);
    std::vector<int64_t> rightData(rightSize * otherDimSize, zeroPoint[0]);
    auto rightPad = om.constantInt(rightData, 
                            {originalPadding[1], newHeight, inputTensorShape[mv::IO_CHANNEL_DIMENSION], inputTensorShape[mv::IO_BATCH_DIMENSION]}, 
                            mv::DType("UInt8"), 
                            inputTensor->getOrder(), 
                            inputTensorQPs);
    om.getSourceOp(rightPad)->set<unsigned>("opId", opId);
    rightPad->set<bool>("is_pad", true);


    // Create concats and update flows
    auto concatH = om.concat({topPad, inputTensor, bottomPad}, "H", mv::DType("UInt8"), inputTensorQPs, opIt->getName() + "_padH");
    om.getSourceOp(concatH)->set<unsigned>("opId", opId);
    auto concatW = om.concat({leftPad, concatH, rightPad}, "W", mv::DType("UInt8"), inputTensorQPs, opIt->getName() + "_padW");
    om.getSourceOp(concatW)->set<unsigned>("opId", opId);
    
    auto sourceFlow = opIt.leftmostInput();
    om.undefineFlow(sourceFlow);
    opIt->setInputTensor(concatW, 0, false);
    om.defineFlow(concatW, opIt, 0);

    opIt->set<std::array<unsigned short, 4>>("padding", {0,0,0,0});
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
    mv::Data::TensorIterator placeAdd0, placeAdd1, placeAdd2, placeAdd3, placeAdd4, placeAdd5, placeAdd6, placeAdd7, conv, bias, newBias;
    std::vector <mv::Data::TensorIterator> convs;
    std::size_t partitionedKernelHeight = oldKernelSize;
    std::size_t partitionedKernelWidth = oldKernelSize;
    bool hasBias = opIt->hasAttr("bias");
    if (hasBias)
        bias =  dm.getTensor(opIt->get<std::string>("bias"));

    for (std::size_t branchId = 0; branchId < partitions; branchId++)
    {
        //NOTE: assuming order of paddings: left,right,top,bottom
        //Pad quadrant 0 with real data from right
        //Pad quadrant 1 with real data from bottom
        //Pad quadrant 2 with real data from top
        //Pad quadrant 3 with real data from left
        bool isasymmetric = false;
        bool propogateDType = true;
        int largeDim = 0;
        if (branchId == 0)
        {
            partitionedKernelHeight = MID_LIMIT_KERNEL_H;
            partitionedKernelWidth = MID_LIMIT_KERNEL_W;

            beginInputShape = {0,0,0,0};
            beginWeightShape = {0,0,0,0};
            propogateDType = false; // branch 1 will propogate, no need to do twice
            // padding = {initialPadding[0], 0, initialPadding[2], 0};
         }
        else if (branchId == 1)
        {
            partitionedKernelHeight = MID_LIMIT_KERNEL_H;
            partitionedKernelWidth = std::ceil((oldKernelSize - MID_LIMIT_KERNEL_W)/2);

            beginInputShape = {MID_LIMIT_KERNEL_W, 0,0,0};
            beginWeightShape = {MID_LIMIT_KERNEL_W,0,0,0};

            // padding = {0, 0, initialPadding[2], 0};
        }
        else if (branchId == 2)
        {
            partitionedKernelHeight = MID_LIMIT_KERNEL_H;
            partitionedKernelWidth = std::floor((oldKernelSize - MID_LIMIT_KERNEL_W)/2);

            beginInputShape = {MID_LIMIT_KERNEL_W + (oldKernelSize-partitionedKernelWidth-MID_LIMIT_KERNEL_W),0,0,0};
            beginWeightShape = {MID_LIMIT_KERNEL_W + (oldKernelSize-partitionedKernelWidth-MID_LIMIT_KERNEL_W),0,0,0};
            // padding = {initialPadding[0], 0, 0, initialPadding[3]};
        }
        else if (branchId == 3)
        {
            partitionedKernelHeight = std::ceil((oldKernelSize - MID_LIMIT_KERNEL_H)/2);
            partitionedKernelWidth = MID_LIMIT_KERNEL_W;

            beginInputShape = {0, MID_LIMIT_KERNEL_H,0,0};
            beginWeightShape = {0, MID_LIMIT_KERNEL_H,0,0};

            // padding = {0, initialPadding[1], 0, initialPadding[3]};
        }
        else if (branchId == 4)
        {
            partitionedKernelHeight = std::ceil((oldKernelSize - MID_LIMIT_KERNEL_H)/2);
            partitionedKernelWidth = std::ceil((oldKernelSize - MID_LIMIT_KERNEL_W)/2);

            beginInputShape = {MID_LIMIT_KERNEL_W, MID_LIMIT_KERNEL_H,0,0};
            beginWeightShape = {MID_LIMIT_KERNEL_W, MID_LIMIT_KERNEL_H,0,0};

            // padding = {0, initialPadding[1], 0, initialPadding[3]};
        }
        else if (branchId == 5)
        {
            partitionedKernelHeight = std::ceil((oldKernelSize - MID_LIMIT_KERNEL_H)/2);
            partitionedKernelWidth = std::floor((oldKernelSize - MID_LIMIT_KERNEL_W)/2);

            beginInputShape = {MID_LIMIT_KERNEL_W + (oldKernelSize-partitionedKernelWidth-MID_LIMIT_KERNEL_W),MID_LIMIT_KERNEL_H,0,0};
            beginWeightShape = {MID_LIMIT_KERNEL_W + (oldKernelSize-partitionedKernelWidth-MID_LIMIT_KERNEL_W),MID_LIMIT_KERNEL_H,0,0};

            // padding = {0, initialPadding[1], 0, initialPadding[3]};
        }
        else if (branchId == 6)
        {
            partitionedKernelHeight = std::floor((oldKernelSize - MID_LIMIT_KERNEL_H)/2);
            partitionedKernelWidth = MID_LIMIT_KERNEL_W;

            beginInputShape = {0, MID_LIMIT_KERNEL_H + (oldKernelSize-partitionedKernelHeight-MID_LIMIT_KERNEL_H),0,0};
            beginWeightShape = {0, MID_LIMIT_KERNEL_H + (oldKernelSize-partitionedKernelHeight-MID_LIMIT_KERNEL_H),0,0};

            // padding = {0, initialPadding[1], 0, initialPadding[3]};
        }
        else if (branchId == 7)
        {
            partitionedKernelHeight = std::floor((oldKernelSize - MID_LIMIT_KERNEL_H)/2);
            partitionedKernelWidth = std::ceil((oldKernelSize - MID_LIMIT_KERNEL_W)/2);

            beginInputShape = {MID_LIMIT_KERNEL_W, MID_LIMIT_KERNEL_H + (oldKernelSize-partitionedKernelHeight-MID_LIMIT_KERNEL_H),0,0};
            beginWeightShape = {MID_LIMIT_KERNEL_W, MID_LIMIT_KERNEL_H + (oldKernelSize-partitionedKernelHeight-MID_LIMIT_KERNEL_H),0,0};

            // padding = {0, initialPadding[1], 0, initialPadding[3]};
        }
        else
        {
            partitionedKernelHeight = std::floor((oldKernelSize - MID_LIMIT_KERNEL_H)/2);
            partitionedKernelWidth = std::floor((oldKernelSize - MID_LIMIT_KERNEL_W)/2);

            beginInputShape = {MID_LIMIT_KERNEL_W + (oldKernelSize-partitionedKernelWidth-MID_LIMIT_KERNEL_W), MID_LIMIT_KERNEL_H + (oldKernelSize-partitionedKernelHeight-MID_LIMIT_KERNEL_H),0,0};
            beginWeightShape = {MID_LIMIT_KERNEL_W + (oldKernelSize-partitionedKernelWidth-MID_LIMIT_KERNEL_W), MID_LIMIT_KERNEL_H + (oldKernelSize-partitionedKernelHeight-MID_LIMIT_KERNEL_H),0,0};

            // padding = {0, initialPadding[1], 0, initialPadding[3]};
        }

        if(partitionedKernelWidth != partitionedKernelHeight)
            isasymmetric = true;

        if(partitionedKernelHeight > partitionedKernelWidth)
            largeDim = 1;

        
        branchWeightSize = {partitionedKernelWidth, partitionedKernelHeight,
                            weightTensor->getShape()[mv::KERNEL_INPUT_CHANNELS],
                            weightTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]};

        branchWidth = inputTensor->getShape()[mv::IO_WIDTH_DIMENSION] - (oldKernelSize - partitionedKernelWidth);
        branchHeight = inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION] - (oldKernelSize - partitionedKernelHeight);
        branchInputSize = {branchWidth, branchHeight,
                            inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION],1};

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
                            opIt->getInputTensor(0)->get<mv::DType>("dType"),
                            opIt->get<mv::QuantizationParams>("quantParams"),
                            opIt->getName() + std::to_string(branchId));
        convs.push_back(conv);
        auto convOp = om.getSourceOp(conv);

        if(isasymmetric)
            convOp->set<unsigned>("asymmetricKernel", 1-largeDim);

        if (hasBias)
        {
            std::string biasName = mv::createBiasName(convOp->getName() + "_bias");
            newBias = dm.defineTensor(mv::Tensor(biasName, bias->getShape(),
                                         inputTensor->get<mv::DType>("dType"), bias->getOrder(), bias->getData(), bias->get<mv::QuantizationParams>("quantParams")));
            om.addAttr(convOp, "bias", biasName);
        }

        auto sliceInputOp = om.getSourceOp(sliceInput);
        auto sliceWeightOp = om.getSourceOp(sliceWeight);
        convOp->set<unsigned>("opId", initialOpId);
        convOp->set<bool>("partitionedKernelToAdd", propogateDType);
        sliceInputOp->set<unsigned>("opId", initialOpId);
        sliceWeightOp->set<unsigned>("opId", initialOpId);

    }
    placeAdd0 = om.eltwise({convs[0], convs[1]}, "Add",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "ADD_Partition0");
    placeAdd1 = om.eltwise({placeAdd0, convs[2]}, "Add",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "ADD_Partition1");
    placeAdd2 = om.eltwise({placeAdd1, convs[3]}, "Add",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "ADD_Partition2");
    placeAdd3 = om.eltwise({placeAdd2, convs[4]}, "Add",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "ADD_Partition3");
    placeAdd4 = om.eltwise({placeAdd3, convs[5]}, "Add",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "ADD_Partition4");
    placeAdd5 = om.eltwise({placeAdd4, convs[6]}, "Add",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "ADD_Partition5");
    placeAdd6 = om.eltwise({placeAdd5, convs[7]}, "Add",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "ADD_Partition6");
    placeAdd7 = om.eltwise({placeAdd6, convs[8]}, "Add",
                mv::DType("Default"), opIt->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "ADD_Partition7");
    nextOpIt->setInputTensor(placeAdd7, 0, false );
    auto placeAdd0Op = om.getSourceOp(placeAdd0);
    auto placeAdd1Op = om.getSourceOp(placeAdd1);
    auto placeAdd2Op = om.getSourceOp(placeAdd2);
    auto placeAdd3Op = om.getSourceOp(placeAdd3);
    auto placeAdd4Op = om.getSourceOp(placeAdd4);
    auto placeAdd5Op = om.getSourceOp(placeAdd5);
    auto placeAdd6Op = om.getSourceOp(placeAdd6);
    auto placeAdd7Op = om.getSourceOp(placeAdd7);
    placeAdd0Op->set<unsigned>("opId", initialOpId);
    placeAdd1Op->set<unsigned>("opId", initialOpId);
    placeAdd2Op->set<unsigned>("opId", initialOpId);
    placeAdd3Op->set<unsigned>("opId", initialOpId);
    placeAdd4Op->set<unsigned>("opId", initialOpId);
    placeAdd5Op->set<unsigned>("opId", initialOpId);
    placeAdd6Op->set<unsigned>("opId", initialOpId);
    placeAdd7Op->set<unsigned>("opId", initialOpId);
    om.defineFlow(placeAdd7, nextOpIt, 0);
    om.removeOp(opIt);
}
