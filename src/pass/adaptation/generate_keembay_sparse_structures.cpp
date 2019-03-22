#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include <math.h>

static void generateSparsityMapsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateSparsityMaps)
        .setFunc(generateSparsityMapsFcn)
        .setDescription(
            "Generates sparsity maps for the Tasks that need them"
        );

        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(generateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
        );
    }
}

mv::Data::TensorIterator createFakeSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& sparsityMapName, const mv::Shape& sparsityShape, const std::vector<int64_t>& sparsityMapData)
{
    auto sparsityMap = om.constantInt(sparsityMapData, sparsityShape, mv::DType("UInt8"), mv::Order("NCHW"), sparsityMapName);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    dpuTaskOp->addInputTensor(sparsityMap);
    om.defineFlow(sparsityMap, dpuTaskOp, dpuTaskOp->inputSlots());

    return sparsityMap;
}


//mv::Data::TensorIterator createSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, mv::Data::TensorIterator tensor)
//{
//    mv::ControlModel cm(om);

//    auto tensorShape = tensor->getShape();
//    std::string tensorName = tensor->getName();
//    tensorName.pop_back(); tensorName.pop_back(); //Necessary for .dot files

//    std::string sparsityMapName(tensorName+"SparsityMap");
//    auto sparsityMap = om.sparsityMap(mv::Shape({tensorShape[0], tensorShape[1], tensorShape[-1]}), mv::DType("Int32"), mv::Order(mv::Order::getRowMajorID(3)), sparsityMapName);
//    tensor->set<bool>("sparse", true);

//    if(tensor->isPopulated())
//    {
//        auto data = tensor->getData();
//        auto nonZeroElements = 0;
//        //default all zeropoints to zero
//        std::vector<unsigned> zeroPoint = tensor->getZeroPointsPerChannel();
//        std::vector<double> sparsityMapData(sparsityMap->getShape().totalSize());
//        std::vector<size_t> sub(tensorShape.ndims());
//        uint8_t map;

//        auto internalOrder = tensor->getInternalOrder();
//        auto order = tensor->getOrder();

//        for (size_t t = 0; t < tensorShape.totalSize(); t += 8)
//        {
//            map = 0;
//            for (size_t i = 0; i < 8; i++)
//            {
//                sub = order.indToSub(tensorShape, t+i);
//                if (sub[2] < tensorShape[2] && data[internalOrder.subToInd(tensorShape, sub)] != zeroPoint[sub[3]])
//                {
//                    map += 1 << i;
//                    nonZeroElements++;
//                }
//            }
//            sparsityMapData[t/8] = map;
//        }
//        sparsityMap->populate(sparsityMapData);

//        //BUG: Why does this give segfault?
//        sparsityMap = om.dMATask(sparsityMap, mv::DmaDirectionEnum::DDR2CMX);
//    }

//    dpuTaskOp->addInputTensor(sparsityMap);
//    om.defineFlow(sparsityMap, dpuTaskOp, dpuTaskOp->inputSlots());
//    std::string deallocationTaskName = sparsityMapName+"Deallocate";
//    om.deallocate(sparsityMap,deallocationTaskName);
//    cm.defineFlow(dpuTaskOp, om.getOp(deallocationTaskName));

//    return sparsityMap;
//}

mv::Data::TensorIterator addWeightsTable(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& kernelWeightsTableName, unsigned outputChannels)
{
    std::vector<int64_t> weightTableData(4 * outputChannels, 0);

    // WeightTableData should be filled here using packing information coming from Sparsity map (and quantization information maybe?)
    for(unsigned i = 0; i < outputChannels; ++i)
    {
        weightTableData[i + 0] = 0; //DATA_PTR
        weightTableData[i + 1] = 0; //SP_PTR
    }
    auto weightTable = om.constantInt(weightTableData, {outputChannels, 1, 1, 4}, mv::DType("UInt32"), mv::Order("WHCN"), kernelWeightsTableName);
    om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    dpuTaskOp->addInputTensor(weightTable);
    om.defineFlow(weightTable, dpuTaskOp, dpuTaskOp->inputSlots());

    return weightTable;
}


static void generateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            unsigned weightsTableSize;
            std::string opName = dpuTask->getName();

            weightsTableSize = dpuTask->getOutputTensor(0)->getShape()[2];

            std::string kernelWeightsTableName(opName + "WeightsTable");
            addWeightsTable(om, dpuTask, kernelWeightsTableName, weightsTableSize);
        }
    }
}

uint16_t getWindowSize(uint16_t kx, uint16_t sx)
{
    //Find max mpe where if calc window <= 32
    //return window size for the max mpe
    uint16_t windowSize, maxMpeWindowSize = 64;
    int mpe = 1;

    //mpe in [1,2,4,8,16]
    while(mpe <= 16)
    {
        if (sx <= kx)
            windowSize = kx + sx * (mpe - 1);
        else
            windowSize = kx * mpe;

        if (windowSize <= 32)
            maxMpeWindowSize = windowSize;

        mpe *= 2;
    }

    return maxMpeWindowSize;
}

std::vector<int8_t> createBitPattern(uint16_t kernelW, uint16_t kernelH, uint16_t windowsSize, uint16_t inputChannels)
{
    std::vector<int8_t> bitpattern;
    bitpattern.reserve(windowsSize*kernelH*inputChannels);
    for(size_t c = 0; c < inputChannels; c++)
        for(size_t y = 0; y < kernelH; y++)
            for(size_t x = 0; x < windowsSize; x++)
                if (x < kernelW)
                    bitpattern.emplace_back(1);
                else
                    bitpattern.emplace_back(0);
    return bitpattern;
}

// This function is necessary, as DMA task hide populated tensors (but should they do it in the first place?)

mv::Data::TensorIterator findSourceOperation(mv::OpModel om, mv::Data::TensorIterator tensor)
{
    auto op = om.getSourceOp(tensor);
    while(op->getOpType() == "DMATask")
    {
        tensor = op->getInputTensor(0);
        op = om.getSourceOp(tensor);
    }

    return tensor;
}

static void generateSparsityMapsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            std::string taskOp = dpuTask->get<std::string>("taskOp");
            pass.log(mv::Logger::MessageType::Debug, " taskOp "  + dpuTask->get<std::string>("taskOp"));
            bool isConv = taskOp == "ChannelMajorConvolution";
            bool isPooling = taskOp == "MaxPool";
            bool isDepthWiseConv = taskOp == "DepthwiseConv";

            //for max pooling and deptwise convolution and channel-major convolution we need to generate sparsity data
            //even if those layers does not support sparsity.
            if (isPooling || isDepthWiseConv || isConv)
            {
                uint16_t kernelW, kernelH;

                auto strides = dpuTask->get<std::array<unsigned short, 2>>("stride");
                auto inputChannels = dpuTask->getInputTensor(0)->getShape()[2];
                auto outputChannels = dpuTask->getOutputTensor(0)->getShape()[2];

                if (isPooling)
                {
                    auto kernelShape = dpuTask->get<std::array<unsigned short, 2>>("kSize");
                    kernelW = kernelShape[0];
                    kernelH = kernelShape[1];
                }
                else// its a depthwise conv or conv with NCHW layout
                {
                    auto weightsShape = dpuTask->getInputTensor(1)->getShape();
                    kernelW = weightsShape[0];
                    kernelH = weightsShape[1];
                }

                auto windowsSize = getWindowSize(kernelW, strides[0]);

                pass.log(mv::Logger::MessageType::Debug, "windowSize " + std::to_string(windowsSize));
                pass.log(mv::Logger::MessageType::Debug, "OutputChannels " + std::to_string(outputChannels));

                std::vector<int8_t> bitpattern;
                std::vector<uint8_t> perChannelSparsity;
                std::vector<std::size_t> ndims(4);
                if (isPooling || isDepthWiseConv)
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, 1));
                    perChannelSparsity.resize(static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)) * 16);//allocate once
                    ndims = {16, static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)), 1, inputChannels};
                }
                else
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, inputChannels));
                    auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8.0)); //how many bytes we need per window
                    auto NumberOfRowsSparistyBytes = static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0 ));
                    perChannelSparsity.resize(NumberOfRowsSparistyBytes * 16);//allocate once
                    ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
                }

                //populate sparsity
                pass.log(mv::Logger::MessageType::Debug, " perChannelSize = " + std::to_string(perChannelSparsity.size()) );
                for (size_t i = 0; i < bitpattern.size(); i++)
                {
                    //pass.log(mv::Logger::MessageType::Debug, " i = " + std::to_string(i));
                    //pass.log(mv::Logger::MessageType::Debug, " perChannelIDx = " + std::to_string(((i/128)*16 + (i%128)/8)) );
                    perChannelSparsity.at((i/128)*16 + (i%128)/8) |= bitpattern[i] << (i%8); //use at to check boundaries - just in case
                }

                //Create Tensor with Sparsity Data
                mv::Shape sparsityShape(ndims);
                std::vector<int64_t> data(sparsityShape.totalSize(), 0);
                //mv::Tensor sparsityTensor("backup", sparsityShape, mv::DType("UInt8"), mv::Order("WHCN"), data);
                auto sparsityTensor = mv::Tensor(dpuTask->getName() + "_sparse_dw", sparsityShape, mv::DType("UInt8"), mv::Order("NCHW"), data);

                for(unsigned kx = 0; kx < sparsityShape[0]; ++kx)
                    for(unsigned ky = 0; ky < sparsityShape[1]; ++ky)
                        for(unsigned ic = 0; ic < sparsityShape[2]; ++ic)
                            for(unsigned oc = 0; oc < sparsityShape[3]; ++oc)
                                sparsityTensor.at({kx, ky, ic, oc}) = static_cast<int64_t>(perChannelSparsity[ky*sparsityShape[0] + kx]);

                std::string opName = dpuTask->getName();

                std::string sparsityMapName(opName + "SparsityMap");
                createFakeSparsityMap(om, dpuTask, sparsityMapName, sparsityShape, sparsityTensor.getIntData());

                dpuTask->set<bool>("fakeSparsity", true);
            }
            else
            {
                unsigned n = dpuTask->getInputTensor().size();
                for (unsigned i = 0; i < n; ++i)
                    if (dpuTask->getInputTensor(i)->getOrder().isZMajor())
                        dpuTask->getInputTensor(i)->setSparse();

                n = dpuTask->getOutputTensor().size();
                for (unsigned i = 0; i < n; ++i)
                    if (dpuTask->getOutputTensor(i)->getOrder().isZMajor())
                        dpuTask->getOutputTensor(i)->setSparse();

            }
        }
    }
}
