#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/data_generator.hpp"

static void convDilationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvolutionDilation)
            .setFunc(convDilationFcn)
            .setDescription(
                "This pass dilates a kernel");
    }
}

void populateSubconvActivationStorageElementMap(mv::Data::TensorIterator activationStorageElement, mv::Data::TensorIterator input, unsigned int subConvIndex,
    unsigned int dilationFactor, unsigned int originalWidth)
{
    auto inputChannels = input->getShape()[mv::IO_CHANNEL_DIMENSION];
    auto width = activationStorageElement->getShape()[mv::IO_WIDTH_DIMENSION];
    auto height = activationStorageElement->getShape()[mv::IO_HEIGHT_DIMENSION];

    std::vector<int64_t> unpopulated_offsets(width*height, 0);

    long int increment = inputChannels * (input->getDType().getSizeInBits() / 8) ;
    long int subConvOffset = increment * subConvIndex;
    long int subConvElementIncrement = increment * dilationFactor;
    long int subConvRowIncrement = increment * originalWidth * dilationFactor;

    unsigned i = 0;
    unsigned rowOffset = subConvOffset;
    for(unsigned h = 0; h < height; ++h)
    {
        for(unsigned w = 0; w < width; ++w)
        {
            unpopulated_offsets[i++] = ((rowOffset + w * subConvElementIncrement )<< SHIFT_FOR_STORAGE_ELEMENT);
        }
        rowOffset += subConvRowIncrement;
    }
    activationStorageElement->populate(unpopulated_offsets, mv::Order("NHWC"));
}

// void convDilationFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
// {

//     MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
//     using namespace mv;

//     mv::OpModel om(model);
//     mv::DataModel dm(model);

//     for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
//     {
//         auto opType = opIt->getOpType();
//         if (opType == "Conv" || opType == "DepthwiseConv")
//         {
//             auto dilationFactor = opIt->get<unsigned>("dilationFactor");

//             if (dilationFactor > 1)
//             {
//                 //Create sub convolutions dF*dF total convs
//                 //TBD calculate padding for new sub convs (padding == DF will be padding = 1 in new subconvs)


//                 //Create Sparsity Map all ones
//                 //Create Storage Elements for each sub conv, we can fill storage elements here since its offset and not address
//                 //The input itself is not split into smaller tensors, we load the whole input once and then the storage elements are used
//                 // for slicing the input
//                 // if streaming is required (full input + storage elements + sparsity map + output of sub conv dont fit in CMX then input might be streamed)
//                 // The output will run in *dense* mode for the conv, but storage elements will be provided for the next layer by compiler, based on addresses
//                 /// of the outputs of all subconvs (they dont have to be contagious the storage elements will do the concat). Here it's not offsets but rather
//                 // full addresses?? SparseToDense?
//                 auto nonDilatedKernel = opIt->getInputTensor(1);
//                 auto nonDilatedKernelShape = nonDilatedKernel->getShape();

//                 auto inputTensor = opIt->getInputTensor(0);
//                 auto originalShape = inputTensor->getShape();
//                 auto numberOfSubConvs = dilationFactor * dilationFactor;
//                 std::vector<mv::Data::OpListIterator> subConvs(numberOfSubConvs);

//                 size_t sliceWidth = originalShape[mv::IO_WIDTH_DIMENSION]/dilationFactor;
//                 size_t sliceHeight = originalShape[mv::IO_HEIGHT_DIMENSION]/dilationFactor;

//                 auto opId = opIt->get<unsigned>("opId");

//                 for (size_t i = 0; i < numberOfSubConvs; i++) {

//                     //TODO Add handling everywhere for dilated Slice: same as slice but both input and output of the dilationSlice are in CMX, so the DMA will be before the dilation
//                     // slice and not after it.(or do we need a separate DilatedSlice Op)

//                     //TODO handle last slice in case of originalShape[mv::IO_WIDTH_DIMENSION]%dilationFactor !=0
//                     mv::Data::TensorIterator sliceInput = om.slice(inputTensor,
//                                 {0, 0, 0, 0}, //childTiles[split].getStartCoord()
//                                 {sliceWidth, sliceHeight, originalShape[mv::IO_CHANNEL_DIMENSION], 1}, //childTiles[split].getSize()
//                                 inputTensor->get<mv::QuantizationParams>("quantParams"),
//                                 opIt->getName() + "_dilatedSlice_" + std::to_string(i));
//                     auto sliceInputOp = om.getSourceOp(sliceInput);
//                     sliceInputOp->set<unsigned>("opId", opId);
//                     sliceInputOp->set<bool>("dilatedSlice", true);

//                     mv::Data::TensorIterator newTensor;
//                     std::array<unsigned short, 4> padding = {1, 1, 1, 1};//TODO
//                     if (opType == "DepthwiseConv")
//                         newTensor = om.depthwiseConv(sliceInput,
//                                         nonDilatedKernel,
//                                         opIt->get<std::array<unsigned short, 2>>("stride"),
//                                         padding,
//                                         1, /// no dilation now
//                                         opIt->get<mv::DType>("dType"),
//                                         opIt->get<mv::QuantizationParams>("quantParams"),
//                                         opIt->getName() + "_subConv_" + std::to_string(i));

//                     if (opType == "Conv")
//                         newTensor = om.conv(sliceInput,
//                                         nonDilatedKernel,
//                                         opIt->get<std::array<unsigned short, 2>>("stride"),
//                                         padding,
//                                         1, /// no dilation now
//                                         opIt->get<unsigned>("group"),
//                                         opIt->get<mv::DType>("dType"),
//                                         opIt->get<mv::QuantizationParams>("quantParams"),
//                                         opIt->getName() + "_subConv_" + std::to_string(i));
//                     subConvs[i] = om.getSourceOp(newTensor);

//                     auto mapShape = mv::Shape({{sliceInput->getShape()[mv::IO_WIDTH_DIMENSION]},
//                                                {sliceInput->getShape()[mv::IO_HEIGHT_DIMENSION]},
//                                                {sliceInput->getShape()[mv::IO_CHANNEL_DIMENSION]/8},
//                                                {1}});

//                     //TODO re-use sparsity map if possible?
//                     std::vector<int64_t> fakeSparsityMapData(mapShape.totalSize(), 255);
//                     mv::QuantizationParams quantParams = {{},{},{},{}};
//                     std::string unpopulatedSparsityMapName = subConvs[i]->getName() + "_activation_map";
//                     auto unpopulatedSparsityMap = om.constantInt(fakeSparsityMapData, mapShape, mv::DType("UInt8"), mv::Order("NHWC"), quantParams, unpopulatedSparsityMapName);
//                     om.getSourceOp(unpopulatedSparsityMap)->set<unsigned>("opId", opId);
//                     unsigned newInputsSize = subConvs[i]->addInputTensor(unpopulatedSparsityMap);
//                     om.defineFlow(unpopulatedSparsityMap, subConvs[i], newInputsSize - 1);
//                     subConvs[i]->set<size_t>("unpopulatedSparsityMapIndex", newInputsSize - 1);

//                     mv::Shape storageElementShape = mv::Shape({{inputTensor->getShape()[mv::IO_WIDTH_DIMENSION]},
//                                                                {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
//                                                                {1},
//                                                                {1}});
//                     std::vector<int64_t> storageElementData(storageElementShape.totalSize(), 0);

//                     //Fill Storage Element
//                     std::string storageElementName = subConvs[i]->getName() + "storage_element_map";
//                     auto storageElement = om.constantInt(storageElementData, storageElementShape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, storageElementName);
//                     om.getSourceOp(storageElement)->set<unsigned>("opId", opId);
//                     newInputsSize = subConvs[i]->addInputTensor(storageElement);
//                     om.defineFlow(storageElement, subConvs[i], newInputsSize - 1);
//                     subConvs[i]->set<size_t>("storageElementIndex", newInputsSize - 1);

//                     populateSubconvActivationStorageElementMap(storageElement, storageElement, i, dilationFactor, originalShape[mv::IO_WIDTH_DIMENSION]);

//                 }

//                 //TODO add Concat with dilated flag (or create a new DilatedConcat layer, this is not the same as concat, because there is no DMAs to do the concat)
//                 // The concat will be done using storage elements.
//             }
//         }
//     }
// }

mv::Data::TensorIterator createDilatedConvSubConv(mv::OpModel om, mv::Data::OpListIterator opIt, mv::Data::TensorIterator sourceTensor,
                                                    std::array<unsigned short, 4> padding, std::string name, mv::Shape newShape)
{
    mv::Data::TensorIterator subConv;
    //TODO handle stride != 1
    subConv = om.dilatedSubConv(sourceTensor,
                opIt->getInputTensor(1),
                opIt->get<std::array<unsigned short, 2>>("stride"),
                padding,
                newShape,
                1,
                opIt->getOpType(), //TODO currently we assume conv need to handle DW
                opIt->get<unsigned>("group"),
                opIt->get<mv::DType>("dType"),
                opIt->get<mv::QuantizationParams>("quantParams"),
                name);

    auto subConvOp = om.getSourceOp(subConv);
    if(opIt->hasAttr("opId"))
    {
        unsigned currentOpId = opIt->get<unsigned>("opId");
        subConvOp->set<unsigned>("opId", currentOpId);
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
        unsigned int p1 = ((newWidth - 1) * oldStride[0] - kernelShape[mv::KERNEL_WIDTH] + newWidth)/2;
        unsigned int p2 = ((newHeight - 1) * oldStride[1] - kernelShape[mv::KERNEL_HEIGHT] + newHeight)/2;
        std::array<unsigned short, 4> padding = {p1, p2, p1, p2};
        return padding;
    }
}
void convDilationFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
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
                auto nonDilatedKernel = opIt->getInputTensor(1);
                auto nonDilatedKernelShape = nonDilatedKernel->getShape();
                auto inputTensor = opIt->getInputTensor(0);
                auto parentOpIt = om.getSourceOp(inputTensor);
                auto name = opIt->getName();

                auto originalShape = inputTensor->getShape();
                auto numberOfSubConvs = dilationFactor * dilationFactor;
                std::vector<std::vector<mv::Data::TensorIterator>> subConvs;

                size_t sliceWidth = originalShape[mv::IO_WIDTH_DIMENSION]/dilationFactor;
                size_t sliceHeight = originalShape[mv::IO_HEIGHT_DIMENSION]/dilationFactor;
                std::array<unsigned short, 4> padding = calcNewPadding(opIt, sliceWidth, sliceHeight);

                //Create sub dilated convs
                for (size_t i = 0; i < dilationFactor; i++)
                {
                    std::vector<mv::Data::TensorIterator> currVec;
                    for (size_t j = 0; j < dilationFactor; j++)
                        currVec.push_back(createDilatedConvSubConv(om, opIt, inputTensor, padding, name + "_DilatedSubConv" + std::to_string(i)+"_"+std::to_string(j), {sliceWidth, sliceHeight, nonDilatedKernelShape[mv::KERNEL_OUTPUT_CHANNELS], 1}));
                    subConvs.push_back(currVec);
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

                // emulate multi dim concat by using three concats
                std::vector<mv::Data::TensorIterator> subConcat;
                for (size_t i = 0; i < dilationFactor; i++)
                {
                    subConcat.push_back(om.concat(subConvs[i],
                        "H",
                        dtype,
                        quantParams,
                        name + "dialtedconcat_H"+ std::to_string(i)));
                    om.getSourceOp(subConcat[i])->set<unsigned>("opId", opId);
                }
                auto concat = om.concat(subConcat,
                        "W",
                        dtype,
                        quantParams,
                        name + "dialtedconcat_W");
                om.getSourceOp(concat)->set<unsigned>("opId", opId);
                for (unsigned j = 0; j < opsToLink.size(); ++j)
                {
                    opsToLink[j]->setInputTensor(concat, inputSlots[j], false);
                    om.defineFlow(concat, opsToLink[j], inputSlots[j]);
                }

            }
        }
    }
}
