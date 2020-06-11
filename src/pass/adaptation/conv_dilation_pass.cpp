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

mv::Data::TensorIterator createDilatedConvSubConv(mv::OpModel om, mv::Data::OpListIterator opIt, mv::Data::TensorIterator sourceTensor, std::string name)                                               
{
    mv::Data::TensorIterator subConv;
    std::array<unsigned short, 4> padding = {1, 1, 1, 1}; // Alessandro says this is correct for SSD layer needs to be checked

    subConv = om.conv(sourceTensor,
                opIt->getInputTensor(1),
                opIt->get<std::array<unsigned short, 2>>("stride"),
                padding,
                1, /// no dilation now
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
                auto sourceTensor = opIt->getInputTensor(0);
                auto parentOpIt = om.getSourceOp(sourceTensor);
                auto name = opIt->getName();
                auto inputTensor = opIt->getInputTensor(0);
                auto originalShape = inputTensor->getShape();
                auto numberOfSubConvs = dilationFactor * dilationFactor;
                std::vector<mv::Data::TensorIterator> subConvs;

                size_t sliceWidth = originalShape[mv::IO_WIDTH_DIMENSION]/dilationFactor;
                size_t sliceHeight = originalShape[mv::IO_HEIGHT_DIMENSION]/dilationFactor;

                //Create sub dilated convs
                for (size_t i = 0; i < numberOfSubConvs; i++) 
                    subConvs.push_back(createDilatedConvSubConv(om, opIt, sourceTensor, name + "_DilatedSubConv" + std::to_string(i)));
                
                linkNewMultipleOperationsReplacement(parentOpIt, subConvs, om, opIt); //applied a tempory hack in this function
                

                // Create Sparsity Map all ones
                // You can't add sparsity maps in this pass, has to be added after convert to DPU task
                            
                // for (size_t i = 0; i < numberOfSubConvs; i++) 
                // {
                //     auto mapShape = mv::Shape({{inputTensor->getShape()[mv::IO_WIDTH_DIMENSION]},
                //                                 {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
                //                                 {inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION]/8},
                //                                 {1}});

                //     //TODO re-use sparsity map if possible?
                //     std::vector<int64_t> fakeSparsityMapData(mapShape.totalSize(), 255);
                //     mv::QuantizationParams quantParams = {{},{},{},{}};
                //     std::string unpopulatedSparsityMapName = subConvs[i]->getName() + "_activation_map";
                //     auto unpopulatedSparsityMap = om.constantInt(fakeSparsityMapData, mapShape, mv::DType("UInt8"), mv::Order("NHWC"), quantParams, unpopulatedSparsityMapName);
                

                //     auto subConvOp = om.getSourceOp(subConvs[i]);

                //     unsigned opId;
                //     if(subConvOp->hasAttr("opId"))
                //         opId = subConvOp->get<unsigned>("opId");
  
                //     om.getSourceOp(unpopulatedSparsityMap)->set<unsigned>("opId", opId);
                
                //     unsigned newInputsSize = subConvOp->addInputTensor(unpopulatedSparsityMap);
                //     om.defineFlow(unpopulatedSparsityMap, subConvOp, newInputsSize - 1);
                //     subConvs[i]->set<size_t>("unpopulatedSparsityMapIndex", newInputsSize - 1);
                // }

                 //Create Storage Elements for each sub conv, we can fill storage elements here since its offset and not address
                //The input itself is not split into smaller tensors, we load the whole input once and then the storage elements are used
                // for slicing the input
                // if streaming is required (full input + storage elements + sparsity map + output of sub conv dont fit in CMX then input might be streamed)
                // The output will run in *dense* mode for the conv, but storage elements will be provided for the next layer by compiler, based on addresses
                /// of the outputs of all subconvs (they dont have to be contagious the storage elements will do the concat). Here it's not offsets but rather
                // full addresses?? SparseToDense?

                // mv::Shape storageElementShape = mv::Shape({{inputTensor->getShape()[mv::IO_WIDTH_DIMENSION]},
                //                                            {inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION]},
                //                                            {1},
                //                                            {1}});
                // std::vector<int64_t> storageElementData(storageElementShape.totalSize(), 0);

                // //Fill Storage Element
                // std::string storageElementName = subConvs[i]->getName() + "storage_element_map";
                // auto storageElement = om.constantInt(storageElementData, storageElementShape, mv::DType("Int32"), mv::Order("NHWC"), quantParams, storageElementName);
                // om.getSourceOp(storageElement)->set<unsigned>("opId", opId);
                // newInputsSize = subConvs[i]->addInputTensor(storageElement);
                // om.defineFlow(storageElement, subConvs[i], newInputsSize - 1);
                // subConvs[i]->set<size_t>("storageElementIndex", newInputsSize - 1);

                    // populateSubconvActivationStorageElementMap(storageElement, storageElement, i, dilationFactor, originalShape[mv::IO_WIDTH_DIMENSION]);

                //}

                //TODO add Concat with dilated flag (or create a new DilatedConcat layer, this is not the same as concat, because there is no DMAs to do the concat)
                // The concat will be done using storage elements.
            }
        }
    }
}
