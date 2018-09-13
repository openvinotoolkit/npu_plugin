#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

static void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MarkHardwareConvolution)
        .setFunc(markHardwareConvolution)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass marks the convolutions that can be executed in NCE"
        );

        MV_REGISTER_PASS(FormatMXWeights)
        .setFunc(formatMXWeights)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass reshapes relevant Convolution weights for the MyriadX NCE"
        );
    }
}

//NOTE: This should not be done in such hardcoded way.
void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{

    int amount_marked = 0;
    int mark_limit = 3;

    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if(!opIterator->isHardwarizeable(pobj) || amount_marked >= mark_limit)
        {
            om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
            continue;
        }

        int mode = 0; // Assuming mode 0
        int noOfBlocks = 1 << mode;
        int inputChannels = opIterator->getInputTensor(0)->getShape()[2];
        int inputWidth = opIterator->getInputTensor(0)->getShape()[0];
        int outputChannels = opIterator->getOutputTensor(0)->getShape()[2];
        auto kernelsize = opIterator->getInputTensor(1)->getShape();

        // // if coEffTotalSize > 128000
        // if( inputChannels >= 256 || outputChannels >= 256)
        // {
        //     // printf("Incompatible because coefficients exceed on-chip channel amounts. TODO: Split over Channels\n");
        //     om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
        //     continue;
        // }

        if(inputChannels % 8)
        {
            // printf("Incompatible because input channels %% 8 != 0\n");
            om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
            continue;
        }
        if(outputChannels % noOfBlocks)
        {
            // printf("Incompatible because output channels %% NoOfBlocks != 0\n");
            om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
            continue;
        }

        if(kernelsize[0] != 1 || kernelsize[1] != 1)
        {
            printf("Incompatible because not a 1x1 conv\n");
            om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
            continue;
        }
        if(inputWidth < 32)
        {
            // printf("Incompatible because input Width < 32 != 0\n");
            om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
            continue;
        }
        om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 1));
        om.addAttr(opIterator, "NCE1_Mode", mv::Attribute(mv::AttrType::IntegerType, mode));
        om.addAttr(opIterator, "NCE1_AssignedCMX", mv::Attribute(mv::AttrType::IntegerType, 0));

        int bytesPerInputPixel = sizeof(double)/2; // Assuming FP16 for inputs
        int bytesPerCoefficient = sizeof(double)/2; // Assuming FP16 for coeffiecients
        int kerDimX = opIterator->getInputTensor(1)->getShape()[0];
        int kerDimY = opIterator->getInputTensor(1)->getShape()[1];
        int coefficientsForInputChannel = kerDimX * kerDimY;
        int coefficientsForInputChannelRequiredMemory = coefficientsForInputChannel * bytesPerCoefficient;
        int inputDimX = opIterator->getInputTensor(0)->getShape()[0];
        int inputChannelsPerRamBlock = inputChannels / noOfBlocks;
        int memoryDPE = 512; //512 bytes for each DPE
        int coefficientsAvailableMemory = memoryDPE * noOfBlocks;
        int splitsOverInputChannels = coefficientsForInputChannelRequiredMemory / coefficientsAvailableMemory + 1;

        int splitsOverHeight = 1;
        unsigned int total_input_size = opIterator->getInputTensor(0)->getShape().totalSize() * 2;
        unsigned int total_output_size = opIterator->getOutputTensor(0)->getShape().totalSize() * 2;

        double CMX_STREAM_SIZE = 256*1024;

        if (total_input_size + total_output_size > CMX_STREAM_SIZE){
            // TODO: Take into consideration previous splits.
            splitsOverHeight = (unsigned int) ceil((total_input_size + total_output_size)/CMX_STREAM_SIZE);
        }

        double doubleOutputChannels = (double) outputChannels;

        int descriptorsSplits = splitsOverHeight * splitsOverInputChannels * ceil(doubleOutputChannels / 256); //<- If mode 0 is used for every subtensor.
        //Assuming no split over H (if possible)
        om.addAttr(opIterator, "NCE1_DescriptorSplits", mv::Attribute(mv::AttrType::IntegerType, descriptorsSplits));
        om.addAttr(opIterator, "NCE1_InputChannelsPerRamBlock", mv::Attribute(mv::AttrType::IntegerType, inputChannelsPerRamBlock));

        om.addAttr(opIterator, "NCE1_TopOutputJunk", mv::Attribute(mv::AttrType::IntegerType, 0));
        om.addAttr(opIterator, "NCE1_BottomOutputJunk", mv::Attribute(mv::AttrType::IntegerType, 0));

        int pixelsPerLine = 128 / (bytesPerInputPixel * 8); // RAMs are arranged in lines of 128bits
        int bytesPerLine = pixelsPerLine * bytesPerInputPixel;
        int localLineStride = (inputDimX + (pixelsPerLine - 1)) / pixelsPerLine;

        om.addAttr(opIterator, "NCE1_LocalLineStride", mv::Attribute(mv::AttrType::IntegerType, localLineStride));

        int sizeOfBlock = (128 * 1024) >> mode; //128KB of total memory
        int chanPerBlock = inputChannels / noOfBlocks;
        int availableBytesPerChan = sizeOfBlock / chanPerBlock;
        int linesPerChan = availableBytesPerChan / bytesPerLine;
        om.addAttr(opIterator, "NCE1_LinesPerChannel", mv::Attribute(mv::AttrType::IntegerType, linesPerChan));

        int localChanStride = linesPerChan * localLineStride;
        om.addAttr(opIterator, "NCE1_LocalChannelStride", mv::Attribute(mv::AttrType::IntegerType, localChanStride));

        int minLines = 0;
        bool poolEn = false;
        if(poolEn)
            minLines = 0; //TODO
        else
            minLines = std::min(kerDimY+1, linesPerChan);
        om.addAttr(opIterator, "NCE1_MinLines", mv::Attribute(mv::AttrType::IntegerType, minLines));

        int streamingMask = 0; //For DDR streaming
        om.addAttr(opIterator, "NCE1_StreamingMask", mv::Attribute(mv::AttrType::IntegerType, streamingMask));
        amount_marked++;
    }
}


//NOTE: This should not be done in such hardcoded way.
void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    mv::OpModel om(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        bool valid = false;
        if(opIterator->hasAttr("NCE1_Compatible"))
        {
            valid = opIterator->getAttr("NCE1_Compatible").getContent<int>();
        }
        if (valid){

            auto weights = opIterator->getInputTensor(1);
            auto wshape = weights->getShape();

            std::cout << mv::Printable::toString(wshape) << std::endl;
            std::cout << wshape[3]/8 << ",64,1,1,8"<<std::endl;

            mv::Shape newShape(
                // (mv::std::size_t)wshape[0],
                // (mv::std::size_t)wshape[1],
                // (mv::std::size_t)(wshape[2] * wshape[3]/8),
                // (mv::std::size_t)8
                {32,
                64,
                1,
                1,
                8}
            );

            mv::Tensor newTensor = mv::Tensor("MX_Weights",
                                                newShape,
                                                weights->getDType(),
                                                weights->getOrder());

            std::vector<double> new_data;
            auto data = weights->getData();

            unsigned int o_iC = wshape[2], o_oC = wshape[3], o_fw = wshape[1];

            for(std::size_t i = 0; i != newShape[0]; i++){
                for(std::size_t j = 0; j != newShape[1]; j++){
                    for(std::size_t x = 0; x != newShape[2]; x++){
                        for(std::size_t y = 0; y != newShape[3]; y++){
                            for(std::size_t z = 0; z != newShape[4]; z++){
                                new_data.push_back(data[
                                    x*o_fw*o_iC*o_oC +  // Kernel Height is largest Dim in original matrix.
                                    y*o_iC*o_oC +       // Followed by Width
                                    j*o_oC +            // then Input Channels
                                    i*8 + z             // Output Channels are written in blocks of 8
                                ]);
                            }
                        }
                    }
                }
            }

            newTensor.populate(new_data, weights->getOrder());

            auto new_op = om.constant(
                newTensor.getData(),
                newTensor.getShape(),
                newTensor.getDType(),
                newTensor.getOrder(),
                mv::Printable::toString(mv::OpType::Constant) + "_" + mv::Printable::toString(om.opsCount(mv::OpType::Constant)) + "MxWeights"
            );

            opIterator->setInputTensor(new_op, 1);

        }
    }
}
