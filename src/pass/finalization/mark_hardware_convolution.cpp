#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

static void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

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

    }

}

//NOTE: This should not be done in such hardcoded way.
void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{

    mv::OpModel om(model);
    bool markedOneConvolution = false;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if(opIterator->getOpType() == mv::OpType::Conv2D)
        {
            if(!opIterator->isHardwarizeable(pobj) || markedOneConvolution)
            {
                om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
                continue;
            }

            int mode = 0; // Assuming mode 0
            int noOfBlocks = 1 << mode;
            int inputChannels = opIterator->getInputTensor(0)->getShape()[2];
            int outputChannels = opIterator->getOutputTensor(0)->getShape()[2];
            if(inputChannels % 8)
            {
                om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
                continue;
            }
            if(outputChannels % noOfBlocks)
            {
                om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 0));
                continue;
            }
            om.addAttr(opIterator, "NCE1_Compatible", mv::Attribute(mv::AttrType::IntegerType, 1));
            om.addAttr(opIterator, "NCE1_Mode", mv::Attribute(mv::AttrType::IntegerType, mode));
            om.addAttr(opIterator, "NCE1_AssignedCMX", mv::Attribute(mv::AttrType::IntegerType, 0));

            int bytesPerInputPixel = sizeof(float)/2; // Assuming FP16 for inputs
            int bytesPerCoefficient = sizeof(float)/2; // Assuming FP16 for coeffiecients
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
            float floatOutputChannels = (float) outputChannels;

            int descriptorsSplits = splitsOverHeight * splitsOverInputChannels * ceil(floatOutputChannels / 256); //<- If mode 0 is used for every subtensor.
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

            int streamingMask = 1; //For DDR streaming
            om.addAttr(opIterator, "NCE1_StreamingMask", mv::Attribute(mv::AttrType::IntegerType, streamingMask));
            markedOneConvolution = true;
            std::cout << "Marked one convolution as executable in HW" << std::endl;
        }
    }

}
