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

    mv::OpModel om(model);
    bool markedOneConvolution = false;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
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
        unsigned int total_input_size = opIterator->getInputTensor(0)->getShape().totalSize() * 2;
        unsigned int total_output_size = opIterator->getOutputTensor(0)->getShape().totalSize() * 2;

        float CMX_STREAM_SIZE = 256*1024;

        printf("%u > %f\n",total_input_size + total_output_size , CMX_STREAM_SIZE );


        if (total_input_size + total_output_size > CMX_STREAM_SIZE){
            // TODO: Take into consideration previous splits.
            printf("Do Split over H");

            splitsOverHeight = (unsigned int) ceil((total_input_size + total_output_size)/CMX_STREAM_SIZE);
        }


        printf("Split over H: %u\n", splitsOverHeight);

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

        int streamingMask = 0; //For DDR streaming
        om.addAttr(opIterator, "NCE1_StreamingMask", mv::Attribute(mv::AttrType::IntegerType, streamingMask));
        markedOneConvolution = true;
        std::cout << "Marked one convolution as executable in HW" << std::endl;
    }
}


//NOTE: This should not be done in such hardcoded way.
void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    mv::OpModel om(model);
    bool markedOneConvolution = false;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        bool valid = false;
        if(opIterator->hasAttr("NCE1_Compatible"))
        {
            valid = opIterator->getAttr("NCE1_Compatible").getContent<int>();
        }
        if (valid){

            std::cout << "Taps Reshape" << std::endl;

            auto weights = opIterator->getInputTensor(1);
            auto wshape = weights->getShape();

            mv::Shape newShape = mv::Shape(
                // (mv::dim_type)wshape[0],
                // (mv::dim_type)wshape[1],
                // (mv::dim_type)(wshape[2] * wshape[3]/8),
                // (mv::dim_type)8
                32,
                64,
                1,
                1,
                8
            );

            std::cout << "Before" << mv::Printable::toString(wshape) << std::endl;
            std::cout << "After" << mv::Printable::toString(newShape) << std::endl;

            mv::Tensor newTensor = mv::Tensor("MX_Weights",
                                                newShape,
                                                weights->getDType(),
                                                weights->getOrder());

            mv::dynamic_vector<mv::float_type> new_data;
            auto data = weights->getData();

            unsigned int o_iC = wshape[2], o_oC = wshape[3], o_fh = wshape[0], o_fw = wshape[1];

            for(int i = 0; i != newShape[0]; i++){
                for(int j = 0; j != newShape[1]; j++){
                    for(int x = 0; x != newShape[2]; x++){
                        for(int y = 0; y != newShape[3]; y++){
                            for(int z = 0; z != newShape[4]; z++){
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

            newTensor.populate(new_data);

            std::cout << new_data[0] << std::endl;
            std::cout << new_data[1] << std::endl;
            std::cout << new_data[2] << std::endl;
            std::cout << new_data[3] << std::endl;
            std::cout << new_data[4] << std::endl;
            std::cout << new_data[5] << std::endl;

            auto new_op = om.constant(
                newTensor.getData(),
                newTensor.getShape(),
                newTensor.getDType(),
                newTensor.getOrder(),
                "MxWeights"
            );

            opIterator->setInputTensor(new_op, 1);

        }
    }
}
