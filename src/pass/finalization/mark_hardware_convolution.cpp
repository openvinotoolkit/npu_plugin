#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

static void markHardwareConvolution(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void scaleFissionFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
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

        MV_REGISTER_PASS(ScaleFission)
        .setFunc(scaleFissionFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "Adds scale layers around HW ops to utilize more bits of fixed-point number representation in MAC HW units"
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

        unsigned int coEffTotalSize = opIterator->getInputTensor(0)->getShape().totalSize();

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

        if (total_input_size + total_output_size > CMX_STREAM_SIZE){
            // TODO: Take into consideration previous splits.
            splitsOverHeight = (unsigned int) ceil((total_input_size + total_output_size)/CMX_STREAM_SIZE);
        }

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
        amount_marked++;
    }
}

void scaleFissionFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    int numFissions = 0 ;
    int maxHWconvs = 3;
    int numHWconvs = 0;

    // define scale factors
    float scaleUpVars[maxHWconvs] = { 1.0f, 7.6f, 8.0f } ;

    std::cout << "SCALE_FISSION PASS:" << std::endl;
    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if ((opIt->getOpType() == OpType::Conv2D)&&(numHWconvs < maxHWconvs))
        {
            if (opIt->hasAttr("NCE1_Compatible"))
            {
                if (opIt->getAttr("NCE1_Compatible").getContent<int>()==1)
                {
                    std::cout << "SCALE_FISSION: detected HW conv "<< opIt->getName() << std::endl;
                    ++numHWconvs;
                    if (numHWconvs <= maxHWconvs)
                    {
                        std::cout << "SCALE_FISSION: detected HW conv "<< opIt->getName() << " inserting scales " << numFissions+1 << std::endl;

                        mv::dynamic_vector<mv::float_type> scaleUpWData = mv::utils::generateSequence<mv::float_type>(opIt->getInputTensor(1)->getShape().totalSize(), scaleUpVars[numFissions], 0.0f);
                        mv::dynamic_vector<mv::float_type> scaleDnData = mv::utils::generateSequence<mv::float_type>(opIt->getOutputTensor(0)->getShape().totalSize(), (1.0f/scaleUpVars[numFissions]), 0.0f);

                        // scale (up) inputs by multiplying weights and bias
//                        auto scaleUpWeights = om.constant(scaleUpWData, opIt->getInputTensor(1)->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar);
                        std::string scaleUpWTensorName = opIt->getName() + "_scale_in";
                        auto scaleUpWeights = dm.defineTensor(scaleUpWTensorName, opIt->getInputTensor(1)->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, scaleUpWData);
                        opIt->getInputTensor(1)->multiply(*scaleUpWeights);

                        if (opIt->hasAttr("bias"))
                        {
                            auto biasTensor = dm.findTensor(opIt->getAttr("bias").getContent<std::string>());
                            mv::dynamic_vector<mv::float_type> scaleUpBData = mv::utils::generateSequence(biasTensor->getShape().totalSize(), scaleUpVars[numFissions], 0.0f);
//                            auto scaleUpBias = om.constant(scaleUpBData, biasTensor->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar);
                            std::string scaleUpBTensorName = opIt->getName() + "_scale_bias";
                            auto scaleUpBias = dm.defineTensor(scaleUpBTensorName, biasTensor->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, scaleUpBData);
                            biasTensor->multiply(*scaleUpBias);
                        }

                        // scale (down) output by adding HWscale attributes to conv
                        std::string scaleTensorName = opIt->getName() + "_scale";
                        auto scaleTensor = dm.defineTensor(scaleTensorName, opIt->getOutputTensor(0)->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, scaleDnData);
                        Attribute scaleAttr(AttrType::StringType, scaleTensor->getName());
                        om.addAttr(opIt, "scale", scaleAttr);
                        // test
                        std::cout << "SCALE_FISSION: added HW scale attributes to "<< opIt->getName() << " hasattrscale= " << opIt->hasAttr("scale") << std::endl;
                        auto testTensor = dm.findTensor(opIt->getAttr("scale").getContent<std::string>());
                        std::cout << "               scale from attribute= "<< testTensor->getData()[0] << std::endl;
                        std::cout << "               name from Tensor= "<< testTensor->getName() << std::endl;
                        ++numFissions;
                    }
                    else
                    {
                        std::cout << "               skipping fission "<< std::endl;
                    }
                }                       
            }
        }
    }
    std::cout << "END SCALE_FISSION PASS:" << std::endl;
}

//NOTE: This should not be done in such hardcoded way.
void formatMXWeights(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
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
std::cout << "exiting formatMXweights pass " << std::endl;
}
