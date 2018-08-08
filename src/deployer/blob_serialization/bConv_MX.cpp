#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bConv_MX.hpp"

namespace mv
{

    void bConv2D::writeStageInfo(WBuffer* b)
    {
        if (1)
        {
            // Hardware
            b->AddBytes(4, this->streamingMask);
            b->AddBytes(4, this->input.getShape().totalSize());
            b->AddBytes(4, this->output.getShape().totalSize());
            b->AddBytes(4, this->concatOffset);
            b->AddBytes(4, this->unloadCMX);
            b->AddBytes(4, this->overwriteInput);
            b->AddBytes(4, this->CMXSize);
            b->AddBytes(4, this->reluSHVAcc);
            b->AddBytes(4, this->shvNegSlope);
            b->AddBytes(4, this->shvPosSlope);
            b->AddBytes(4, this->desc_count);

            std::cout << "Streaming Mask: " << this->streamingMask << std::endl;
            std::cout << "Total Input Size: " << this->input.getShape().totalSize() << std::endl;
            std::cout << "Total Output Size: " << this->output.getShape().totalSize() << std::endl;
            std::cout << "concatOffset: " << this->concatOffset << std::endl;
            std::cout << "unloadCMX: " << this->unloadCMX << std::endl;
            std::cout << "overwriteInput: " << this->overwriteInput << std::endl;
            std::cout << "CMXSize: " << this->CMXSize << std::endl;
            std::cout << "reluSHVAcc: " << this->reluSHVAcc << std::endl;
            std::cout << "shvNegSlope: " << this->shvNegSlope << std::endl;
            std::cout << "shvPosSlope: " << this->shvPosSlope << std::endl;
            std::cout << "Desc Count: " << this->desc_count << std::endl;

            for (unsigned i = 0; i != this->desc_count; i++)
            {
                dump_descriptors(&this->descriptors[i]);
                for(unsigned j = 0; j != 32; j++){
                    printf("halfline - %x\n", ((int *) &this->descriptors[i])[j]);
                    b->AddBytes(4, ((int *) &this->descriptors[i])[j]);
                }
            }

            int fp16_size = 2;
            // TODO:

            Blob_Tensor inputBlobTensor = Blob_Tensor(
                this->input.getShape()[0],   // X
                this->input.getShape()[1],   // Y
                this->input.getShape()[2],   // Z
                fp16_size,
                fp16_size*this->input.getShape()[1],
                fp16_size*this->input.getShape()[1]*this->input.getShape()[0],
                -1, // Offset - Memory Manager
                -1, // Location - Memory Manager
                0,
                1
            );
            Blob_Tensor outputBlobTensor = Blob_Tensor(
                this->output.getShape()[0],   // X
                this->output.getShape()[1],   // Y
                this->output.getShape()[2],   // Z
                fp16_size,
                fp16_size*this->output.getShape()[2]*this->output.getShape()[0],
                fp16_size*this->output.getShape()[1],
                 -1, // Offset - Memory Manager
                -1, // Location - Memory Manager
                0,
                2
            );

            Blob_Tensor tapsBlobTensor = Blob_Tensor(
                this->taps.getShape()[3],   // z
                this->taps.getShape()[2],   // y
                this->taps.getShape()[0]*this->taps.getShape()[1],  // X
                fp16_size, // SZ
                fp16_size*this->taps.getShape()[3], // Taps Sy
                fp16_size*this->taps.getShape()[2]*this->taps.getShape()[3],
                -1, // Offset - Memory Manager
                -1, // Location - Memory Manager
                0,
                1
            );
            Blob_Tensor biasBlobTensor = Blob_Tensor(
                // this->output.getShape().totalSize(),   // X
                // 0x01,   // Y
                // 0x01,   // Z
                0,
                0,
                0,
                fp16_size,     // X Stride
                0,
                0,
                // fp16_size*this->output.getShape().totalSize(),    // Y Stride
                // fp16_size*this->output.getShape().totalSize(),    // z Stride
                 -1, // Offset - Memory Manager
                -1, // Location - Memory Manager
                0,
                1
            );

            printf("Warning: Currently no Scale absorb support in HW\n");
            Blob_Tensor scaleBlobTensor = Blob_Tensor(
                // this->taps.getShape()[0]*this->taps.getShape()[1],  // X
                // this->taps.getShape()[2],   // y
                // this->taps.getShape()[3],   // z
                0,
                0,
                0,
                // fp16_size*this->taps.getShape()[2]*this->taps.getShape()[3],
                // fp16_size*this->taps.getShape()[3], // Taps Sy
                0,
                0,
                fp16_size, // SZ
                 -1, // Offset - Memory Manager
                -1, // Location - Memory Manager
                0,
                0
            );

            inputBlobTensor.write(b);
            outputBlobTensor.write(b);
            tapsBlobTensor.write(b);
            biasBlobTensor.write(b);
            scaleBlobTensor.write(b);

        }else{
            // Software
        }
    }

    bConv2D::bConv2D(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input(*(it->getInputTensor(0))),
          output(*(it->getOutputTensor(0))),
          taps(*(it->getInputTensor(1)))
    {


        printf("Serializing a HW Conv\n");

        int cmxSize = 1024*256;
        int descriptors_count = 1;

        if (! it->hasAttr("NCE1_AssignedCMX"))
        {
            printf("WARNING: Needs Attribute 'NCE1_AssignedCMX'. Defaulting to 1024*256\n");
        }
        else
        {
            cmxSize = it->getAttr("NCE1_AssignedCMX").getContent<int>();
        }

        if (! it->hasAttr("NCE1_DescriptorSplits"))
        {
            printf("WARNING: Needs Attribute 'NCE1_DescriptorSplits'. Defaulting to 1\n");
        }
        else
        {
            descriptors_count = it->getAttr("NCE1_DescriptorSplits").getContent<int>();
        }

        if (! it->hasAttr("NCE1_StreamingMask"))
        {
            printf("WARNING: Needs Attribute 'NCE1_StreamingMask'. Defaulting to 1\n");
            this->streamingMask = 1;
        }
        else
        {
            this->streamingMask = it->getAttr("NCE1_StreamingMask").getContent<int>();
        }
        if (! it->hasAttr("NCE1_Mode"))
        {
            printf("WARNING: Needs Attribute 'NCE1_Mode'. Defaulting to 0\n");
            this->opMode = 0;
        }
        else
        {
            this->opMode = it->getAttr("NCE1_Mode").getContent<int>();
        }


        if (it->hasAttr("bias"))
        {
            this->bias = it->getAttr("bias").getContent<mv::dynamic_vector<float>>();
        }
        else
        {
            this->bias = mv::dynamic_vector<float>();
        }

        this->concatOffset = 0; // Concat not supported currently
        this->unloadCMX = 0;
        this->overwriteInput = 0;

        this->CMXSize = cmxSize;
        this->reluSHVAcc = 0;
        float val = 0;
        float val2 = 1;
        this->shvNegSlope = *(int * )(&val);
        this->shvPosSlope = *(int * )(&val2);

        this->desc_count = descriptors_count;

        // this->descriptors = (cnnConvolutionPoolStructure *)malloc(128 * this->desc_count);
        this->descriptors = new cnnConvolutionPoolStructure[this->desc_count];

        int chPerRamBlock = 1;
        int topJunk = 1, bottomJunk = 1;
        int localLS = 1, localCS = 1;
        int LPC = 1;
        int minLines = 1;
        int stride = 1;
        int padEn = 1;


        if (! it->hasAttr("NCE1_InputChannelsPerRamBlock"))
        {
            printf("WARNING: Needs Attribute 'NCE1_InputChannelsPerRamBlock'. Defaulting to 1\n");
        }
        else
        {
            chPerRamBlock = it->getAttr("NCE1_InputChannelsPerRamBlock").getContent<int>();
        }

        if (! it->hasAttr("NCE1_TopOutputJunk"))
        {
            printf("WARNING: Needs Attribute 'NCE1_TopOutputJunk'. Defaulting to 1\n");
        }
        else
        {
            topJunk = it->getAttr("NCE1_TopOutputJunk").getContent<int>();
        }

        if (! it->hasAttr("NCE1_BottomOutputJunk"))
        {
            printf("WARNING: Needs Attribute 'NCE1_BottomOutputJunk'. Defaulting to 1\n");
        }
        else
        {
            bottomJunk = it->getAttr("NCE1_BottomOutputJunk").getContent<int>();
        }

        if (! it->hasAttr("NCE1_LocalLineStride"))
        {
            printf("WARNING: Needs Attribute 'NCE1_LocalLineStride'. Defaulting to 1\n");
        }
        else
        {
            localLS = it->getAttr("NCE1_LocalLineStride").getContent<int>();
        }

        if (! it->hasAttr("NCE1_LocalChannelStride"))
        {
            printf("WARNING: Needs Attribute 'NCE1_LocalChannelStride'. Defaulting to 1\n");
        }
        else
        {
            localCS = it->getAttr("NCE1_LocalChannelStride").getContent<int>();
        }

        if (! it->hasAttr("NCE1_LinesPerChannel"))
        {
            printf("WARNING: Needs Attribute 'NCE1_LinesPerChannel'. Defaulting to 1\n");
        }
        else
        {
            LPC = it->getAttr("NCE1_LinesPerChannel").getContent<int>();
        }

        if (! it->hasAttr("NCE1_MinLines"))
        {
            printf("WARNING: Needs Attribute 'NCE1_MinLines'. Defaulting to 1\n");
        }
        else
        {
            minLines = it->getAttr("NCE1_MinLines").getContent<int>();
        }

        if (! it->hasAttr("stride"))
        {
            printf("WARNING: Needs Attribute 'stride'. Defaulting to 1\n");
        }
        else
        {
            stride = it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0;
        }

        if (! it->hasAttr("padding"))
        {
            printf("WARNING: Needs Attribute 'padding'. Defaulting to 1\n");
        }
        else
        {
            padEn = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0;
        }


        for (unsigned i = 0; i != this->desc_count; i++)
        {
            this->descriptors[i] =  cnnConvolutionPoolStructure();

            // Relations to other Descriptors
            this->descriptors[i].Line0.linkAddress = i*32*4;
            this->descriptors[i].Line0.id = 0;

            // Layer Meta Information - Layout & DataTypes
            this->descriptors[i].Line0.type = NCE1_CONV;
            this->descriptors[i].Line0.interleavedInput = 0;
            this->descriptors[i].Line0.interleavedOutput = 0;
            this->descriptors[i].Line0.cm = NCE1_DTYPE_FP16;
            this->descriptors[i].Line0.dm = NCE1_DTYPE_FP16;

            // Standard Fields for Convolution
            this->descriptors[i].kernelWidth = this->taps.getShape()[0];
            this->descriptors[i].kernelHeight = this->taps.getShape()[1];

            this->descriptors[i].chStride = stride;  // Stride of Kernel (Square only)

            if (padEn > 0)
            {
                this->descriptors[i].padEn = 1;
            }
            else
            {
                this->descriptors[i].padEn = 0;
            }

            this->descriptors[i].padType = 0;   // Zero Padding

            this->descriptors[i].inputWidth = this->input.getShape()[0];
            this->descriptors[i].inputHeight = this->input.getShape()[1];
            this->descriptors[i].inputChannels = this->input.getShape()[2];

            this->descriptors[i].outputChannels = this->output.getShape()[2];

            // Descriptor Buffers

            this->descriptors[i].dataBaseAddr = -1;
            this->descriptors[i].dataChStr = -1;
            this->descriptors[i].dataLnStr = -1;

            this->descriptors[i].coeffBaseAddr = -1;
            this->descriptors[i].coeffChStrOut = -1;
            this->descriptors[i].coeffChStrIn = -1;

            this->descriptors[i].outLnStr = -1;
            this->descriptors[i].outBaseAddr = -1;
            this->descriptors[i].outChStr = -1;

            this->descriptors[i].biasBaseAddr = -1;
            this->descriptors[i].scaleBaseAddr = -1;

            // Myriad X DPU Assignment & Execution Configuration
            this->descriptors[i].Line0.mode = this->opMode;
            this->descriptors[i].Line0.it = 0;  // Interrupt Trigger
            this->descriptors[i].Line0.disInt = 0;  // 0 - Interrupts Enabled, 1 - Interrupts disabled.

            this->descriptors[i].chPerRamBlock = chPerRamBlock;        // Input Channels per Ram Block


            // Myriad X Compensation Fields
            this->descriptors[i].topOutputJunk = topJunk;
            this->descriptors[i].bottomOutputJunk = bottomJunk;

            this->descriptors[i].localLs =  localLS;
            this->descriptors[i].localCs =  localCS;
            this->descriptors[i].linesPerCh = LPC;

            this->descriptors[i].rud = 0;   // Re-Use bit

            this->descriptors[i].minLines = minLines;     // Minimum lines of data required to carry out function

            this->descriptors[i].coeffLpb = this->descriptors[i].chPerRamBlock * this->descriptors[i].kernelWidth * this->descriptors[i].kernelHeight;
            this->descriptors[i].css = this->descriptors[i].kernelWidth * this->descriptors[i].kernelHeight -1 ;
            this->descriptors[i].outputX = this->output.getShape()[2];

            // Myriad X - Splitting groups
            this->descriptors[i].sohGroup = 0;
            this->descriptors[i].sodGroup = 0;

            // Fused ReLU
            this->descriptors[i].t0 = 0;
            this->descriptors[i].a0 = 0;
            this->descriptors[i].a1 = 1;
            this->descriptors[i].reluxEn = 0;
            this->descriptors[i].reluEn = 0;

            // Fused Pooling
            if (0)
            {
                this->descriptors[i].Line0.type = NCE1_CONV_POOL;
            }
            this->descriptors[i].avgPoolX = 0;
            this->descriptors[i].poolType = 0;
            this->descriptors[i].poolEn = 0;
            this->descriptors[i].poolKernelHeight = 0;
            this->descriptors[i].poolKernelWidth = 0;

            // Reserved fields of the hw descriptor. Leave as zero or live in eternal fear.
            this->descriptors[i].Line0.rsvd1 = 0;
            this->descriptors[i].rsvd2 = 0;
            this->descriptors[i].rsvd3 = 0;
            this->descriptors[i].rsvd4 = 0;
            this->descriptors[i].rsvd5 = 0;
            this->descriptors[i].rsvd6 = 0;
            this->descriptors[i].rsvd7 = 0;
            this->descriptors[i].rsvd9 = 0;
            this->descriptors[i].rsvd10 = 0;
            this->descriptors[i].rsvd13 = 0;
            this->descriptors[i].rsvd8 = 0;

            // Palette for Weights Lookup (Currently Unsupported).
            this->descriptors[i].p0 = 0;
            this->descriptors[i].p1 = 0;
            this->descriptors[i].p2 = 0;
            this->descriptors[i].p3 = 0;
            this->descriptors[i].p4 = 0;
            this->descriptors[i].p5 = 0;
            this->descriptors[i].p6 = 0;
            this->descriptors[i].p7 = 0;
            this->descriptors[i].p8 = 0;
            this->descriptors[i].p9 = 0;
            this->descriptors[i].p10 = 0;
            this->descriptors[i].p11 = 0;
            this->descriptors[i].p12 = 0;
            this->descriptors[i].p13 = 0;
            this->descriptors[i].p14 = 0;
            this->descriptors[i].p15 = 0;
        }
    }

}
