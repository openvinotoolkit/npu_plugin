#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bCompatibility.hpp"

namespace mv
{
    void bCompatibility::writeStageInfo(WBuffer* b)
    {
        if (1)
        {
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

            inputBlobTensor.write(b);
            outputBlobTensor.write(b);

        }else{
            // Software
        }
    }

    bCompatibility::bCompatibility(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input(*(it->getInputTensor(0))),
          output(*(it->getOutputTensor(0)))
    {
    }

}
