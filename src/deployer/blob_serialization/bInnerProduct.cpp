#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bInnerProduct.hpp"

namespace mv
{

    void bInnerProduct::writeStageInfo(mv::OpModel * om, mv::Blob_buffer* b)
    {
        if (1)
        {

            int fp16_size = 2;
            mv::DataModel dm(*om);
            mv::ControlModel cm(*om);

            Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
            Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);
            Blob_Tensor tapsBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->taps);

            printf("Warning: Currently no Bias support in FC\n");
            Blob_Tensor biasBlobTensor = Blob_Tensor(
                // this->output->getShape().totalSize(),   // X
                // 0x01,   // Y
                // 0x01,   // Z
                0,
                0,
                0,
                fp16_size,     // X Stride
                0,
                0,
                // fp16_size*this->output->getShape().totalSize(),    // Y Stride
                // fp16_size*this->output->getShape().totalSize(),    // z Stride
                0, // Offset - Memory Manager
                0, // Location - Memory Manager
                0,
                1
            );

            b->reloc_table.push_entry(std::pair<int, bLocation>(666, bLocation::Constant ));

            inputBlobTensor.write(b);
            outputBlobTensor.write(b);
            tapsBlobTensor.write(b);
            biasBlobTensor.write(b);

        }else{
            // Hardware
        }
    }

    bInnerProduct::bInnerProduct(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0))),
          taps((it->getInputTensor(1)))
    {
        if (it->hasAttr("bias"))
        {
            this->bias = it->getAttr("bias").getContent<mv::dynamic_vector<float>>();
        }
        else
        {
            this->bias = mv::dynamic_vector<float>();
        }

    }

}
