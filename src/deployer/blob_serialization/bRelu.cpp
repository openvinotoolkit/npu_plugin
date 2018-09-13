#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bRelu.hpp"

namespace mv
{
    void bRelu::writeStageInfo(mv::OpModel * om, Blob_buffer* b)
    {

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        printf("Serialization Warning: Manual Override of ReLU Software layer order\n");
        this->output->setOrder(OrderType::RowMajor);
        this->input->setOrder(OrderType::RowMajor);

        Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
        Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);

        b->AddBytes(4, this->opX);

        inputBlobTensor.write(b);
        outputBlobTensor.write(b);

        b->AddBytes(4, this->post_strideX);
        b->AddBytes(4, this->post_strideY);

    }

    bRelu::bRelu(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0)))
    {
        this->opX = 0; // HARDCODED.
        this->post_strideX = 0; // HARDCODED.
        this->post_strideY = 0; // HARDCODED.
    }

    int bRelu::getSerializedSize(){
        int fields = 0;
        fields += 3;    // Individual
        fields += 2*10 ; // Input, Output
        return fields*4 ;
    }

}
