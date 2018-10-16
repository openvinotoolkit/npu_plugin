#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bSoftmax.hpp"

namespace mv
{

    int bSoftmax::getSerializedSize(){
        int fields = 0;
        fields += 1;     // Individuals
        fields += 2*10;  // Two buffers

        return fields*4;    // All Ints
    }

    void bSoftmax::writeStageInfo(mv::OpModel * om, Blob_buffer* b)
    {

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        //printf("Serialization Warning: Manual Override of Pooling Software layer order\n");
        //this->output->setOrder(OrderType::RowMajor);
        //this->input->setOrder(OrderType::RowMajor);

        Blob_Tensor inputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->input);
        Blob_Tensor outputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->output);

        b->AddBytes(4, this->axis);

        inputBlobTensor.write(b);
        outputBlobTensor.write(b);

    }

    bSoftmax::bSoftmax(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          axis(1),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0)))
    {

    }

}
