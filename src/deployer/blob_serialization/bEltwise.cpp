#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bEltwise.hpp"

namespace mv
{


    int bEltwise::getSerializedSize(){
        int fields = 0;
        fields += 10;     // Individuals
        fields += 2*10;  // Two buffers

        return fields*4;    // All Ints
    }

    void bEltwise::writeStageInfo(mv::OpModel * om, Blob_buffer* b)
    {

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        printf("Serialization Warning: Manual Override of bEltwise Software layer order\n");
        this->output->setOrder(Order::RowMajor);
        this->input0->setOrder(Order::RowMajor);
        this->input1->setOrder(Order::RowMajor);

        Blob_Tensor input0BlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input0);
        Blob_Tensor input1BlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input1);
        Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);

        input0BlobTensor.write(b);
        outputBlobTensor.write(b);
        input1BlobTensor.write(b);
        
    }

    bEltwise::bEltwise(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input0((it->getInputTensor(0))),
          input1((it->getInputTensor(1))),
          output((it->getOutputTensor(0)))
    {

    }

}
