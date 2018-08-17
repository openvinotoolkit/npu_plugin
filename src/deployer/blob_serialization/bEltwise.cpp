#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bEltwise.hpp"

namespace mv
{


    int bEltwise::getSerializedSize(){
        int fields = 0;
        fields += 10;     // Individuals
        fields += 3*10;  // Two buffers

        return fields*4;    // All Ints
    }

    void bEltwise::writeStageInfo(mv::OpModel * om, Blob_buffer* b)
    {
        int fp16_size = 2;

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        printf("Warning: Manual Override of bEltwise Software layer order\n");
        this->output->setOrder(Order::RowMajor);
        this->input0->setOrder(Order::RowMajor);
        this->input1->setOrder(Order::RowMajor);

        Blob_Tensor input0BlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input0);
        Blob_Tensor input1BlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input1);
        Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);

        input0BlobTensor.write(b);
        outputBlobTensor.write(b);
        input1BlobTensor.write(b);

        b->AddBytes(4, 0x03);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x01);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x01);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x04);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x0C);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x0C);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x00);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x00);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x00);  // Hardcoded values for in-place relus
        b->AddBytes(4, 0x00);  // Hardcoded values for in-place relus


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
