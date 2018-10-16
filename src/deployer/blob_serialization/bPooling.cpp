#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bPooling.hpp"

namespace mv
{

    int bPooling::getSerializedSize(){
        int fields = 0;
        fields += 7;     // Individuals
        fields += 2*10;  // Two buffers

        return fields*4;    // All Ints
    }


    void bPooling::writeStageInfo(mv::OpModel * om, Blob_buffer* b)
    {

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        //printf("Serialization Warning: Manual Override of Pooling Software layer order\n");
        //this->output->setOrder(OrderType::RowMajor);
        //this->input->setOrder(OrderType::RowMajor);

        Blob_Tensor inputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->input);
        Blob_Tensor outputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->output);

        b->AddBytes(4, this->kernelRadixX);
        b->AddBytes(4, this->kernelRadixY);
        b->AddBytes(4, this->kernelStrideX);
        b->AddBytes(4, this->kernelStrideY);
        b->AddBytes(4, this->kernelPadX);
        b->AddBytes(4, this->kernelPadY);
        b->AddBytes(4, this->kernelPadStyle);

        inputBlobTensor.write(b);
        outputBlobTensor.write(b);

    }

    bPooling::bPooling(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          kernelRadixX(it->get<std::array<unsigned short, 2>>("kSize")[0]),
          kernelRadixY(it->get<std::array<unsigned short, 2>>("kSize")[1]),
          kernelStrideX(it->get<std::array<unsigned short, 2>>("stride")[0]),
          kernelStrideY(it->get<std::array<unsigned short, 2>>("stride")[1]),
          kernelPadX(0),
          kernelPadY(0),
          kernelPadStyle(2),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0)))
    {

    }

}
