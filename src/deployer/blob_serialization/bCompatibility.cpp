#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bCompatibility.hpp"

namespace mv
{
    void bCompatibility::writeStageInfo(mv::OpModel * om, Blob_buffer* b)
    {
        
        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        if (this->input->getOrder() == this->output->getOrder()){
            printf("Serialization Warning: Manual Override of Conversion layer due to non-difference\n");
            this->input->setOrder(OrderType::ColumnMajor);
        }

        Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
        Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);

        inputBlobTensor.write(b);
        outputBlobTensor.write(b);

    }

    bCompatibility::bCompatibility(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0)))
    {
    }

    int bCompatibility::getSerializedSize(){
        int fields = 0;
        fields += 0;    // Individual
        fields += 2*10 ; // Input, Output
        return fields*4 ;
    }

}
