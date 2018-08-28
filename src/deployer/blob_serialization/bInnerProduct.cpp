#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bInnerProduct.hpp"

namespace mv
{

    void bInnerProduct::writeStageInfo(mv::OpModel * om, mv::Blob_buffer* b)
    {

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);


        if(this->bias_name != "")
            this->bias = dm.findTensor(this->bias_name);
        else
            this->bias = {} ;


        if (1)
        {

            Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
            Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);
            Blob_Tensor tapsBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->taps);
            Blob_Tensor biasBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->taps);

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
            this->bias_name = it->getAttr("bias").getContent<std::string>();
            std::cout << "Conv has Bias" << std::endl;
        }
        else
        {
            this->bias_name = "";
            std::cout << "Conv has no Bias" <<  std::endl;
        }

    }

}
