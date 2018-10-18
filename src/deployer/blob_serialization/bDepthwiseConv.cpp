#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bDepthwiseConv.hpp"

namespace mv
{

    void bDepthwiseConv2D::writeStageInfo(mv::OpModel& om, mv::Blob_buffer *b)
    {

        //std::cout << "RADIX : " << this->radixX << "*" <<  this->radixY << std::endl;

        int fp16_size = 2;

        mv::DataModel dm(om);
        mv::ControlModel cm(om);

        mv::Data::TensorIterator conv_bias = dm.tensorEnd();
        mv::Data::TensorIterator conv_scale = dm.tensorEnd();

        if(this->bias_name != "")
        {
            this->bias = dm.findTensor(this->bias_name);
            conv_bias = this->bias;
        }

        if(this->scale_name != "")
        {
            this->scale = dm.findTensor(this->scale_name);
            conv_scale = this->scale;
        }

        // Software

        b->AddBytes(4, this->radixX );
        b->AddBytes(4, this->radixY );
        b->AddBytes(4, this->strideX); //strideX  (0x70)
        b->AddBytes(4, this->strideY); //strideY

        // Ignore asymmetric padding (ignore elements elements p_r and p_b from padding = [p_l, p_r, p_t, p_b])
        b->AddBytes(4, this->padX);  // padX
        b->AddBytes(4, this->padY);  // padY
        b->AddBytes(4, this->padStyle);   // 0x80
        b->AddBytes(4, this->dilation);

        Blob_Tensor inputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->input);
        Blob_Tensor outputBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->output);
        Blob_Tensor tapsBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, this->taps);
        Blob_Tensor biasBlobTensor = Blob_Tensor(dm, cm, b->reloc_table, conv_bias);

        inputBlobTensor.write(b);
        outputBlobTensor.write(b);
        tapsBlobTensor.write(b);
        biasBlobTensor.write(b);

    }

    bDepthwiseConv2D::bDepthwiseConv2D(mv::Control::OpListIterator it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0))),
          taps((it->getInputTensor(1))),
          radixX(it->getInputTensor(1)->getShape()[2]),
          radixY(it->getInputTensor(1)->getShape()[3])
    {

        if (it->hasAttr("bias"))
            this->bias_name = it->get<std::string>("bias");
        else
            this->bias_name = "";

        if (it->hasAttr("scale"))
        {
            this->scale_name = it->get<std::string>("scale");
            std::cout << "   in bConvHW contructor : scale tensor name = "<< this->scale_name << std::endl;
        }
        else
            this->scale_name = "";

        // printf("Serializing a SW Conv\n");
        this->radixX = it->getInputTensor(1)->getShape()[0];
        this->radixY = it->getInputTensor(1)->getShape()[1];
        this->strideX = it->get<std::array<unsigned short, 2>>("stride")[0];
        this->strideY = it->get<std::array<unsigned short, 2>>("stride")[1];
        this->padX = it->get<std::array<unsigned short, 4>>("padding")[0];
        this->padY = it->get<std::array<unsigned short, 4>>("padding")[2];
        this->padStyle = 2; // HARDCODED.
        this->dilation = 1; // HARDCODED.

    }

    bDepthwiseConv2D::~bDepthwiseConv2D()
    {

    }
}
