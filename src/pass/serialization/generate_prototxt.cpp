#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <stdint.h>
#include <algorithm>
#include <fstream> // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <string>
#include "caffe.pb.h"
#include <iostream>
#include <caffe/caffe.hpp>

static void generateProtoFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateProto)
        .setFunc(generateProtoFcn)
        .setGenre(PassGenre::Serialization)
        .defineArg(json::JSONType::String, "outputPrototxt")
        .defineArg(json::JSONType::String, "outputCaffeModel")
        .setDescription(
            "Generates a caffe prototxt file"
        );
        
    }

}

void generateProtoFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput)
{   

    using namespace mv;

    if (compDesc["GenerateProto"]["outputPrototxt"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate prototxt pass");
    
    if (compDesc["GenerateProto"]["outputCaffeModel"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate prototxt pass");
        
        /*Network object*/
    caffe::NetParameter netParamPrototxt;
    caffe::NetParameter netParamCaffeModel;

    mv::OpModel &opModel = dynamic_cast<mv::OpModel &>(model);

    /*create caffemodel*/
    std::fstream output("weights.caffemodel", std::ios::out | std::ios::binary);

    /*create prototxt*/
    std::ofstream ofs;
    ofs.open("test.prototxt", std::ofstream::out | std::ofstream::trunc);

    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == mv::OpType::Input)
        {
            /*Don't add layer for input*/

            caffe::InputParameter inputParamPrototxt;
            caffe::InputParameter inputParamCaffeModel;

            /*Set name and type of the layer*/
            //inputParam.set_name("Input_0");

            /* add input dimensions*/
            /*create caffemodel*/
            netParamPrototxt.add_input("Input_0");
            netParamCaffeModel.add_input("Input_0");

            netParamPrototxt.add_input_dim(0);
            netParamPrototxt.add_input_dim(1);
            netParamPrototxt.add_input_dim(2);
            netParamPrototxt.add_input_dim(3);

            netParamPrototxt.set_input_dim(0, 1);
            netParamPrototxt.set_input_dim(1, 3);
            netParamPrototxt.set_input_dim(2, 224);
            netParamPrototxt.set_input_dim(3, 224);
        }

        if (opIt->getOpType() == mv::OpType::Conv2D)
        {

            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Convolution");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Convolution");

            /*Get the input operation*/
            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));
            layerParamPrototxt->add_bottom(parentOpIt->getName());
            layerParamCaffeModel->add_bottom(parentOpIt->getName());

            /*Get the output operation*/
            //auto sourceOpIt = opIt.leftmostChild();
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a conv parameter*/
            caffe::ConvolutionParameter *convParamPrototxt = layerParamPrototxt->mutable_convolution_param();
            caffe::ConvolutionParameter *convParamCaffeModel = layerParamCaffeModel->mutable_convolution_param();

            /*Set stride on ConvolutionParameter object*/
            convParamPrototxt->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            convParamCaffeModel->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);

            /*Set kernel on ConvolutionParameter object*/
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));
            convParamPrototxt->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);
            convParamCaffeModel->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);

            /*Set number of output channels*/
            convParamPrototxt->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);
            convParamCaffeModel->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);

            convParamPrototxt->set_bias_term(0);
            convParamCaffeModel->set_bias_term(0);

            /* add weights*/
            caffe::BlobProto *blobProto = layerParamCaffeModel->add_blobs();
            caffe::BlobShape *blobShape = blobProto->mutable_shape();

            blobShape->add_dim(0);
            blobShape->add_dim(1);
            blobShape->add_dim(2);
            blobShape->add_dim(3);

            blobShape->set_dim(0, parentOpIt1->get<mv::Shape>("shape")[3]);
            blobShape->set_dim(1, parentOpIt1->get<mv::Shape>("shape")[2]);
            blobShape->set_dim(2, parentOpIt1->get<mv::Shape>("shape")[1]);
            blobShape->set_dim(3, parentOpIt1->get<mv::Shape>("shape")[0]);

            blobProto->clear_double_data();
            blobProto->clear_double_diff();

            /*ColumnMajor is format for caffemodel*/
            auto weights = opIt->getInputTensor(1);
            weights->setOrder(mv::OrderType::ColumnMajor);
            std::vector<double> caffeModelWeights = (*weights).getData();

            for (unsigned i = 0; i < caffeModelWeights.size(); ++i)
            {
                blobProto->add_double_data(caffeModelWeights[i]);
            }
        }

        if (opIt->getOpType() == mv::OpType::Softmax)
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Softmax");

            layerParamPrototxt->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Softmax");

            /*Get the input operation*/
            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));
            layerParamPrototxt->add_bottom(parentOpIt->getName());
            layerParamCaffeModel->add_bottom(parentOpIt->getName());

            /*Get the output operation*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a softmax parameter*/
        }
    }

    /*create caffemodel*/
    netParamCaffeModel.SerializeToOstream(&output);

    ofs << netParamPrototxt.Utf8DebugString();
    ofs.close();

}