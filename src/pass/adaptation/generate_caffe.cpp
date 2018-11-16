#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "caffe.pb.h"
#include <caffe/caffe.hpp>

static void generateCaffeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &compDesc, mv::json::Object &compOutput);

namespace mv
{
    namespace pass
    {
        //TODO: This pass should be moved to a validation pass in future. It is a temporary adaptation pass until there is functionality to selectively
        //run validation passes when specified.
        
        MV_REGISTER_PASS(GenerateCaffe)
        .setFunc(generateCaffeFcn)
        .setGenre(PassGenre::Adaptation)
        .defineArg(json::JSONType::String, "outputPrototxt")
        .defineArg(json::JSONType::String, "outputCaffeModel")
        .setDescription("Generates a caffe prototxt file");
        
        } // namespace pass
} // namespace mv

void generateCaffeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &compDesc, mv::json::Object &compOutput)
{
    using namespace mv;

    if (compDesc["GenerateCaffe"]["outputPrototxt"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate prototxt pass");

    if (compDesc["GenerateCaffe"]["outputCaffeModel"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate prototxt pass");

    /*Create generated Prototxt and CaffeModel file names*/
    std::string generatedPrototxtFileName = compDesc["GenerateCaffe"]["outputPrototxt"].get<std::string>();
    std::string generatedCaffeModelFileName = compDesc["GenerateCaffe"]["outputCaffeModel"].get<std::string>();

    /*Create Network objects*/
    caffe::NetParameter netParamPrototxt;
    caffe::NetParameter netParamCaffeModel;

    mv::OpModel opModel(model);

    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "Input")
        {
            /*Create layers*/
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Input");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Input");

            /*Set layer to have a input parameter*/
            caffe::InputParameter *inputParamPrototxt = layerParamPrototxt->mutable_input_param();
            caffe::InputParameter *inputParamCaffeModel = layerParamCaffeModel->mutable_input_param();

            caffe::BlobShape *blobShapePrototxt = inputParamPrototxt->add_shape();
            caffe::BlobShape *blobShapeCaffeModel = inputParamCaffeModel->add_shape();

            /*Dimensions for prototxt*/
            blobShapePrototxt->add_dim(0);
            blobShapePrototxt->add_dim(1);
            blobShapePrototxt->add_dim(2);
            blobShapePrototxt->add_dim(3);
            blobShapePrototxt->set_dim(0, 1);
            blobShapePrototxt->set_dim(1, opIt->get<mv::Shape>("shape")[2]);
            blobShapePrototxt->set_dim(2, opIt->get<mv::Shape>("shape")[1]);
            blobShapePrototxt->set_dim(3, opIt->get<mv::Shape>("shape")[0]);

            /*Dimensions for caffemodel*/
            blobShapeCaffeModel->add_dim(0);
            blobShapeCaffeModel->add_dim(1);
            blobShapeCaffeModel->add_dim(2);
            blobShapeCaffeModel->add_dim(3);
            blobShapeCaffeModel->set_dim(0, 1);
            blobShapeCaffeModel->set_dim(1, opIt->get<mv::Shape>("shape")[2]);
            blobShapeCaffeModel->set_dim(2, opIt->get<mv::Shape>("shape")[1]);
            blobShapeCaffeModel->set_dim(3, opIt->get<mv::Shape>("shape")[0]);

            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());
        }

        if (opIt->getOpType() == "Conv")
        {
            /*Create layers*/
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Convolution");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Convolution");

            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));

            /*The bottom attribute stores the name of the input blob*/
            layerParamPrototxt->add_bottom(parentOpIt->getName());
            layerParamCaffeModel->add_bottom(parentOpIt->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a conv parameter*/
            caffe::ConvolutionParameter *convParamPrototxt = layerParamPrototxt->mutable_convolution_param();
            caffe::ConvolutionParameter *convParamCaffeModel = layerParamCaffeModel->mutable_convolution_param();

            /*Set stride*/
            convParamPrototxt->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            convParamCaffeModel->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);

            /*Set padding*/
            convParamPrototxt->add_pad(opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            convParamCaffeModel->add_pad(opIt->get<std::array<unsigned short, 4>>("padding")[0]);

            /*Set kernel*/
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));
            convParamPrototxt->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);
            convParamCaffeModel->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);

            /*Set number of output channels*/
            convParamPrototxt->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);
            convParamCaffeModel->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);

            /*Add weights*/
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

            /*ColumnMajor is format for caffemodel*/
            auto weights = opIt->getInputTensor(1);
            weights->setOrder(mv::Order("NCWH"));

            std::vector<double> caffeModelWeights = weights->getData();

            for (unsigned i = 0; i < caffeModelWeights.size(); ++i)
            {
                blobProto->add_double_data(caffeModelWeights[i]);
            }

            /*Specify if convolution has bias*/
            if (opIt.leftmostChild()->getOpType() == "Bias")
            {
                convParamPrototxt->set_bias_term(1);
                convParamCaffeModel->set_bias_term(1);

                /*add bias*/
                caffe::BlobProto *blobProtobias = layerParamCaffeModel->add_blobs();
                caffe::BlobShape *blobShapebias = blobProtobias->mutable_shape();

                blobShapebias->add_dim(0);
                blobShapebias->set_dim(0, opIt.leftmostChild()->getInputTensor(0)->get<mv::Shape>("shape")[2]);

                blobProtobias->clear_double_data();

                /*ColumnMajor is format for caffemodel*/
                auto bias = opIt.leftmostChild()->getInputTensor(1);
                bias->setOrder(mv::Order("W"));

                std::vector<double> caffeModelBias = bias->getData();

                for (unsigned i = 0; i < caffeModelBias.size(); ++i)
                {
                    blobProtobias->add_double_data(caffeModelBias[i]);
                }
            }
            else
            {
                /*No bias term - set false*/
                convParamPrototxt->set_bias_term(0);
                convParamCaffeModel->set_bias_term(0);
            }
        }

        if (opIt->getOpType() == "DepthwiseConv")
        {
            /*Create layers*/
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Convolution");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Convolution");

            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));

            /*The bottom attribute stores the name of the input blob*/
            layerParamPrototxt->add_bottom(parentOpIt->getName());
            layerParamCaffeModel->add_bottom(parentOpIt->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a conv parameter*/
            caffe::ConvolutionParameter *convParamPrototxt = layerParamPrototxt->mutable_convolution_param();
            caffe::ConvolutionParameter *convParamCaffeModel = layerParamCaffeModel->mutable_convolution_param();

            /*Set stride*/
            convParamPrototxt->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            convParamCaffeModel->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);

            /*Set padding*/
            convParamPrototxt->add_pad(opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            convParamCaffeModel->add_pad(opIt->get<std::array<unsigned short, 4>>("padding")[0]);

            /*Set kernel*/
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));
            convParamPrototxt->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);
            convParamCaffeModel->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);

            /*Set number of output channels*/
            convParamPrototxt->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);
            convParamCaffeModel->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);

            /*Set group for deptwise convolution -> group equals number of kernel channels*/
            convParamPrototxt->has_group();
            convParamCaffeModel->has_group();

            convParamPrototxt->set_group(parentOpIt1->get<mv::Shape>("shape")[2]);
            convParamCaffeModel->set_group(parentOpIt1->get<mv::Shape>("shape")[2]);

            /*Add weights*/
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

            /*ColumnMajor is format for caffemodel*/
            auto weights = opIt->getInputTensor(1);
            weights->setOrder(mv::Order("NCWH"));

            std::vector<double> caffeModelWeights = weights->getData();

            for (unsigned i = 0; i < caffeModelWeights.size(); ++i)
            {
                blobProto->add_double_data(caffeModelWeights[i]);
            }

            /*Specify if convolution has bias*/
            if (opIt.leftmostChild()->getOpType() == "Bias")
            {
                convParamPrototxt->set_bias_term(1);
                convParamCaffeModel->set_bias_term(1);

                /*add bias*/
                caffe::BlobProto *blobProtobias = layerParamCaffeModel->add_blobs();
                caffe::BlobShape *blobShapebias = blobProtobias->mutable_shape();

                blobShapebias->add_dim(0);
                blobShapebias->set_dim(0, opIt.leftmostChild()->getInputTensor(0)->get<mv::Shape>("shape")[2]);

                blobProtobias->clear_double_data();

                /*ColumnMajor is format for caffemodel*/
                auto bias = opIt.leftmostChild()->getInputTensor(1);
                bias->setOrder(mv::Order("W"));

                std::vector<double> caffeModelBias = bias->getData();

                for (unsigned i = 0; i < caffeModelBias.size(); ++i)
                {
                    blobProtobias->add_double_data(caffeModelBias[i]);
                }
            }
            else
            {
                /*No bias term - set false*/
                convParamPrototxt->set_bias_term(0);
                convParamCaffeModel->set_bias_term(0);
            }
        }

        if (opIt->getOpType() == "Softmax")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Softmax");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Softmax");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));

            /*Check if previous op is bias*/
            if (parentOpIt->getOpType() == "Bias")
            {
                auto parentOpIt1 = opModel.getSourceOp(parentOpIt->getInputTensor(0));

                layerParamPrototxt->add_bottom(parentOpIt1->getName());
                layerParamCaffeModel->add_bottom(parentOpIt1->getName());

                /*The top attribute stores the name of the output blob*/
                layerParamPrototxt->add_top(opIt->getName());
                layerParamCaffeModel->add_top(opIt->getName());
            }
            else
            {
                layerParamPrototxt->add_bottom(parentOpIt->getName());
                layerParamCaffeModel->add_bottom(parentOpIt->getName());

                layerParamPrototxt->add_top(opIt->getName());
                layerParamCaffeModel->add_top(opIt->getName());
            }
        }

        if (opIt->getOpType() == "Relu")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("ReLU");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("ReLU");

            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            if (parentOpIt0->getOpType() == "Bias")
            {
                auto parentOpIt1 = opModel.getSourceOp(parentOpIt0->getInputTensor(0));

                /*The bottom attribute stores the name of the input blob*/
                layerParamPrototxt->add_bottom(parentOpIt1->getName());

                /*The top attribute stores the name of the output blob*/
                layerParamPrototxt->add_top(opIt->getName());
                layerParamCaffeModel->add_top(opIt->getName());
            }
            else
            {
                layerParamPrototxt->add_bottom(parentOpIt0->getName());

                /*The top attribute stores the name of the output blob*/
                layerParamPrototxt->add_top(opIt->getName());
                layerParamCaffeModel->add_top(opIt->getName());
            }
        }

        if (opIt->getOpType() == "Elu")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("ELU");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("ELU");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a LRN parameter*/
            caffe::ELUParameter *eluParamPrototxt = layerParamPrototxt->mutable_elu_param();
            caffe::ELUParameter *eluParamCaffeModel = layerParamCaffeModel->mutable_elu_param();

            eluParamPrototxt->set_alpha(opIt->get<unsigned>("alpha"));
            eluParamCaffeModel->set_alpha(opIt->get<unsigned>("alpha"));
        }

         //TODO add bias
         if (opIt->getOpType() == "FullyConnected")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("InnerProduct");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("InnerProduct");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a Inner product parameter*/
            caffe::InnerProductParameter *innerProductParamPrototxt = layerParamPrototxt->mutable_inner_product_param();
            caffe::InnerProductParameter *innerProductParamCaffeModel = layerParamCaffeModel->mutable_inner_product_param();

            /*Get the num_output parameter for inner product from the weights tensor*/
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));
            
            /*Set the number of output channels*/
            innerProductParamPrototxt->set_num_output(parentOpIt1->get<mv::Shape>("shape")[-1]);
            innerProductParamCaffeModel->set_num_output(parentOpIt1->get<mv::Shape>("shape")[-1]);

             /*Add weights to caffemodel*/
            caffe::BlobProto *blobProto = layerParamCaffeModel->add_blobs();
            caffe::BlobShape *blobShape = blobProto->mutable_shape();

            blobShape->add_dim(0);
            blobShape->add_dim(1);
           
            blobShape->set_dim(0, parentOpIt1->get<mv::Shape>("shape")[1]);
            blobShape->set_dim(1, parentOpIt1->get<mv::Shape>("shape")[0]);

            blobProto->clear_double_data();

            /*ColumnMajor is format for caffemodel*/
            auto weights = opIt->getInputTensor(1);
            weights->setOrder(mv::Order("WH"));

            std::vector<double> caffeModelWeights = weights->getData();

            for (unsigned i = 0; i < caffeModelWeights.size(); ++i)
            {
                blobProto->add_double_data(caffeModelWeights[i]);
            }
        }

        if (opIt->getOpType() == "LeakyRelu")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("ReLU");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("ReLU");

            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            if (parentOpIt0->getOpType() == "Bias")
            {
                auto parentOpIt1 = opModel.getSourceOp(parentOpIt0->getInputTensor(0));

                /*The bottom attribute stores the name of the input blob*/
                layerParamPrototxt->add_bottom(parentOpIt1->getName());

                /*The top attribute stores the name of the output blob*/
                layerParamPrototxt->add_top(opIt->getName());
                layerParamCaffeModel->add_top(opIt->getName());
            }
            else
            {
                layerParamPrototxt->add_bottom(parentOpIt0->getName());

                /*The top attribute stores the name of the output blob*/
                layerParamPrototxt->add_top(opIt->getName());
                layerParamCaffeModel->add_top(opIt->getName());
            }

            /*Set layer to have a Relu parameter*/
            caffe::ReLUParameter *reluParamPrototxt = layerParamPrototxt->mutable_relu_param();
            caffe::ReLUParameter *reluParamCaffeModel = layerParamCaffeModel->mutable_relu_param();

            reluParamPrototxt->set_negative_slope(opIt->get<double>("alpha"));
            reluParamCaffeModel->set_negative_slope(opIt->get<double>("alpha"));
        }

        //TODO Need to add bias parameter for LRN 
        if (opIt->getOpType() == "LocalResponseNormalization")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("LRN");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("LRN");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a LRN parameter*/
            caffe::LRNParameter *lrnParamPrototxt = layerParamPrototxt->mutable_lrn_param();
            caffe::LRNParameter *lrnParamCaffeModel = layerParamCaffeModel->mutable_lrn_param();

            lrnParamPrototxt->set_local_size(opIt->get<unsigned>("size"));
            lrnParamCaffeModel->set_local_size(opIt->get<unsigned>("size"));

        }
        if (opIt->getOpType() == "Tanh")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("TanH");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("TanH");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());
        }

        if (opIt->getOpType() == "Sigmoid")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Sigmoid");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Sigmoid");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());
        }

        //TODO - PRELU needs to be tested - disabled in cppwrapper.py
        if (opIt->getOpType() == "Prelu")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("PReLU");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("PReLU");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob, which for convenience, 
              is generally taken to be the same as the name of the layer.*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Store slope data in a blob*/
            caffe::BlobProto *blobProtoprelu = layerParamCaffeModel->add_blobs();
            caffe::BlobShape *blobShapeprelu = blobProtoprelu->mutable_shape();

            blobShapeprelu->add_dim(0);
            blobShapeprelu->set_dim(0, opIt->getInputTensor(0)->get<mv::Shape>("shape")[2]);

            blobProtoprelu->clear_double_data();

            /*ColumnMajor is format for caffemodel*/
            auto slopeData = opIt->getInputTensor(1);
            slopeData->setOrder(mv::Order("W"));

            std::vector<double> preluSlopeData = slopeData->getData();

            for (unsigned i = 0; i < preluSlopeData.size(); ++i)
            {
                blobProtoprelu->add_double_data(preluSlopeData[i]);
            }
        }

        if (opIt->getOpType() == "Scale")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Scale");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Scale");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            auto parentOpIt1 = parentOpIt0.leftmostParent();

            layerParamPrototxt->add_bottom(parentOpIt1->getName());
            layerParamCaffeModel->add_bottom(parentOpIt1->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*add scale data*/
            caffe::BlobProto *blobProtoscale = layerParamCaffeModel->add_blobs();
            caffe::BlobShape *blobShapescale = blobProtoscale->mutable_shape();

            blobShapescale->add_dim(0);
            blobShapescale->set_dim(0, opIt.leftmostChild()->getInputTensor(0)->get<mv::Shape>("shape")[2]);

            blobProtoscale->clear_double_data();

            /*ColumnMajor is format for caffemodel*/
            auto scale = opIt->getInputTensor(1);
            scale->setOrder(mv::Order("W"));

            std::vector<double> caffeModelScale = scale->getData();

            for (unsigned i = 0; i < caffeModelScale.size(); ++i)
            {
                blobProtoscale->add_double_data(caffeModelScale[i]);
            }

            /*Specify if scale has bias*/
            if (opIt.leftmostChild()->getOpType() == "Bias")
            {
                /*Set layer to have a scale parameter*/
                caffe::ScaleParameter *scaleParamPrototxt = layerParamPrototxt->mutable_scale_param();
                caffe::ScaleParameter *scaleParamCaffeModel = layerParamCaffeModel->mutable_scale_param();

                scaleParamPrototxt->set_bias_term(1);
                scaleParamCaffeModel->set_bias_term(1);

                /*add bias*/
                caffe::BlobProto *blobProtobias = layerParamCaffeModel->add_blobs();
                caffe::BlobShape *blobShapebias = blobProtobias->mutable_shape();

                blobShapebias->add_dim(0);
                blobShapebias->set_dim(0, opIt.leftmostChild()->getInputTensor(0)->get<mv::Shape>("shape")[2]);

                blobProtobias->clear_double_data();

                /*ColumnMajor is format for caffemodel*/
                auto bias = opIt.leftmostChild()->getInputTensor(1);

                bias->setOrder(mv::Order("W"));

                std::vector<double> caffeModelBias = bias->getData();

                for (unsigned i = 0; i < caffeModelBias.size(); ++i)
                {
                    blobProtobias->add_double_data(caffeModelBias[i]);
                }
            }
        }

        if (opIt->getOpType() == "MaxPool")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Pooling");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Pooling");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a pooling parameter*/
            caffe::PoolingParameter *poolingParamPrototxt = layerParamPrototxt->mutable_pooling_param();
            caffe::PoolingParameter *poolingParamCaffeModel = layerParamCaffeModel->mutable_pooling_param();

            poolingParamPrototxt->set_kernel_size(opIt->get<std::array<unsigned short, 2>>("kSize")[0]);
            poolingParamPrototxt->set_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            poolingParamPrototxt->set_pad_w(opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            poolingParamPrototxt->set_pad_h(opIt->get<std::array<unsigned short, 4>>("padding")[2]);
            poolingParamPrototxt->set_pool(caffe::PoolingParameter_PoolMethod_MAX);
            

            poolingParamCaffeModel->set_kernel_size(opIt->get<std::array<unsigned short, 2>>("kSize")[0]);
            poolingParamCaffeModel->set_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            poolingParamCaffeModel->set_pad_w(opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            poolingParamCaffeModel->set_pad_h(opIt->get<std::array<unsigned short, 4>>("padding")[2]);
            poolingParamCaffeModel->set_pool(caffe::PoolingParameter_PoolMethod_MAX);
        }

        if (opIt->getOpType() == "AveragePool")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Pooling");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Pooling");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a pooling parameter*/
            caffe::PoolingParameter *poolingParamPrototxt = layerParamPrototxt->mutable_pooling_param();
            caffe::PoolingParameter *poolingParamCaffeModel = layerParamCaffeModel->mutable_pooling_param();

            poolingParamPrototxt->set_kernel_size(opIt->get<std::array<unsigned short, 2>>("kSize")[0]);
            poolingParamPrototxt->set_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            poolingParamPrototxt->set_pad_w(opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            poolingParamPrototxt->set_pad_h(opIt->get<std::array<unsigned short, 4>>("padding")[2]);
            poolingParamPrototxt->set_pool(caffe::PoolingParameter_PoolMethod_AVE);

            poolingParamCaffeModel->set_kernel_size(opIt->get<std::array<unsigned short, 2>>("kSize")[0]);
            poolingParamCaffeModel->set_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            poolingParamCaffeModel->set_pad_w(opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            poolingParamCaffeModel->set_pad_h(opIt->get<std::array<unsigned short, 4>>("padding")[2]);
            poolingParamCaffeModel->set_pool(caffe::PoolingParameter_PoolMethod_AVE);
        }

        if (opIt->getOpType() == "Add")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Eltwise");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Eltwise");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));

            if (parentOpIt0->getOpType() == "Constant" || parentOpIt1->getOpType() == "Constant")
                throw RuntimeError(*parentOpIt0, "The generate prototxt pass does not handle constant inputs to Eltwise Add");

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamPrototxt->add_bottom(parentOpIt1->getName());

            layerParamCaffeModel->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt1->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have an eltwise parameter*/
            layerParamPrototxt->has_eltwise_param();
            layerParamCaffeModel->has_eltwise_param();

            caffe::EltwiseParameter *eltwiseParamPrototxt = layerParamPrototxt->mutable_eltwise_param();
            caffe::EltwiseParameter *eltwiseParamCaffeModel = layerParamCaffeModel->mutable_eltwise_param();

            eltwiseParamPrototxt->set_operation(caffe::EltwiseParameter_EltwiseOp_SUM);
            eltwiseParamCaffeModel->set_operation(caffe::EltwiseParameter_EltwiseOp_SUM);
        }

        if (opIt->getOpType() == "Multiply")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Eltwise");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Eltwise");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));

            if (parentOpIt0->getOpType() == "Constant" || parentOpIt1->getOpType() == "Constant")
                throw RuntimeError(*parentOpIt0, "The generate prototxt pass does not handle constant inputs to Eltwise Prodcut");

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamPrototxt->add_bottom(parentOpIt1->getName());

            layerParamCaffeModel->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt1->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have an eltwise parameter*/
            layerParamPrototxt->has_eltwise_param();
            layerParamCaffeModel->has_eltwise_param();

            caffe::EltwiseParameter *eltwiseParamPrototxt = layerParamPrototxt->mutable_eltwise_param();
            caffe::EltwiseParameter *eltwiseParamCaffeModel = layerParamCaffeModel->mutable_eltwise_param();

            eltwiseParamPrototxt->set_operation(caffe::EltwiseParameter_EltwiseOp_PROD);
            eltwiseParamCaffeModel->set_operation(caffe::EltwiseParameter_EltwiseOp_PROD);
        }

        if (opIt->getOpType() == "Concat")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Concat");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Concat");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamPrototxt->add_bottom(parentOpIt1->getName());

            layerParamCaffeModel->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt1->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());
        }

        if (opIt->getOpType() == "BatchNormalization")
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("BatchNorm");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("BatchNorm");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob*/
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a pooling parameter*/
            caffe::BatchNormParameter *batchNormParamPrototxt = layerParamPrototxt->mutable_batch_norm_param();
            caffe::BatchNormParameter *batchNormParamCaffeModel = layerParamCaffeModel->mutable_batch_norm_param();

            batchNormParamPrototxt->set_use_global_stats(true);
            batchNormParamPrototxt->set_eps(opIt->get<double>("eps"));

            batchNormParamCaffeModel->set_use_global_stats(true);
            batchNormParamCaffeModel->set_eps(opIt->get<double>("eps"));
        }

           //Note: This is meant to be the equivalent of Onnx Gemm. It can either be MatMul or fullyConnected ops. 
        // if (opIt->getOpType() == "MatMul")
        // {
        //     caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
        //     caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

        //     /*Set name and type of the layer*/
        //     layerParamPrototxt->set_name(opIt->getName());
        //     layerParamPrototxt->set_type("InnerProduct");

        //     layerParamCaffeModel->set_name(opIt->getName());
        //     layerParamCaffeModel->set_type("InnerProduct");

        //     /*The bottom attribute stores the name of the input blob*/
        //     auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

        //     layerParamPrototxt->add_bottom(parentOpIt0->getName());
        //     layerParamCaffeModel->add_bottom(parentOpIt0->getName());

        //     /*The top attribute stores the name of the output blob*/
        //     layerParamPrototxt->add_top(opIt->getName());
        //     layerParamCaffeModel->add_top(opIt->getName());
        // }
    }

    /*create caffemodel*/
    std::fstream output(generatedCaffeModelFileName, std::ios::out | std::ios::binary);
    netParamCaffeModel.SerializeToOstream(&output);
    output.close();

    /*create prototxt*/
    std::ofstream ofs;
    ofs.open(generatedPrototxtFileName, std::ofstream::out | std::ofstream::trunc);
    ofs << netParamPrototxt.Utf8DebugString();
    ofs.close();
}
