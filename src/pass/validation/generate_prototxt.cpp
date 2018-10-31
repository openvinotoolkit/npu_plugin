#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/env_loader.hpp"

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

static void generateProtoFcn(mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &compDesc, mv::json::Object &compOutput);

namespace mv
{

namespace pass
{

MV_REGISTER_PASS(GenerateProto)
    .setFunc(generateProtoFcn)
    .setGenre(PassGenre::Validation)
    .defineArg(json::JSONType::String, "outputPrototxt")
    .defineArg(json::JSONType::String, "outputCaffeModel")
    .setDescription(
        "Generates a caffe prototxt file");
}

} // namespace mv

void generateProtoFcn(mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &compDesc, mv::json::Object &compOutput)
{

    using namespace mv;

    if (compDesc["GenerateProto"]["outputPrototxt"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate prototxt pass");

    if (compDesc["GenerateProto"]["outputCaffeModel"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate prototxt pass");

    /*Create generated Prototxt and CaffeModel file names*/
    std::string projectRootPath = utils::projectRootPath();
    const std::string generatedCaffeFilesPath_ = "/generatedCaffeFiles/";
    std::string savedPath = utils::projectRootPath() + generatedCaffeFilesPath_;
    std::string generatedPrototxtFileName = savedPath + compDesc["GenerateProto"]["outputPrototxt"].get<std::string>();
    std::string generatedCaffeModelFileName = savedPath + compDesc["GenerateProto"]["outputCaffeModel"].get<std::string>();

    /*Create Network objects*/
    caffe::NetParameter netParamPrototxt;
    caffe::NetParameter netParamCaffeModel;

    mv::OpModel &opModel = dynamic_cast<mv::OpModel &>(model);

    for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == mv::OpType::Input)
        {
            /*Don't create a LayerParameter for input */

            /* add input dimensions to the network objects*/
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

            /*Create layers*/
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Convolution");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Convolution");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt = opModel.getSourceOp(opIt->getInputTensor(0));
            layerParamPrototxt->add_bottom(parentOpIt->getName());
            layerParamCaffeModel->add_bottom(parentOpIt->getName());

            /*The top attribute stores the name of the output blob, which for convenience, 
              is generally taken to be the same as the name of the layer.
            */
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a conv parameter*/
            caffe::ConvolutionParameter *convParamPrototxt = layerParamPrototxt->mutable_convolution_param();
            caffe::ConvolutionParameter *convParamCaffeModel = layerParamCaffeModel->mutable_convolution_param();

            /*Set stride on ConvolutionParameter object*/
            convParamPrototxt->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            convParamCaffeModel->add_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);

            /*Set padding on ConvolutionParameter object*/
            convParamPrototxt->add_pad(opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            convParamCaffeModel->add_pad(opIt->get<std::array<unsigned short, 4>>("padding")[0]);

            /*Set kernel on ConvolutionParameter object*/
            auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1));
            convParamPrototxt->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);
            convParamCaffeModel->add_kernel_size(parentOpIt1->get<mv::Shape>("shape")[0]);

            /*Set number of output channels*/
            convParamPrototxt->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);
            convParamCaffeModel->set_num_output(parentOpIt1->get<mv::Shape>("shape")[3]);

            /*add weights*/
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
            weights->setOrder(mv::OrderType::ColumnMajor);

            std::vector<double> caffeModelWeights = (*weights).getData();

            for (unsigned i = 0; i < caffeModelWeights.size(); ++i)
            {
                blobProto->add_double_data(caffeModelWeights[i]);
            }

            /*Specify if convolution has bias*/
            if (opIt.leftmostChild()->getOpType() == mv::OpType::Bias)
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
                bias->setOrder(mv::OrderType::ColumnMajor);

                std::vector<double> caffeModelBias = (*bias).getData();

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

        //TODO Set layer to have a softmax parameter - this may not be required, needs investigation
        if (opIt->getOpType() == mv::OpType::Softmax)
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

            /*Check if previosu op is bias*/
            if (parentOpIt->getOpType() == mv::OpType::Bias)
            {
                auto parentOpIt1 = opModel.getSourceOp(parentOpIt->getInputTensor(0));

                layerParamPrototxt->add_bottom(parentOpIt1->getName());
                layerParamCaffeModel->add_bottom(parentOpIt1->getName());

                /*The top attribute stores the name of the output blob, which for convenience, 
                is generally taken to be the same as the name of the layer.
                */
                layerParamPrototxt->add_top(opIt->getName());
                layerParamCaffeModel->add_top(opIt->getName());
            }
            else
            {
                layerParamPrototxt->add_bottom(parentOpIt->getName());
                layerParamCaffeModel->add_bottom(parentOpIt->getName());

                /*The top attribute stores the name of the output blob, which for convenience, 
                is generally taken to be the same as the name of the layer.
                */
                layerParamPrototxt->add_top(opIt->getName());
                layerParamCaffeModel->add_top(opIt->getName());
            }
        }

        //TODO Set layer to have a relu parameter - this may not be required, needs investigation
        if (opIt->getOpType() == mv::OpType::ReLU)
        {
            caffe::LayerParameter *layerParamPrototxt = netParamPrototxt.add_layer();
            caffe::LayerParameter *layerParamCaffeModel = netParamCaffeModel.add_layer();

            /*Set name and type of the layer*/
            layerParamPrototxt->set_name(opIt->getName());
            layerParamPrototxt->set_type("Relu");

            layerParamCaffeModel->set_name(opIt->getName());
            layerParamCaffeModel->set_type("Relu");

            /*The bottom attribute stores the name of the input blob*/
            auto parentOpIt0 = opModel.getSourceOp(opIt->getInputTensor(0));

            //TODO Deal with fused ops here instead of going back two operations
            /*If this pass runs before the fuse bias pass, then we need to traverse back two operations to get the bottom*/
            auto parentOpIt1 = parentOpIt0.leftmostParent();
            layerParamPrototxt->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob, which for convenience, 
              is generally taken to be the same as the name of the layer.
            */
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());
        }

        //TODO Do you add two bottoms for the two inputs? It appears not from t6-Enet model in MDK
        //Need to store the slope data in a blob in the layer? - Yes done.

        if (opIt->getOpType() == mv::OpType::PReLU)
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
            //auto parentOpIt1 = opModel.getSourceOp(opIt->getInputTensor(1)); //Is this needed??

            //TODO Deal with fused ops here instead of going back two operations
            /*If this pass runs before the fuse bias pass, then we need to traverse back two operations to get the bottom*/
            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());
            //layerParamPrototxt->add_bottom(parentOpIt1->getName()); //Is this needed??

            /*The top attribute stores the name of the output blob, which for convenience, 
              is generally taken to be the same as the name of the layer.
            */
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
            slopeData->setOrder(mv::OrderType::ColumnMajor);

            std::vector<double> preluSlopeData = (*slopeData).getData();

            for (unsigned i = 0; i < preluSlopeData.size(); ++i)
            {
                blobProtoprelu->add_double_data(preluSlopeData[i]);
            }
        }

        if (opIt->getOpType() == mv::OpType::Scale)
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

            //TODO Deal with fused ops here instead of going back two operations
            /*If this pass runs before the fuse bias pass, then we need to traverse back two operations to get the bottom*/
            auto parentOpIt1 = parentOpIt0.leftmostParent();

            layerParamPrototxt->add_bottom(parentOpIt1->getName());
            layerParamCaffeModel->add_bottom(parentOpIt1->getName());

            /*The top attribute stores the name of the output blob, which for convenience, 
              is generally taken to be the same as the name of the layer.
            */
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a conv parameter*/
            caffe::ScaleParameter *scaleParamPrototxt = layerParamPrototxt->mutable_scale_param();
            caffe::ScaleParameter *scaleParamCaffeModel = layerParamCaffeModel->mutable_scale_param();

            /*Specify if scale has bias*/

            /* Case (1) Bias will be a seprate operation before the fuse bias pass*/
            /* Case (2) Bias will be an attribute after the fuse bias pass*/

            // Case(1)
            if (opIt.leftmostChild()->getOpType() == mv::OpType::Bias)
            {
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
                bias->setOrder(mv::OrderType::ColumnMajor);

                std::vector<double> caffeModelBias = (*bias).getData();

                for (unsigned i = 0; i < caffeModelBias.size(); ++i)
                {
                    blobProtobias->add_double_data(caffeModelBias[i]);
                }
            }

            // TODO:Case(2)
        }

        if (opIt->getOpType() == mv::OpType::MaxPool2D)
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

            //TODO Deal with fused ops here instead of going back two operations
            /*If this pass runs before the fuse bias pass, then we need to traverse back two operations to get the bottom*/
            //auto parentOpIt1 = parentOpIt0.leftmostParent();

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob, which for convenience, 
              is generally taken to be the same as the name of the layer.
            */
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a pooling parameter*/
            caffe::PoolingParameter *poolingParamPrototxt = layerParamPrototxt->mutable_pooling_param();
            caffe::PoolingParameter *poolingParamCaffeModel = layerParamCaffeModel->mutable_pooling_param();

            poolingParamPrototxt->set_kernel_size(opIt->get<std::array<unsigned short, 2>>("kSize")[0]);
            poolingParamPrototxt->set_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            poolingParamPrototxt->set_pool(caffe::PoolingParameter_PoolMethod_MAX);

            poolingParamCaffeModel->set_kernel_size(opIt->get<std::array<unsigned short, 2>>("kSize")[0]);
            poolingParamCaffeModel->set_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            poolingParamCaffeModel->set_pool(caffe::PoolingParameter_PoolMethod_MAX);
        }

        if (opIt->getOpType() == mv::OpType::AvgPool2D)
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

            //TODO Deal with fused ops here instead of going back two operations
            /*If this pass runs before the fuse bias pass, then we need to traverse back two operations to get the bottom*/
            //auto parentOpIt1 = parentOpIt0.leftmostParent();

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt0->getName());

            /*The top attribute stores the name of the output blob, which for convenience, 
              is generally taken to be the same as the name of the layer.
            */
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have a pooling parameter*/
            caffe::PoolingParameter *poolingParamPrototxt = layerParamPrototxt->mutable_pooling_param();
            caffe::PoolingParameter *poolingParamCaffeModel = layerParamCaffeModel->mutable_pooling_param();

            poolingParamPrototxt->set_kernel_size(opIt->get<std::array<unsigned short, 2>>("kSize")[0]);
            poolingParamPrototxt->set_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            poolingParamPrototxt->set_pool(caffe::PoolingParameter_PoolMethod_AVE);

            poolingParamCaffeModel->set_kernel_size(opIt->get<std::array<unsigned short, 2>>("kSize")[0]);
            poolingParamCaffeModel->set_stride(opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            poolingParamCaffeModel->set_pool(caffe::PoolingParameter_PoolMethod_AVE);
        }

        if (opIt->getOpType() == mv::OpType::Add)
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

            //TODO Deal with fused ops here instead of going back two operations
            /*If this pass runs before the fuse bias pass, then we need to traverse back two operations to get the bottom*/
            //auto parentOpIt1 = parentOpIt0.leftmostParent();

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamPrototxt->add_bottom(parentOpIt1->getName());

            layerParamCaffeModel->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt1->getName());

            /*The top attribute stores the name of the output blob, which for convenience,
                  is generally taken to be the same as the name of the layer.
                */
            layerParamPrototxt->add_top(opIt->getName());
            layerParamCaffeModel->add_top(opIt->getName());

            /*Set layer to have an eltwise parameter*/
            layerParamPrototxt->has_eltwise_param();
            layerParamCaffeModel->has_eltwise_param();

            caffe::EltwiseParameter *eltwiseParamPrototxt = layerParamPrototxt->mutable_eltwise_param();
            caffe::EltwiseParameter *eltwiseParamCaffeModel = layerParamCaffeModel->mutable_eltwise_param();

            eltwiseParamPrototxt->set_operation(caffe::EltwiseParameter_EltwiseOp_SUM);
            eltwiseParamCaffeModel->set_operation(caffe::EltwiseParameter_EltwiseOp_SUM);

            // if (parentOpIt0->getOpType() == mv::OpType::Constant || parentOpIt1->getOpType() == mv::OpType::Constant)
            //  throw RuntimeError(, "The generate prototxt pass does not handle constant");
        }

        if (opIt->getOpType() == mv::OpType::Multiply)
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

            //TODO Deal with fused ops here instead of going back two operations
            /*If this pass runs before the fuse bias pass, then we need to traverse back two operations to get the bottom*/
            //auto parentOpIt1 = parentOpIt0.leftmostParent();

            layerParamPrototxt->add_bottom(parentOpIt0->getName());
            layerParamPrototxt->add_bottom(parentOpIt1->getName());

            layerParamCaffeModel->add_bottom(parentOpIt0->getName());
            layerParamCaffeModel->add_bottom(parentOpIt1->getName());

            /*The top attribute stores the name of the output blob, which for convenience,
                  is generally taken to be the same as the name of the layer.
                */
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