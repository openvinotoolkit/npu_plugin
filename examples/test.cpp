#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

namespace mv
{

    static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
        const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
        [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
        std::string& errMsg) -> std::pair<bool, std::size_t>
    {

        if (inputs[0]->getShape().ndims() != 3)
        {
            errMsg = "Shape ndims is not equal to 3";
            return {false, 0};
        }

        if (inputs[1]->getShape().ndims() != 4)
        {
            errMsg = "Shape ndims is not equal to 4";
            return {false, 1};
        }
        
        if (inputs[0]->getShape()[2] != inputs[1]->getShape()[2])
        {
            errMsg = "Does not match the channel dimension of input";
            return {false, 1};
        }
        
        auto padding = args.at("padding").get<std::array<unsigned short, 4>>();

        if (inputs[0]->getShape()[0] + padding[0] + padding[1] < inputs[1]->getShape()[0])
        {
            errMsg = "Width exceeds padded input width";
            return {false, 1};
        }
        
        if (inputs[0]->getShape()[1] + padding[2] + padding[3] < inputs[1]->getShape()[1])
        {
            errMsg = "Height exceeds padded input height";
            return {false, 1};
        }

        return {true, 0};

    };
            
    static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
        std::vector<Tensor>&)> outputDefFcn =
        [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
    {

        auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
        auto stride = args.at("stride").get<std::array<unsigned short, 2>>();

        // Make sure that the result of subtract will not be negative
        mv::Shape outputShape({(inputs[0]->getShape()[0] + padding[0] + padding[1] - inputs[1]->getShape()[0]) / stride[0] + 1, (
            inputs[0]->getShape()[1] + padding[2] + padding[3] - inputs[1]->getShape()[1]) / stride[1] + 1, inputs[1]->getShape()[3]});

        outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder()));

    };
   
    namespace op
    {

        MV_REGISTER_OP(Conv)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 4>>("padding")
        .setArg<std::array<unsigned short, 2>>("stride")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn);

    }

}

int main()
{

     // Define blank computation model (op view)
    mv::OpModel om("Model1");

    // Initialize weights data
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);

    // Compose model - use Composition API to create ops and obtain tensors
    auto input = om.input({128, 128, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto weights1 = om.constant(weights1Data, {3, 3, 3, 8}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto conv1 = om.conv2D(input, weights1, {2, 2}, {1, 1, 1, 1});
    om.output(conv1);


    mv::OpModel m1("m1");
    mv::Op o1(m1, "Conv", "conv1", {input, weights1},
        {
            { "padding", std::array<unsigned short, 4>({1, 1, 1, 1}) },
            { "stride", std::array<unsigned short, 2>({2, 2}) }
        }
    );

    std::cout << o1.toString() << std::endl;
    std::cout << o1.getInputTensor("data")->getName() << std::endl;
    std::cout << o1.getOutputTensor(0)->toString() << std::endl;
}