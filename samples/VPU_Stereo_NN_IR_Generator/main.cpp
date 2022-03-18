// Copyright (C) 2021 Intel Corporation
#include <fstream>
#include <inference_engine.hpp>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "VPU_Stereo_NN_IR_Generator.h"
#include <nlohmann/json.hpp>

using namespace InferenceEngine;
using namespace ngraph;
using Json = nlohmann::json;
namespace ie = InferenceEngine;

bool parseCommandLine(int argc, char* argv[]) {
    std::ostringstream usage;
    usage << "Usage: " << argv[0] << "[<options>]";
    gflags::SetUsageMessage(usage.str());

    std::ostringstream version;
    version << ie::GetInferenceEngineVersion();
    gflags::SetVersionString(version.str());

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h || FLAGS_help) {
        showUsage(argv[0]);
        return false;
    }
    std::cout << "Parameters:" << std::endl;
    std::cout << "    Input width:      " << FLAGS_width << std::endl;
    std::cout << "    Input height:     " << FLAGS_height << std::endl;
    std::cout << "    Output folder:    " << FLAGS_o << std::endl;
    std::cout << "    Json file:        " << FLAGS_json << std::endl;
    std::cout << std::endl;
    return true;
}

std::shared_ptr<ngraph::Function> create_advanced_function() {
    std::ifstream ifs(FLAGS_json);
    Json jf = Json::parse(ifs);

    std::vector<float> bias = jf["value0"]["value1"].get<std::vector<float>>();
    std::vector<float> weights = jf["value0"]["value0"].get<std::vector<float>>();

    std::vector<float16> bias_v(std::begin(bias), std::end(bias));
    std::vector<float16> weights_v(std::begin(weights), std::end(weights));

    auto data = std::make_shared<ngraph::opset3::Parameter>(
            ngraph::element::f16, ngraph::Shape{1, 1, (long unsigned int)FLAGS_height, (long unsigned int)FLAGS_width});
    auto input_low_data = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {0.});
    auto input_high_data = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {255.});
    auto output_low_data = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {0.});
    auto output_high_data = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {255.});
    auto data_fakequantize = std::make_shared<ngraph::opset3::FakeQuantize>(data, input_low_data, input_high_data,
                                                                            output_low_data, output_high_data, 256);

    auto kernel_constant9x9 =
            ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{32, 1, 9, 9}, weights_v);
    auto input_low_kernel = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {-127.});
    auto input_high_kernel = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {127.});
    auto output_low_kernel = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {-127.});
    auto output_high_kernel = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {127.});
    auto kernel9x9_fakequantize = std::make_shared<ngraph::opset3::FakeQuantize>(
            kernel_constant9x9, input_low_kernel, input_high_kernel, output_low_kernel, output_high_kernel, 256);

    Strides strides({1, 1});
    CoordinateDiff pad9x9({4, 4});
    auto convolution9x9 = std::make_shared<ngraph::opset3::Convolution>(data_fakequantize, kernel9x9_fakequantize,
                                                                        strides, pad9x9, pad9x9, strides);

    auto bias_constant = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1, 32, 1, 1}, bias_v);
    auto res = std::make_shared<ngraph::opset3::Add>(convolution9x9, bias_constant);

    auto input_low_res = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {0.});
    auto input_high_res = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {1.});
    auto output_low_res = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {0.});
    auto output_high_res = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {1.});
    auto res_min_max = std::make_shared<ngraph::opset3::Clamp>(res, 0., 1.);

    auto res_fakequantize = std::make_shared<ngraph::opset3::FakeQuantize>(res_min_max, input_low_res, input_high_res,
                                                                           output_low_res, output_high_res, 2);

    return std::make_shared<ngraph::Function>(res_fakequantize, ngraph::ParameterVector{data});
}

int main(int argc, char* argv[]) {
    if (parseCommandLine(argc, argv)) {
        if (!FLAGS_width || !FLAGS_height || (FLAGS_json == "")) {
            std::cerr << "Invalid call. One or more required arguments are missing.\n"
                      << "    For help please use: " << argv[0] << " --help\n"
                      << std::endl;
            return EXIT_FAILURE;
        }

        CNNNetwork net(create_advanced_function());
        net.serialize(FLAGS_o + "/stereo_" + std::to_string(FLAGS_width) + "_" + std::to_string(FLAGS_height) + ".xml",
                      FLAGS_o + "/stereo_" + std::to_string(FLAGS_width) + "_" + std::to_string(FLAGS_height) + ".bin");
        std::cout << "IR successfully generated at the following path: "
                  << FLAGS_o + "/stereo_" + std::to_string(FLAGS_width) + "_" + std::to_string(FLAGS_height) +
                             "(.bin & .xml)\n"
                  << std::endl;
    }
    return EXIT_SUCCESS;
}
