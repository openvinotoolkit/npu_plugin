#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>
int main()
{
    mv::CompilationUnit unit("PriorBoxModel");
    mv::OpModel& om = unit.model();
    // Define Params
    auto flip = 1;
    auto clip = 0;
    auto step_w = 0.0;
    auto step_h = 0.0;
    auto offset = 0.5;

    std::vector<double> minSizesData({105.0});
    std::vector<double> maxSizesData({150.0});
    std::vector<double> aspectRatiosData({2.0,3.0});
    std::vector<double> variancesData({0.1,0.1,0.2,0.2});

    auto imageShape = mv::Shape({300,300,1,1});
    auto priorboxesShape = mv::Shape({10,10,1,1});

    // Define tensors
    auto image = om.input("image", imageShape, mv::DType("Float16"), mv::Order::getZMajorID(4));
    std::vector<int64_t> priorboxesData0 = mv::utils::generateSequence<int64_t> (priorboxesShape[0]*priorboxesShape[1]*priorboxesShape[2]*priorboxesShape[3]);
    auto priorboxes = om.constantInt("priorboxes", priorboxesData0,priorboxesShape, mv::DType("Float16"), mv::Order::getZMajorID(4));
    auto minSizes = om.constant("minSizes", minSizesData,{minSizesData.size(),1,1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4));
    auto maxSizes = om.constant("maxSizes", maxSizesData,{maxSizesData.size(),1,1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4));
    auto aspectRatios = om.constant("aspectRatios", aspectRatiosData,{aspectRatiosData.size(),1,1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4));
    auto variances = om.constant("variances", variancesData,{variancesData.size(),1,1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4));
    image->setQuantParams({{0},{1.0},{},{}});
    priorboxes->setQuantParams({{0},{1.0},{},{}});
    minSizes->setQuantParams({{0},{1.0},{},{}});
    maxSizes->setQuantParams({{0},{1.0},{},{}});
    aspectRatios->setQuantParams({{0},{1.0},{},{}});
    variances->setQuantParams({{0},{1.0},{},{}});

    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(priorboxes);
    inputs.push_back(image);
    inputs.push_back(minSizes);
    inputs.push_back(maxSizes);
    inputs.push_back(aspectRatios);
    inputs.push_back(variances);

    // Build Model
    auto priorbox0 = om.priorbox("priorbox", inputs, flip, clip, step_w, step_h, offset);
    priorbox0->setQuantParams({{128},{0.007843137718737125},{-1.0},{1.0}});

    om.output("", priorbox0);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
