#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    // Define the primary compilation unit
    mv::CompilationUnit unit("cm_two_branch");

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input({24, 24, 20}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto pool1It = cm.maxpool2D(input, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto pool2It = cm.maxpool2D(pool1It, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto pool3It = cm.maxpool2D(pool1It, {1, 1}, {1, 1}, {0, 0, 0, 0});

    std::vector<mv::Data::TensorIterator> cin = {pool3It, pool2It};
    auto concat1It = cm.concat(cin, 2);
    auto pool4It = cm.maxpool2D(concat1It, {1, 1}, {1, 1}, {0, 0, 0, 0});
    cm.output(pool4It);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);

    // Define the manadatory arguments for passes using compilation descriptor obtained from compilation unit
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("cm_two_branch.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("cm_two_branch.blob");

    // Initialize compilation
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);
    //unit.passManager().disablePass(mv::PassGenre::Adaptation);

    // Run all passes
    unit.run();

    system("dot -Tsvg cm_two_branch.dot -o cm_two_branch.svg");
    system("dot -Tsvg cm_two_branch_adapt.dot -o cm_two_branch_adapt.svg");
    system("dot -Tsvg cm_two_branch_final.dot -o cm_two_branch_final.svg");
    return 0;

}