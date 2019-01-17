/**
 * @brief Tutorial on basic usage of a compuatiton model
 * 
 * The computation model is an object that represents the computation in the form of a graph.
 * Additionally, the computation model stores all neccessary information that is required for
 * a compilation process that consists of adaptation, optimization, finalization and serialization
 * for a selected target platform. This information is divided into components of the computation model.
 * Each component is exposed from the model as a set of methods (member function) that enables its
 * creation, manipulation and deletion. Typically, creation methods return a handle in the form of
 * an iterator object that allows accessing the component owned internally by the computation model.
 * 
 * The general list of computation components:
 *  - Computation operations - mathematical operations in the form of nodes
 *  - Computation flows - dependencies between operations in the form of edges
 *  - Tensors - multidimensional data structures that are inputs and outputs of operations
 *  - Groups - labeling mechanism for components
 *  - Computation stages - groups of operations inside which a state of computation resources does not change
 *  - Computation resources - memory, computation accelerators and other features provided by 
 *    the target platform used for an execution of a model
 * 
 * Computation components can be categorized into contexts of their usage. There are three main contexts:
 * operation, data and control. Lists of components that belong to a particular context:
 *  - Operation context: computation operation, data flow, group
 *  - Data context: data flow, tensor, memory
 *  - Control context: computation operation, control flow, computation stage, remaining computation resources
 * 
 * The full set of features of the computation model (in the form of available methods calls) is extensive.
 * Because of that each context is explicitly expressed as a class derived from the base class ComputationModel.
 * To conclude, there are three views (object types) of computation model that are available for a developer:
 *  - OpModel (operation context)
 *  - DataModel (data context)
 *  - ControlModel (control context)
 * 
 * The base class ComputationModel in inaccessible directly (is an abstract type). ComputationModel class is defined
 * in include/mcm/computation/model/computation_model.hpp
 * In this tutorial, the basic process of creation and manipulation of model's views will be pictured and the 
 * usage of iterators will be shown.
 * 
 * @file 01_model_basics.cpp
 * @author Stanislaw Maciag
 * @date 2018-08-09
 */


#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

int main()
{


    /*
        Regarding the lifecycle of a computation model, OpModel (view of operation context) is supervisory,
        because it is the only view that allows an independent initialization. Actual creation of a computation
        model is done by the constructor of the OpModel
    */
    mv::OpModel opModel0;

    /*
        TODO: UPDATE Additionally, model provides a logging mechanism that will log important actions. By default logs will
        be redirected to the standard output. The constructor above sets the logging verbose to silent, which
        means that no logs will be visible. There are five levels of verbose: silent, error (log errors only),
        warning (log errors and warnings), info (log errors, warning and important information) and debug (log everything).
        The verbose level can be specified in the call of OpModel's constructor
    */
    mv::OpModel opModel1;

    /*
        Having OpModel defined, two other views - DataModel and ControlModel - can be created out of it. All modifications
        of a model are propagated among all model that have a common initialization ancestor.
    */
    mv::DataModel dataModel0(opModel1);
    mv::ControlModel controlModel0(opModel1);

    /*
        All initialized instances can be further used for an initialization of more views. The cost of initialization using
        existing intance is low, as it is a shallow copy of computation model. For solving real world problems more than one
        view is often required.
    */
    mv::DataModel dataModel1(controlModel0);
    mv::OpModel opModel2(dataModel0);

    /*
        A valid model (in the context of any view) must not define a disjoined computation graph and must have input and 
        output operation defined. An empty model is considered to be invalid.
    */
    if (!opModel1.isValid() || !dataModel0.isValid() || !controlModel0.isValid())
        std::cout << "Empty model is not a valid computation model" << std::endl;
        
    /*
        Input and output operations are special, because they define the boundaries of a computation.
        Currently only one input and output can be defined (will change in the future). Operations can be added to the computation
        model using OpModel view. Below a minimal valid model is being defined.
    */
    auto input = opModel1.input({32, 32, 3}, mv::DType("Float16"), mv::OrderType::ColumnMajor, "OptionalName-input");
    auto output = opModel1.output(input);

    if (opModel1.isValid() && dataModel0.isValid() && controlModel0.isValid())
        std::cout << "Model with an input which is redirected to an output is a valid compuation model" << std::endl;

    /*
        The modification made using opModel1 is reflected in every related view.
    */
    std::cout << "Now dataModel1 has " << dataModel1.tensorsCount() << " tensor, used to have 0 at the beginning" << std::endl;
    std::cout << "The view opModel2 was also affected by the change of the opModel1, now it has an input "
        << opModel2.getInput()->getName() << std::endl;

    /*
        Typically, methods of views that create some components return iterators. Iterators are wrapped smart pointers that can
        be used to access a particular component, which is in fact owned by the computation model. They can be also incremented
        to access another component that is related somehow. The traversal logic of an iterator is case dependent and will be
        discussed separately in following tutorials.
        Below a typical usage of an iterator is shown. If some call of a view returns an iterator, this view provides the method
        that returns end() iterator as well. End() iterator is a special case of an iterator which points to the element after 
        the last element in a sequence. It cannot be dereferenced, but can be used for a comparison, e.g. in a for-loop.
    */
    std::cout << "Computation model viewed by the opModel1 has following operations defined:" << std::endl;
    for (mv::Data::OpListIterator opIt = opModel1.getInput(); opIt != opModel1.opEnd(); ++opIt)
        std::cout << "\t" << opIt->getName() << std::endl;

    /*
        The members of an element pointed by an iterator can be accessed using the access member by pointer operator "->". Some
        iterators provide additional features that can accessed using access member operator ".".
    */
    mv::Data::TensorIterator inputTensor = dataModel1.flowBegin()->getTensor();
    std::cout << "Input tensor has a shape: " << inputTensor->getShape().toString() << std::endl;
    mv::Data::OpListIterator inputOp = opModel1.getInput();
    std::cout << "Input operation has " << inputOp.childrenSize() << " child operation" << std::endl;

    return 0;

}