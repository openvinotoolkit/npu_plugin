#include "include/mcm/utils/compositional_model_recorder.hpp"

std::string mv::CompositionalModelRecorder::toString(const std::array<unsigned short, 2>& arr)
{
	return "{" + std::to_string(arr[0]) + ", " + std::to_string(arr[1]) + "}";
}

std::string mv::CompositionalModelRecorder::toString(const std::array<unsigned short, 4>& arr)
{
	return "{" + std::to_string(arr[0]) + ", " + std::to_string(arr[1])  + ", " +
		std::to_string(arr[2]) + ", " + std::to_string(arr[3]) + ", " + "}";
}

mv::CompositionalModelRecorder::CompositionalModelRecorder(OpModel& model, std::string recordingsPath = "/recordings/") :
ComputationModel(model.name_),
modelRef_(model),
savedRecordingsPath_(recordingsPath)
{

}

mv::CompositionalModelRecorder::~CompositionalModelRecorder()
{

}

mv::Data::TensorIterator mv::CompositionalModelRecorder::input(const Shape& shape, DType dType, Order order, const std::string& name)
{

	/* Input is the first call to the CompositionalModel.
	 *
	 * Open two files to store the generated source code and weight vector definitions.
	 *
	 * Populate both source files with the necessary #includes and main() calls etc.
	 *
	 * Close the files.
	 */
    this->createRecordedSourceFiles();

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Input) << " =" << " rc.input(mv::Shape"
        << shape.toString() << ", " << "mv::DTypeType::" << dType.toString() << ", " << "mv::OrderType::"
        << order.toString() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

    auto result = modelRef_.input(shape, dType, order, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::output(Data::TensorIterator inputTensor, const std::string& name)
{

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Output) << " =" << " rc.output(" << sourceIt0->getName() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	/*Populate the 'compilation passes' and end of the source file i.e. return 0*/
	this->completeRecordedSourceFile();

	auto result = modelRef_.output(inputTensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::constant(const std::vector<double>& data, const Shape& shape,
    DType dType, Order order, const std::string& name)
{
	/*Create weights vector name e.g weights_0, weights_1 */
	std::string weightsVectorName = "weights_" + std::to_string(weightsVectorCounter);
	weightsVectorCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(), std::ofstream::out | std::ofstream::app);

	/*Weights are defined in a separate source file, declared as 'extern in the source file*/
	ss << "extern std::vector<double> " << weightsVectorName << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	/*Write the weights to a separate source file*/
	writeWeightsToFile(data, weightsVectorName);

	outputSourceFile.open(recordedSourceFileNameCpp.c_str(), std::ofstream::out | std::ofstream::app);
	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Constant) << " =" << " rc.constant("
		<< weightsVectorName << ", " << "mv::Shape" << shape.toString() << ", "
        << "mv::DTypeType::" << dType.toString() << ", " <<"mv::OrderType::" << order.toString() << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

    auto result = modelRef_.constant(data, shape, dType, order, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::depthwiseConv2D(Data::TensorIterator inputTensor,
    Data::TensorIterator filtersTensor, std::array<unsigned short, 2> stride,
    std::array<unsigned short, 4> padding, const std::string& name)
{
    /*get the name of the argument(s)*/
    auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
    auto sourceIt1 = modelRef_.getSourceOp(filtersTensor);

    /*open the recording file*/
    outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

    /*Construct a std::string and write to file*/
    ss << "auto " << modelRef_.getOpName_(OpType::DepthwiseConv2D) << " =" << " rc.depthwiseConv2D("
        << sourceIt0->getName() << ", " << sourceIt1->getName() << ", " << toString(stride)
        << ", " << toString(padding) << ")" << ";" << "\n";
    outputSourceFile << ss.str();
    ss.str("");
    outputSourceFile.close();

    auto result = modelRef_.depthwiseConv2D(inputTensor, filtersTensor, stride, padding, name);
    return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::conv2D(Data::TensorIterator inputTensor,
	Data::TensorIterator filtersTensor, std::array<unsigned short, 2> stride,
	std::array<unsigned short, 4> padding, const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(filtersTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
    ss << "auto " << modelRef_.getOpName_(OpType::Conv2D) << " =" << " rc.conv2D("
		<< sourceIt0->getName() << ", " << sourceIt1->getName() << ", " << toString(stride)
		<< ", " << toString(padding) << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.conv2D(inputTensor, filtersTensor, stride, padding, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::matMul(Data::TensorIterator input0Tensor,
	Data::TensorIterator input1Tensor, const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::MatMul) << " =" << " rc.matMul(" << sourceIt0->getName()
		<< ", " << sourceIt1->getName() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.matMul(input0Tensor, input1Tensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::maxpool2D(Data::TensorIterator inputTensor,
	std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride,
	std::array<unsigned short, 4> padding, const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::MaxPool2D) << " =" << " rc.maxpool2D("
		<< sourceIt0->getName()<< ", " << toString(kernelSize) << ", " << toString(stride)
		<< ", " << toString(padding) << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.maxpool2D(inputTensor, kernelSize, stride, padding, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::avgpool2D(Data::TensorIterator inputTensor,
	std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride,
	std::array<unsigned short, 4> padding, const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::AvgPool2D) << " =" << " rc.avgpool2D(" << sourceIt0->getName()
		<< ", " << toString(kernelSize) << ", " << toString(stride) << ", " << toString(padding)
		<< ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.avgpool2D(inputTensor, kernelSize, stride, padding, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::concat(std::vector<mv::Data::TensorIterator>& inputTensors,
	const std::string& name)
{
	/*Create concat vector name e.g concat_0, concat_1 */
	std::string concatVectorName = "concat_" + std::to_string(weightsVectorCounter);
	weightsVectorCounter++;

    /*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*an array holding the names of the TensorIterators*/
    unsigned num_inputs = inputTensors.size();

    /*Begin the vector definition*/
	ss << "std::vector<mv::Data::TensorIterator> " << concatVectorName << "= {";
	outputSourceFile << ss.str();
	ss.str("");

    /*Finish the vector definition*/
	for(unsigned i = 0; i!= num_inputs; i++){
	ss << modelRef_.getSourceOp(inputTensors.at(i))->getName() << ", ";
	}

	/*Overwrite the last ',' and append the closing }*/
	ss.seekp(-2,ss.cur);
	ss << '}'  << ";" << "\n";
	
	/*write to file*/
	outputSourceFile << ss.str();
	ss.str("");

	/*Construct a string of the CompositionalModel concat() call and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Concat) << " =" << " rc.concat(";
	ss << concatVectorName  << ")" << ";" << "\n";
	
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.concat(inputTensors, name);
    return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::batchNorm(Data::TensorIterator inputTensor, Data::TensorIterator meanTensor,
	Data::TensorIterator varianceTensor, Data::TensorIterator offsetTensor, Data::TensorIterator scaleTensor, double varianceEps,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(meanTensor);
	auto sourceIt2 = modelRef_.getSourceOp(varianceTensor);
	auto sourceIt3 = modelRef_.getSourceOp(offsetTensor);
	auto sourceIt4 = modelRef_.getSourceOp(scaleTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::BatchNorm) << " =" << " rc.batchNorm(" << sourceIt0->getName()
		<< ", " << sourceIt1->getName() << ", " << sourceIt2->getName() << ", " << sourceIt3->getName()<< ", "
		<< sourceIt4->getName() << ", " << varianceEps << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.batchNorm(inputTensor, meanTensor, varianceTensor, offsetTensor, scaleTensor, varianceEps, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::scale(Data::TensorIterator inputTensor, Data::TensorIterator scaleTensor,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(scaleTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Scale) << " =" << " rc.scale(" << sourceIt0->getName() << ", "
		<< sourceIt1->getName() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.scale(inputTensor, scaleTensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::relu(Data::TensorIterator inputTensor, const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::ReLU) << " =" << " rc.relu(" << sourceIt0->getName() << ")" <<";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.relu(inputTensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::prelu(Data::TensorIterator inputTensor, Data::TensorIterator negativeSlope,
	const std::string& name)
{
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(negativeSlope);
	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	// << ", " << negative_slope
	ss << "auto " << modelRef_.getOpName_(OpType::PReLU) << " =" << " rc.prelu(" << sourceIt0->getName()
		<< ", " << sourceIt1->getName() << ")" <<";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.prelu(inputTensor, negativeSlope, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::softmax(Data::TensorIterator inputTensor, const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " <<  modelRef_.getOpName_(OpType::Softmax) << " =" << " rc.softmax(" << sourceIt0->getName() << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.softmax(inputTensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::add(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	ss << "auto " << modelRef_.getOpName_(OpType::Add) << " =" << " rc.add(" << sourceIt0->getName() << ", "
		<< sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.add(input0Tensor, input1Tensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::subtract(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Subtract) << " =" << " rc.subtract(" << sourceIt0->getName() << ", "
		<< sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.subtract(input0Tensor, input1Tensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::multiply(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Multiply) << " =" << " rc.multiply("<< sourceIt0->getName() << ", "
		<< sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.multiply(input0Tensor, input1Tensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::divide(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Divide) << " =" << " rc.divide("<< sourceIt0->getName() << ", "
		<< sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.divide(input0Tensor, input1Tensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::reshape(Data::TensorIterator inputTensor, const Shape& shape,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Reshape) << " =" << " rc.reshape(" << sourceIt0->getName() << ", "
		<< shape.toString() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.reshape(inputTensor, shape, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::bias(Data::TensorIterator inputTensor, Data::TensorIterator biasesTensor,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(biasesTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::Bias) << " =" << " rc.bias(" << sourceIt0->getName() <<", " << sourceIt1->getName()
		 << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.bias(inputTensor, biasesTensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::fullyConnected(Data::TensorIterator inputTensor,
	Data::TensorIterator weightsTensor, const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(weightsTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::FullyConnected) << " =" << " rc.fullyConnected("
		<< sourceIt0->getName() << ", " << sourceIt1->getName()<< ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.fullyConnected(inputTensor, weightsTensor, name);
	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::dropOut(Data::TensorIterator inputTensor,
	const std::string& name)
{
	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a std::string and write to file*/
	ss << "auto " << modelRef_.getOpName_(OpType::DropOut) << " =" << " rc.dropOut("
		<< sourceIt0->getName() << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	auto result = modelRef_.dropOut(inputTensor, name);
	return result;
}

mv::Data::OpListIterator mv::CompositionalModelRecorder::getSourceOp(Data::TensorIterator tensor)
{
	auto result = modelRef_.getSourceOp(tensor);
	return result;
}

void mv::CompositionalModelRecorder::addAttr(Data::OpListIterator opIt, const std::string& name, const Attribute& attr)
{
	modelRef_.addAttr(opIt, name, attr);
}

bool mv::CompositionalModelRecorder::isValid() const
{
	auto result = modelRef_.isValid();
	return result;
}

bool mv::CompositionalModelRecorder::isValid(const Data::TensorIterator &it) const
{
	auto result = modelRef_.isValid(it);
	return result;
}

bool mv::CompositionalModelRecorder::isValid(const Data::OpListIterator &it) const
{
	auto result = modelRef_.isValid(it);
    return result;
}

void mv::CompositionalModelRecorder::createRecordedSourceFiles() {

	/* 1) Create the source file to store compositional model API calls.
	 *    The name of the source file is "recorded.cpp"
	 *
	 * 2) Create source file to store the recorded weights data.
	 *    The name of the weights file is "recorded_weights.cpp"
	 *
	 * 3) Create the 'beginning' of each source file i.e. #include's and main().
	 *
	 * 4) This method closes the two source files before exiting.
	 *
	 */

	/*Create source file name*/
	std::string projectRootPath = utils::projectRootPath();
	std::string savedPath = utils::projectRootPath() + savedRecordingsPath_;
	recordedSourceFileName = savedPath + "recorded";
	recordedSourceFileNameCpp = recordedSourceFileName + ".cpp";

	/*Create weights file name and open the file*/
	recordedWeghtsFileName = recordedSourceFileName + "_weights.cpp";

	/*open the source file with the truncate-option to delete previous content*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::trunc);


	if (!outputSourceFile)
	{
		log(Logger::MessageType::MessageError, "Failed to open file to record Compositional Model API calls");
		exit(1);
	}

	outputWeightsFile.open(recordedWeghtsFileName.c_str(), std::ofstream::out | std::ofstream::trunc);

	if (!outputWeightsFile)
	{
		log(Logger::MessageType::MessageError, "Failed to open file to record the weights");
		exit(1);
	}


	/*Include required headers in the weights source file*/
	ws << "#include " << "\"include/mcm/compiler/compilation_unit.hpp\"" << "\n";

	/* Include required headers in the source file and beginning of the source file i.e create main() and create CompilationUnit instance*/
	ss << "#include " << "\"include/mcm/compiler/compilation_unit.hpp\"" << "\n";
	ss << "#include " << "\"include/mcm/utils/data_generator.hpp\"" << "\n" << "\n";
	ss << "int main() {" << "\n" << "\n";

	ss << "mv::CompilationUnit unit(\"recordedNetwork\");" << "\n";
	ss << "mv::CompositionalModel& rc = unit.model();" << "\n" << "\n";

	outputSourceFile << ss.str();
	ss.str("");

	outputSourceFile.close();
	outputWeightsFile.close();
}

void mv::CompositionalModelRecorder::completeRecordedSourceFile() {


	std::string projectRootPath = utils::projectRootPath();
	std::string savedPath = utils::projectRootPath() + savedRecordingsPath_;

	/*Populate the 'compilation passes' and end of the source file i.e. return 0*/

	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Create the end of the source file i.e. the compilation passes and create blob commands */
	ss << "if (!unit.loadTargetDescriptor(mv::Target::ma2480))" << "\n";
	ss << "exit(1);" << "\n" << "\n";

	ss << "unit.compilationDescriptor()[\"GenerateDot\"][\"output\"] = std::string(\"" << recordedSourceFileName <<".dot\");" << "\n";
	ss << "unit.compilationDescriptor()[\"GenerateDot\"][\"scope\"] = std::string(\"ExecOpControlModel\");" << "\n";
	ss << "unit.compilationDescriptor()[\"GenerateDot\"][\"content\"] = std::string(\"full\");" << "\n";
	ss << "unit.compilationDescriptor()[\"GenerateDot\"][\"html\"] = true;" <<  "\n";
	ss << "unit.compilationDescriptor()[\"GenerateBlob\"][\"output\"] = std::string(\"" << recordedSourceFileName  << ".blob\");" << "\n";
	ss << "unit.compilationDescriptor()[\"GenerateJson\"][\"output\"] = std::string(\"" << recordedSourceFileName  << ".json\");" << "\n";

	ss << "unit.initialize();" << "\n";
	ss << "auto result = unit.run();" << "\n";

	ss << "std::cout << result.stringifyPretty() << std::endl;" << "\n";
    ss << "system(\"dot -Tpng " << savedPath << "recorded.dot -o " << savedPath << "recorded.png\");" << "\n";
    ss << "system(\"dot -Tpng " << savedPath << "recorded_adapt.dot -o " << savedPath << "recorded_adapt.png\");" << "\n";
    ss << "system(\"dot -Tpng " << savedPath << "recorded_final.dot -o " << savedPath << "recorded_final.png\");" << "\n";

	ss << "return 0;" << "\n";
	ss << "}" << "\n";

	outputSourceFile << ss.str();
	ss.str("");

	/*close the files*/
	outputSourceFile.close();
}


void mv::CompositionalModelRecorder::writeWeightsToFile(const std::vector<double>& weightsData, std::string weightsVectorName) {

	outputWeightsFile.open(recordedWeghtsFileName.c_str(),std::ofstream::out | std::ofstream::app);

	/*Create start of vector definition*/
	ws << "std::vector<double> " << weightsVectorName << "{";
	outputWeightsFile << ws.str();
	ws.str("");

	/*Write the value to the stream*/
	for(std::vector<double>::const_iterator i = weightsData.begin(); i != weightsData.end(); ++i) {
		ws << *i << ", ";
	}

	/*Overwrite the last ',' and append the closing } and clear the stream */
	ws.seekp(-2,ss.cur);

	/*Complete end of vector definition*/
	ws << '}'  << ";" << "\n" << "\n";
	outputWeightsFile  << ws.str();
	ws.str("");
	outputWeightsFile.close();

}

std::string mv::CompositionalModelRecorder::getLogID() const
{
	return "RecordedModel " + name_;
}
