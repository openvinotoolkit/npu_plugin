#include "include/mcm/utils/compositional_model_recorder.hpp"

const std::string mv::CompositionalModelRecorder::savedRecordingsPath_ = "/recordings/";


mv::CompositionalModelRecorder::CompositionalModelRecorder(Logger::VerboseLevel verboseLevel, bool logTime, OpModel& model): modelRef_(model)
{
}

mv::CompositionalModelRecorder::~CompositionalModelRecorder()
{
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::input(const Shape& shape, DType dType, Order order, const string& name)
{

	/* Input is the first call to the CompositionalModel. Therefore, open two files
	 * to store generated source code and weight vector definitions*/

    this->createAndOpenFile();

	auto result = modelRef_.input(shape, dType, order, name);

	/*create the unique name to hold return of function call i.e. input_0, input_1*/
	string res = "input_" + std::to_string(inputCounter);
	inputCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.input(mv::Shape" << shape.toString() << ", " << "mv::DType::" << dtypeToString(dType) << ", " <<"mv::Order::" << orderToString(order) << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::output(Data::TensorIterator inputTensor, const string& name)
{
	auto result = modelRef_.output(inputTensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*create the unique name to hold return of function*/
	string res = "output_" + std::to_string(outputCounter);
	outputCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.output(" << sourceIt0->getName() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	/*close the two files*/
	this->closeFile();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name)
{
	auto result = modelRef_.constant(data, shape, dType, order, name);

	/*Create weights vector name e.g weights_0, weights_1 */
	string weightsVectorName = "weights_" + std::to_string(weightsVectorCounter);
	weightsVectorCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Weights are defined in a separate source file, declared as 'extern in the source file*/
	ss << "extern mv::dynamic_vector<mv::float_type> " << weightsVectorName << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	/*Write the weights to a separate source file*/
	writeWeightsToFile(data, weightsVectorName);

	/*create the unique name to hold return of function*/
	string res = "const_" + std::to_string(constantCounter);
	constantCounter++;

	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);
	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.constant(" << weightsVectorName << ", " << "mv::Shape" << shape.toString() << ", " << "mv::DType::" << dtypeToString(dType) << ", " <<"mv::Order::" << orderToString(order) << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::conv2D(Data::TensorIterator inputTensor, Data::TensorIterator filtersTensor, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{
	auto result = modelRef_.conv2D(inputTensor, filtersTensor, stride, padding, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(filtersTensor);

	/*create the unique name to hold return of function*/
	string res = "conv2d_" + std::to_string(conv2DCounter);
	conv2DCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
    ss << "auto " << res << " =" << " rc.conv2D(" << sourceIt0->getName() << ", " << sourceIt1->getName() << ", " << vector2DToString(stride)  <<", " << vector4DToString(padding) << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::matMul(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
	auto result = modelRef_.matMul(input0Tensor, input1Tensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	string res = "matmul_" + std::to_string(multiplyCounter);
	multiplyCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.matMul(" << sourceIt0->getName() << ", " << sourceIt1->getName() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::maxpool2D(Data::TensorIterator inputTensor, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{
	auto result = modelRef_.maxpool2D(inputTensor, kernelSize, stride, padding, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*create the unique name to hold return of function*/
	string res = "maxpool2d_" + std::to_string(maxpool2DCounter);
	maxpool2DCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.maxpool2D(" << sourceIt0->getName()<< ", " << vector2DToString(kernelSize) << ", " << vector2DToString(stride) << ", " << vector4DToString(padding) << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::avgpool2D(Data::TensorIterator inputTensor, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name)
{
	auto result = modelRef_.avgpool2D(inputTensor, kernelSize, stride, padding, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*create the unique name to hold return of function*/
	string res = "avgpool2d_" + std::to_string(avgpool2DCounter);
	avgpool2DCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.avgpool2D(" << sourceIt0->getName() << ", " << vector2DToString(kernelSize) << ", " << vector2DToString(stride) << ", " << vector4DToString(padding) << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::concat(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
	auto result = modelRef_.concat(input0Tensor, input1Tensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	/*create the unique name to hold return of function call*/
	string res = "concat_" + std::to_string(concatCounter);
	concatCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.concat(" << sourceIt0->getName() << ", " << sourceIt1->getName() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

    return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::batchNorm(Data::TensorIterator inputTensor, Data::TensorIterator meanTensor, Data::TensorIterator varianceTensor, Data::TensorIterator offsetTensor, Data::TensorIterator scaleTensor, float_type varianceEps, const string& name)
{
	auto result = modelRef_.batchNorm(inputTensor, meanTensor, varianceTensor, offsetTensor, scaleTensor, varianceEps, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(meanTensor);
	auto sourceIt2 = modelRef_.getSourceOp(varianceTensor);
	auto sourceIt3 = modelRef_.getSourceOp(offsetTensor);
	auto sourceIt4 = modelRef_.getSourceOp(scaleTensor);

	/*create the unique name to hold return of function*/
	string res = "batchnorm_" + std::to_string(batchNormCounter);
	batchNormCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.batchNorm(" << sourceIt0->getName() << ", " << sourceIt1->getName() << ", " << sourceIt2->getName() << ", " << sourceIt3->getName()<< ", " <<sourceIt4->getName() << ", " << varianceEps << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::scale(Data::TensorIterator inputTensor, Data::TensorIterator scaleTensor, const string& name)
{
	auto result = modelRef_.scale(inputTensor, scaleTensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(scaleTensor);

	/*create the unique name to hold return of function*/
	string res = "scale_" + std::to_string(scaleCounter);
	scaleCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.scale(" << sourceIt0->getName() << ", " << sourceIt1->getName() << ")" << ";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::relu(Data::TensorIterator inputTensor, const string& name)
{
	auto result = modelRef_.relu(inputTensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*create the unique name to hold return of function*/
	string res = "relu_" + std::to_string(reluCounter);
	reluCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.relu(" << sourceIt0->getName() << ")" <<";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::softmax(Data::TensorIterator inputTensor, const string& name)
{
	auto result = modelRef_.softmax(inputTensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	/*create the unique name to hold return of function*/
	string res = "softmax_" + std::to_string(softmaxCounter);
	softmaxCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.softmax(" << sourceIt0->getName() << ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::add(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
	auto result = modelRef_.add(input0Tensor, input1Tensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	/*create the unique name to hold return of function*/
	string res = "add_" + std::to_string(addCounter);
	addCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	ss << "auto " << res << " =" << " rc.add(" << sourceIt0->getName() <<", " << sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::subtract(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
	auto result = modelRef_.subtract(input0Tensor, input1Tensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	string res = "subtract_" + std::to_string(subtractCounter);
	subtractCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.subtract(" << sourceIt0->getName() <<", " << sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::multiply(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
	auto result = modelRef_.multiply(input0Tensor, input1Tensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	string res = "multiply_" + std::to_string(multiplyCounter);
	multiplyCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.multiply("<< sourceIt0->getName() <<", " << sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::divide(Data::TensorIterator input0Tensor, Data::TensorIterator input1Tensor, const string& name)
{
	auto result = modelRef_.divide(input0Tensor, input1Tensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(input0Tensor);
	auto sourceIt1 = modelRef_.getSourceOp(input1Tensor);

	string res = "divide_" + std::to_string(divideCounter);
	divideCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.divide("<< sourceIt0->getName() <<", " << sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::reshape(Data::TensorIterator inputTensor, const Shape& shape, const string& name)
{
	auto result = modelRef_.reshape(inputTensor, shape, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);

	string res = "reshape_" + std::to_string(reshapeCounter);
	reshapeCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.reshape(" << sourceIt0->getName() << ", " << shape.toString() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::bias(Data::TensorIterator inputTensor, Data::TensorIterator biasesTensor, const string& name)
{
	auto result = modelRef_.bias(inputTensor, biasesTensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(biasesTensor);

	string res = "bias_" + std::to_string(biasCounter);
	biasCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.bias(" << sourceIt0->getName() <<", " << sourceIt1->getName() << ")" <<";" << "\n";
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::TensorIterator mv::CompositionalModelRecorder::fullyConnected(Data::TensorIterator inputTensor, Data::TensorIterator weightsTensor, const string& name)
{
	auto result = modelRef_.fullyConnected(inputTensor, weightsTensor, name);

	/*get the name of the argument(s)*/
	auto sourceIt0 = modelRef_.getSourceOp(inputTensor);
	auto sourceIt1 = modelRef_.getSourceOp(weightsTensor);

	/*create the unique name to hold return of function*/
	string res = "fullyconnected_" + std::to_string(fullyConnectedCounter);
	fullyConnectedCounter++;

	/*open the recording file*/
	outputSourceFile.open(recordedSourceFileNameCpp.c_str(),std::ofstream::out | std::ofstream::app);

	/*Construct a string and write to file*/
	ss << "auto " << res << " =" << " rc.fullyConnected(" << sourceIt0->getName() << ", " << sourceIt1->getName()<< ")" << ";" << "\n";;
	outputSourceFile << ss.str();
	ss.str("");
	outputSourceFile.close();

	return result;
}

mv::Data::OpListIterator mv::CompositionalModelRecorder::getSourceOp(Data::TensorIterator tensor)
{
	auto result = modelRef_.getSourceOp(tensor);
	return result;
}

bool mv::CompositionalModelRecorder::addAttr(Data::OpListIterator opIt, const string& name, const Attribute& attr)
{
	auto result = modelRef_.addAttr(opIt, name, attr);
	return result;
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

mv::string mv::dtypeToString(DType dType)
{
	switch (dType)
	{
	case DType::Float:   return "Float";
	case DType::Unknown: return "Unknown";
	default:             return "Unknown";
	}
}

mv::string mv::orderToString(Order order)
{
	switch (order)
	{
	case Order::ColumnMajor:   return "ColumnMajor";
	case Order::RowMajor:      return "RowMajor";
	case Order::RowMajorPlanar:return "RowMajorPlanar";
	default:                   return "Unknown";
	}
}



void mv::CompositionalModelRecorder::createAndOpenFile() {

	/* 1) Open the source file to store compositional model API calls.
	 * The name of the source file is recorded.cpp
	 *
	 * 2) Open a source file to store the recorded weights data.
	 * The name of the weights file is recorded_weights.cpp
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


	if (!outputSourceFile) {
		logger_.log(Logger::MessageType::MessageError, "Failed to open file to record Compositional Model API calls");
		exit(1);
	}

	outputWeightsFile.open(recordedWeghtsFileName.c_str(), std::ofstream::out | std::ofstream::trunc);

	if (!outputWeightsFile) {
		logger_.log(Logger::MessageType::MessageError, "Failed to open file to record the weights");
		exit(1);
	}


	/*Include required headers in the weights source file*/
	ws << "#include " << "\"include/mcm/compiler/compilation_unit.hpp\"" << "\n";

	/* Include required headers in the source file and beginning of the source file i.e create main() and create CompilationUnit instance*/
	ss << "#include " << "\"include/mcm/compiler/compilation_unit.hpp\"" << "\n";
	ss << "#include " << "\"include/mcm/utils/data_generator.hpp\"" << "\n" << "\n";
	ss << "int main() {" << "\n" << "\n";

	ss << "mv::Logger::VerboseLevel verboseLevel = mv::Logger::VerboseLevel::VerboseInfo;" <<  "\n";

	ss << "mv::CompilationUnit unit(verboseLevel);" << "\n";
	ss << "mv::CompositionalModel& rc = unit.model();" << "\n" << "\n";

	outputSourceFile << ss.str();
	ss.str("");

	outputSourceFile.close();
	outputWeightsFile.close();
}

void mv::CompositionalModelRecorder::closeFile() {



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

	ss << "return 0;}" << "\n";

	outputSourceFile << ss.str();
	ss.str("");

	/*close the files*/
	outputSourceFile.close();
}


void mv::CompositionalModelRecorder::writeWeightsToFile(const dynamic_vector<float_type>& weightsData, string weightsVectorName) {

	outputWeightsFile.open(recordedWeghtsFileName.c_str(),std::ofstream::out | std::ofstream::app);

	/*Create start of vector definition*/
	ws << "mv::dynamic_vector<mv::float_type> " << weightsVectorName << "{";
	outputWeightsFile << ws.str();
	ws.str("");

	/*Write the value to the stream*/
	for(dynamic_vector<float_type>::const_iterator i = weightsData.begin(); i != weightsData.end(); ++i) {
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









