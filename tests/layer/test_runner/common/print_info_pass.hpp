#pragma once
#include <include/mcm/pass/pass_registry.hpp>
#include <include/mcm/op_model.hpp>
#include <include/mcm/computation/model/data_model.hpp>

static void printBlobInfoFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
	struct {
		std::string taskOp{"NaN"};
		std::string inputDType{"NaN"};
		std::string inputOrder{"NaN"};
		std::string inputShape{"NaN"};
		std::string outputShape{"NaN"};
	} blobInfo;

    mv::OpModel om{model};
    mv::DataModel dm{model};

	for (auto op = om.opBegin(); op != om.opEnd(); ++op) {
		if (op->getOpType() != "UPATask") continue;
		blobInfo.taskOp = op->get<std::string>("taskOp");
		break;
    }

    for (auto op = om.opBegin(); op != om.opEnd(); ++op) {
		if (op->getOpType() != "Input") continue;
		blobInfo.inputDType = op->get<mv::DType>("dType").toString();
		blobInfo.inputOrder = op->get<mv::Order>("order").toString();
		blobInfo.inputShape = op->get<mv::Shape>("shape").toString();
		break;
	}

    for (auto op = om.opBegin(); op != om.opEnd(); ++op) {
		if (op->getOpType() != "Output") continue;
		blobInfo.outputShape = op->getInputTensor(0)->getShape().toString();
		break;
	}

	const auto layerType = blobInfo.taskOp == "Custom" ? "Custom" : "Native";

    std::cout << om.getName() << ";" << layerType << ";"
    		  << blobInfo.inputShape << ";" << blobInfo.outputShape << ";"
    		  << blobInfo.inputOrder << ";" << blobInfo.inputDType << "\n";
}

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(PrintInfo)
            .setFunc(printBlobInfoFcn)
            .setDescription("Prints information about one layer blob");
    }
}