//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/utils/dot_graph_writer.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/GraphWriter.h>
#include <llvm/Support/raw_ostream.h>

using namespace vpux;

namespace {

constexpr size_t MAX_EDGE_NUM = 64;
constexpr size_t MAX_ATTR_STR_SIZE = 80;

enum class EdgeDir { EDGE_SKIP, EDGE_NORMAL, EDGE_REVERSE };

// This is a optimized for VPUX copy of the LLVM GraphWriter (llvm/Support/GraphWriter.h).
class GraphWriter final {
public:
    GraphWriter(mlir::Block& block, llvm::raw_ostream& os, const GraphWriterParams& params)
            : _block(block), _os(os), _params(params) {
    }

public:
    void writeGraph() {
        writeHeader();
        writeNodes();
        writeFooter();
    }

private:
    void writeHeader(StringRef title = {});
    void writeNodes();

    void writeFooter() {
        _os << "}\n";
    }

private:
    static EdgeDir getEdgeDirection(mlir::Operation* source, mlir::Operation* target);
    bool isNodeHidden(mlir::Operation* op) const;
    static bool isTaskNode(mlir::Operation* op);
    static bool printNodeAttributes(mlir::Operation* op, llvm::raw_ostream& os);
    static std::string getNodeLabel(mlir::Operation* op);
    void writeNode(mlir::Operation* op);
    void writeEdges(mlir::Operation* op);
    void writeEdge(mlir::Operation* source, mlir::Operation* target);
    void emitEdge(const void* sourceID, const void* targetID);

private:
    mlir::Block& _block;
    llvm::raw_ostream& _os;
    GraphWriterParams _params;
};

void GraphWriter::writeHeader(StringRef title) {
    if (title.empty()) {
        _os << "digraph unnamed {\n";
    } else {
        const auto titleStr = title.str();
        _os << "digraph \"" << llvm::DOT::EscapeString(titleStr) << "\" {\n";
        _os << "\tlabel=\"" << llvm::DOT::EscapeString(titleStr) << "\";\n";
    }

    _os << "\n";
}

void GraphWriter::writeNodes() {
    llvm::DenseSet<mlir::Operation*> processedNodes;

    bool generating = _params.startAfter.empty();
    size_t subGraphId = 0;

    const auto writeNodesAndEdges = [&](mlir::Operation* op) {
        if (!isNodeHidden(op) && !processedNodes.contains(op)) {
            writeNode(op);
            writeEdges(op);
            processedNodes.insert(op);
        }
    };

    for (auto& op : _block) {
        const auto opName = vpux::stringifyLocation(op.getLoc());

        if (generating) {
            if (op.getNumRegions() == 0) {
                writeNodesAndEdges(&op);
            } else if (!processedNodes.contains(&op)) {
                _os << "subgraph cluster_" << subGraphId++ << " {\n\tstyle=filled;color=beige;\n";
                writeNode(&op);

                for (auto& region : op.getRegions()) {
                    for (auto& block : region.getBlocks()) {
                        for (auto& innerOp : block) {
                            writeNodesAndEdges(&innerOp);
                        }
                    }
                }

                _os << "}\n";

                writeEdges(&op);
                processedNodes.insert(&op);
            }

            for (auto operand : op.getOperands()) {
                if (auto* producer = operand.getDefiningOp()) {
                    writeNodesAndEdges(producer);
                }
            }
        }

        if (isTaskNode(&op)) {
            if (opName == _params.startAfter) {
                generating = true;
            }
            if (opName == _params.stopBefore) {
                break;
            }
        }
    }
}

EdgeDir GraphWriter::getEdgeDirection(mlir::Operation* source, mlir::Operation* target) {
    auto sideEffectsOp = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(target);
    if (sideEffectsOp == nullptr) {
        return EdgeDir::EDGE_NORMAL;
    }

    const auto targetOperands = target->getOperands();
    const auto it = llvm::find_if(targetOperands, [&](mlir::Value arg) {
        return arg.getDefiningOp() == source;
    });
    VPUX_THROW_UNLESS(it != targetOperands.end(), "Wrong edge");

    const auto val = *it;
    if (sideEffectsOp.getEffectOnValue<mlir::MemoryEffects::Write>(val).hasValue()) {
        return EdgeDir::EDGE_REVERSE;
    }

    return EdgeDir::EDGE_NORMAL;
}

bool GraphWriter::isNodeHidden(mlir::Operation* op) const {
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return true;
    }

    if (op->hasTrait<mlir::OpTrait::ConstantLike>() && !_params.printConst) {
        return true;
    }

    const auto isDeclarationOp = [](mlir::Operation* op) {
        if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
            return false;
        }

        return op->hasTrait<DeclarationOp>() || mlir::isa<mlir::memref::AllocOp>(op);
    };

    if (isDeclarationOp(op) && !_params.printDeclarations) {
        return true;
    }

    return false;
}

bool GraphWriter::isTaskNode(mlir::Operation* op) {
    if (mlir::isa<mlir::memref::AllocOp>(op) || mlir::isa<mlir::memref::DeallocOp>(op)) {
        return false;
    }

    if (op->hasTrait<DeclarationOp>() || op->hasTrait<mlir::OpTrait::ConstantLike>() ||
        op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return false;
    }

    return true;
}

bool GraphWriter::printNodeAttributes(mlir::Operation* op, llvm::raw_ostream& os) {
    if (auto dotInterface = mlir::dyn_cast<DotInterface>(op)) {
        switch (dotInterface.getNodeColor()) {
        case DotNodeColor::RED:
            os << " style=filled, fillcolor=red";
            return true;
        case DotNodeColor::GREEN:
            os << " style=filled, fillcolor=green";
            return true;
        case DotNodeColor::AQUA:
            os << " style=filled, fillcolor=cyan";
            return true;
        case DotNodeColor::BLUE:
            os << " style=filled, fillcolor=blue";
            return true;
        case DotNodeColor::ORANGE:
            os << " style=filled, fillcolor=orange";
            return true;
        case DotNodeColor::AQUAMARINE:
            os << " style=filled, fillcolor=aquamarine";
            return true;
        default:
            return false;
        }
    }

    return false;
}

std::string GraphWriter::getNodeLabel(mlir::Operation* op) {
    std::string ostr;
    llvm::raw_string_ostream os(ostr);

    // Reuse the print output for the node labels.
    os << op->getName() << "\n";

    if (auto dotInterface = mlir::dyn_cast<DotInterface>(op)) {
        // In case Operation implements custom attribute printer skip default attributes printing
        if (dotInterface.printAttributes(os)) {
            os << "\n";
            return os.str();
        }
    }

    if (!mlir::isa<mlir::async::ExecuteOp>(op)) {
        os << stringifyLocation(op->getLoc()) << "\n";

        // Print resultant types
        for (const auto type : op->getResultTypes()) {
            if (const auto memref = type.dyn_cast<mlir::MemRefType>()) {
                for (auto dim : memref.getShape()) {
                    if (mlir::ShapedType::isDynamic(dim)) {
                        os << '?';
                    } else {
                        os << dim;
                    }
                    os << 'x';
                }

                std::string temp_str;
                llvm::raw_string_ostream temp_os(temp_str);
                memref.getElementType().print(temp_os);

                if (temp_str.size() < MAX_ATTR_STR_SIZE) {
                    os << temp_str;
                } else {
                    os << temp_str.substr(0, MAX_ATTR_STR_SIZE) << "[...]";
                }

                os << " #";
                DimsOrder::fromType(memref).printFormat(os);

                if (memref.getMemorySpace() != nullptr) {
                    os << " at " << memref.getMemorySpace();
                }
            } else {
                os << type;
            }

            os << ", ";
        }

        os << "\n";
    }

    for (const auto attr : op->getAttrs()) {
        os << '\n' << attr.first << ": ";

        std::string temp_str;
        llvm::raw_string_ostream temp_os(temp_str);
        attr.second.print(temp_os);

        if (temp_str.size() < MAX_ATTR_STR_SIZE) {
            os << temp_str;
        } else {
            os << "[...]";
        }
    }

    return os.str();
}

void GraphWriter::writeNode(mlir::Operation* op) {
    _os << "\tNode" << static_cast<const void*>(op) << " [shape=record,";

    if (printNodeAttributes(op, _os)) {
        _os << ",";
    }

    _os << "label=\"{";
    _os << llvm::DOT::EscapeString(getNodeLabel(op));
    _os << "}\"];\n";
}

void GraphWriter::writeEdges(mlir::Operation* op) {
    for (auto* target : op->getUsers()) {
        if (!isNodeHidden(target)) {
            writeEdge(op, target);
        }
    }
}

void GraphWriter::writeEdge(mlir::Operation* source, mlir::Operation* target) {
    switch (getEdgeDirection(source, target)) {
    case EdgeDir::EDGE_NORMAL:
        emitEdge(static_cast<const void*>(source), static_cast<const void*>(target));
        break;
    case EdgeDir::EDGE_REVERSE:
        emitEdge(static_cast<const void*>(target), static_cast<const void*>(source));
        break;
    default:
        break;
    }
}

void GraphWriter::emitEdge(const void* sourceID, const void* targetID) {
    _os << "\tNode" << sourceID;
    _os << " -> Node" << targetID;
    _os << ";\n";
}

}  // namespace

mlir::LogicalResult vpux::writeDotGraph(mlir::Block& block, StringRef fileName, const GraphWriterParams& params) {
    int FD = -1;
    const auto EC =
            llvm::sys::fs::openFileForWrite(fileName, FD, llvm::sys::fs::CD_CreateAlways, llvm::sys::fs::OF_Text);
    if ((FD == -1) || (EC && EC != std::errc::file_exists)) {
        llvm::errs() << "Error writing into file " << fileName << "\n";
        return mlir::failure();
    }

    llvm::raw_fd_ostream os(FD, /*shouldClose=*/true);

    GraphWriter writer(block, os, params);
    writer.writeGraph();

    return mlir::success();
}
