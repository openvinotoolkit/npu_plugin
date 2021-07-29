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

#pragma once

#include <cstddef>
#include <iterator>
#include <string>
#include <vector>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/strings.hpp"

namespace llvm {

namespace DOT {  // Private functions...

std::string EscapeString(const std::string& Label);

}  // end namespace DOT

}  // namespace llvm

namespace vpux {

constexpr int MAX_EDGE_NUM = 64;
constexpr int MAX_ATTR_STR_SIZE = 80;
enum class EdgeDir { EDGE_SKIP, EDGE_NORMAL, EDGE_REVERSE };

// This is a optimized for VPUX copy of the LLVM GraphWriter: llvm/Support/GraphWriter.h
// Before implementing new functionality please check if it already present in the original writer
class GraphWriter {
    using NodeRef = mlir::Operation*;
    using ChildIteratorType = mlir::Operation::user_iterator;
    using child_iterator = mlir::Operation::user_iterator;

    // mlir::Operation's destructor is private so use mlir::Operation* instead and use
    // mapped iterator.
    static mlir::Operation* AddressOf(mlir::Operation& op) {
        return &op;
    }
    using nodes_iterator = llvm::mapped_iterator<mlir::Block::iterator, decltype(&AddressOf)>;
    static nodes_iterator nodes_begin(mlir::Block* b) {
        return nodes_iterator(b->begin(), &AddressOf);
    }
    static nodes_iterator nodes_end(mlir::Block* b) {
        return nodes_iterator(b->end(), &AddressOf);
    }
    llvm::raw_ostream& OStream;
    mlir::Block*& Block;
    bool PrintConst, PrintDeclarations;

    auto getEdgeDirection(NodeRef Node, NodeRef TargetNode) {
        auto sideEffectsOp = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(TargetNode);
        if (sideEffectsOp == nullptr) {
            return EdgeDir::EDGE_NORMAL;
        }

        const auto targetOperands = TargetNode->getOperands();
        const auto it = llvm::find_if(targetOperands, [&](mlir::Value arg) {
            return arg.getDefiningOp() == Node;
        });
        VPUX_THROW_UNLESS(it != targetOperands.end(), "Wrong edge");

        const auto val = *it;
        if (sideEffectsOp.getEffectOnValue<mlir::MemoryEffects::Write>(val).hasValue()) {
            return EdgeDir::EDGE_REVERSE;
        }

        return EdgeDir::EDGE_NORMAL;
    }

    bool isNodeHidden(NodeRef Node) {
        if (auto dotInterface = mlir::dyn_cast<DotInterface>(Node)) {
            if (dotInterface.isDeclaration() && !PrintDeclarations)
                return true;
        }
        if (Node->hasTrait<mlir::OpTrait::ConstantLike>() && !PrintConst)
            return true;
        if (Node->hasTrait<mlir::OpTrait::IsTerminator>())
            return true;

        return false;
    }

    std::string getNodeAttributes(NodeRef Node, const mlir::Block*) {
        std::string ret;
        if (auto dotInterface = mlir::dyn_cast<DotInterface>(Node)) {
            auto color = dotInterface.getNodeColor();
            switch (color) {
            case DotNodeColor::RED:
                ret = " style=filled, fillcolor=red";
                break;
            case DotNodeColor::GREEN:
                ret = " style=filled, fillcolor=green";
                break;
            case DotNodeColor::AQUA:
                ret = " style=filled, fillcolor=cyan";
                break;
            case DotNodeColor::BLUE:
                ret = " style=filled, fillcolor=BLUE";
                break;
            case DotNodeColor::ORANGE:
                ret = " style=filled, fillcolor=orange";
                break;
            case DotNodeColor::AQUAMARINE:
                ret = " style=filled, fillcolor=aquamarine";
                break;
            default:
                break;
            }
        }

        return ret;
    }

    std::string getNodeLabel(NodeRef Node, const mlir::Block* /*b*/) {
        // Reuse the print output for the node labels.
        std::string ostr;
        llvm::raw_string_ostream os(ostr);
        os << Node->getName() << "\n";
        if (auto dotInterface = mlir::dyn_cast<DotInterface>(Node)) {
            auto attrStr = dotInterface.printAttributes();
            // In case Operation implements custom attribute printer skip default attributes printing
            if (attrStr.size()) {
                os << attrStr;
                return os.str();
            }
        }
        if (!mlir::isa<mlir::async::ExecuteOp>(Node)) {
            os << vpux::stringifyLocation(Node->getLoc()) << "\n";

            // Print resultant types
            for (auto type : Node->getResultTypes()) {
                if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
                    for (int64_t dim : memref.getShape()) {
                        if (mlir::ShapedType::isDynamic(dim))
                            os << '?';
                        else
                            os << dim;
                        os << 'x';
                    }
                    os << memref.getElementType();
                    os << " #";
                    DimsOrder::fromType(memref).printFormat(os);
                    if (memref.getMemorySpace())
                        os << " at " << memref.getMemorySpace();
                } else
                    os << type;
                os << ", ";
            }
            os << "\n";
        }

        for (auto attr : Node->getAttrs()) {
            os << '\n' << attr.first << ": ";

            std::string temp_str;
            llvm::raw_string_ostream str_os(temp_str);
            attr.second.print(str_os);
            if (temp_str.size() < MAX_ATTR_STR_SIZE)
                os << temp_str;
            else
                os << "[...]";
        }
        return os.str();
    }

    void writeNode(NodeRef Node) {
        std::string NodeAttributes = getNodeAttributes(Node, Block);

        OStream << "\tNode" << static_cast<const void*>(Node) << " [shape=record,";
        if (!NodeAttributes.empty())
            OStream << NodeAttributes << ",";
        OStream << "label=\"{";

        OStream << llvm::DOT::EscapeString(getNodeLabel(Node, Block));

        std::string edgeSourceLabels;
        llvm::raw_string_ostream EdgeSourceLabels(edgeSourceLabels);
        bool hasEdgeSourceLabels = getEdgeSourceLabels(EdgeSourceLabels, Node);

        if (hasEdgeSourceLabels) {
            OStream << "|{" << EdgeSourceLabels.str() << "}";
        }

        OStream << "}\"];\n";  // Finish printing the "node" line
    }

    void writeEdges(NodeRef Node) {
        // Output all of the edges now
        child_iterator EI = Node->user_begin();
        child_iterator EE = Node->user_end();
        for (unsigned i = 0; EI != EE && i != MAX_EDGE_NUM; ++EI, ++i)
            if (!isNodeHidden(*EI))
                writeEdge(Node, i, EI);
        for (; EI != EE; ++EI)
            if (!isNodeHidden(*EI))
                writeEdge(Node, MAX_EDGE_NUM, EI);
    }

    void writeEdge(NodeRef Node, unsigned edgeidx, child_iterator EI) {
        if (NodeRef TargetNode = *EI) {
            int DestPort = -1;

            if (getEdgeSourceLabel(Node, EI).empty())
                edgeidx = -1;

            switch (getEdgeDirection(Node, TargetNode)) {
            case EdgeDir::EDGE_NORMAL:
                emitEdge(static_cast<const void*>(Node), edgeidx, static_cast<const void*>(TargetNode), DestPort,
                         getEdgeAttributes(Node, EI, Block));
                break;
            case EdgeDir::EDGE_REVERSE:
                emitEdge(static_cast<const void*>(TargetNode), DestPort, static_cast<const void*>(Node), edgeidx,
                         getEdgeAttributes(Node, EI, Block));
                break;
            default:
                break;
            }
        }
    }

    std::string getEdgeSourceLabel(const void*, child_iterator) {
        return "";
    }

    bool hasEdgeDestLabels() {
        return false;
    }

    std::string getEdgeAttributes(NodeRef, child_iterator, mlir::Block*) {
        return "";
    }

    // Writes the edge labels of the node to OStream and returns true if there are any
    // edge labels not equal to the empty string "".
    bool getEdgeSourceLabels(llvm::raw_ostream& OStream, NodeRef Node) {
        child_iterator EI = Node->user_begin();
        child_iterator EE = Node->user_end();
        bool hasEdgeSourceLabels = false;

        for (unsigned i = 0; EI != EE && i != MAX_EDGE_NUM; ++EI, ++i) {
            std::string label = getEdgeSourceLabel(Node, EI);

            if (label.empty())
                continue;

            hasEdgeSourceLabels = true;

            if (i)
                OStream << "|";

            OStream << "<s" << i << ">" << llvm::DOT::EscapeString(label);
        }

        if (EI != EE && hasEdgeSourceLabels)
            OStream << "|<s64>truncated...";

        return hasEdgeSourceLabels;
    }

public:
    GraphWriter(llvm::raw_ostream& ostream, mlir::Block*& block, bool printConst, bool printDeclarations)
            : OStream(ostream), Block(block), PrintConst(printConst), PrintDeclarations(printDeclarations) {
    }

    void writeGraph() {
        // Output the header for the graph...
        writeHeader("");

        // Emit all of the nodes in the graph...
        writeNodes();

        // Output the end of the graph
        writeFooter();
    }

    void writeHeader(const std::string& Title) {
        std::string GraphName("");

        if (!Title.empty())
            OStream << "digraph \"" << llvm::DOT::EscapeString(Title) << "\" {\n";
        else if (!GraphName.empty())
            OStream << "digraph \"" << llvm::DOT::EscapeString(GraphName) << "\" {\n";
        else
            OStream << "digraph unnamed {\n";

        if (!Title.empty())
            OStream << "\tlabel=\"" << llvm::DOT::EscapeString(Title) << "\";\n";
        else if (!GraphName.empty())
            OStream << "\tlabel=\"" << llvm::DOT::EscapeString(GraphName) << "\";\n";
        OStream << "\n";
    }

    void writeFooter() {
        // Finish off the graph
        OStream << "}\n";
    }

    void writeNodes() {
        unsigned SubGraphId = 0;
        // Loop over the graph, printing it out...
        for (const auto Node : make_range(nodes_begin(Block), nodes_end(Block))) {
            if (Node->getNumRegions() > 0) {
                OStream << "subgraph cluster_" << SubGraphId++ << " {\n\tstyle=filled;color=beige;\n";
                writeNode(Node);
                for (auto& region : Node->getRegions()) {
                    for (auto& block : region.getBlocks()) {
                        for (auto& BodyNode : block) {
                            if (!isNodeHidden(&BodyNode)) {
                                writeNode(&BodyNode);
                                writeEdges(&BodyNode);
                            }
                        }
                    }
                }
                OStream << "}\n";
                writeEdges(Node);
            } else if (!isNodeHidden(Node)) {
                writeNode(Node);
                writeEdges(Node);
            }
        }
    }

    /// emitSimpleNode - Outputs a simple (non-record) node
    void emitSimpleNode(const void* ID, const std::string& Attr, const std::string& Label, unsigned NumEdgeSources = 0,
                        const std::vector<std::string>* EdgeSourceLabels = nullptr) {
        OStream << "\tNode" << ID << "[ ";
        if (!Attr.empty())
            OStream << Attr << ",";
        OStream << " label =\"";
        if (NumEdgeSources)
            OStream << "{";
        OStream << llvm::DOT::EscapeString(Label);
        if (NumEdgeSources) {
            OStream << "|{";

            for (unsigned i = 0; i != NumEdgeSources; ++i) {
                if (i)
                    OStream << "|";
                OStream << "<s" << i << ">";
                if (EdgeSourceLabels)
                    OStream << llvm::DOT::EscapeString((*EdgeSourceLabels)[i]);
            }
            OStream << "}}";
        }
        OStream << "\"];\n";
    }

    /// emitEdge - Output an edge from a simple node into the graph...
    void emitEdge(const void* SrcNodeID, int SrcNodePort, const void* DestNodeID, int DestNodePort,
                  const std::string& Attrs) {
        if (SrcNodePort > MAX_EDGE_NUM)
            return;  // Eminating from truncated part?
        if (DestNodePort > MAX_EDGE_NUM)
            DestNodePort = MAX_EDGE_NUM;  // Targeting the truncated part?

        OStream << "\tNode" << SrcNodeID;
        if (SrcNodePort >= 0)
            OStream << ":s" << SrcNodePort;
        OStream << " -> Node" << DestNodeID;
        if (DestNodePort >= 0 && hasEdgeDestLabels())
            OStream << ":d" << DestNodePort;

        if (!Attrs.empty())
            OStream << "[" << Attrs << "]";
        OStream << ";\n";
    }

    /// getOStream - Get the raw output stream into the graph file. Useful to
    /// write fancy things using addCustomGraphFeatures().
    llvm::raw_ostream& getOStream() {
        return OStream;
    }
};

llvm::raw_ostream& WriteGraph(llvm::raw_ostream& OStream, mlir::Block* Block, bool PrintConst = false,
                              bool PrintDeclarations = false) {
    // Start the graph emission process...
    GraphWriter W(OStream, Block, PrintConst, PrintDeclarations);

    // Emit the graph.
    W.writeGraph();

    return OStream;
}

std::string WriteGraph(std::string Filename, mlir::Block* Block, bool PrintConst = false,
                       bool PrintDeclarations = false) {
    int FD;
    std::error_code EC =
            llvm::sys::fs::openFileForWrite(Filename, FD, llvm::sys::fs::CD_CreateAlways, llvm::sys::fs::OF_Text);
    if (EC == std::errc::file_exists) {
        llvm::errs() << "file exists, overwriting"
                     << "\n";
    } else if (EC) {
        llvm::errs() << "error writing into file"
                     << "\n";
        return "";
    }

    llvm::raw_fd_ostream OStream(FD, /*shouldClose=*/true);

    if (FD == -1) {
        llvm::errs() << "error opening file '" << Filename << "' for writing!\n";
        return "";
    }

    WriteGraph(OStream, Block, PrintConst, PrintDeclarations);
    llvm::errs() << " dotfile written. \n";

    return Filename;
}

}  // end namespace vpux
