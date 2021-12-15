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

#include "vpux/compiler/backend/ELF.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

std::vector<char> ELF::exportToELF(mlir::ModuleOp module, mlir::TimingScope&, const std::vector<PreProcessInfo>&,
                                   Logger log) {
    log.setName("VPUIP::BackEnd (ELF File)");

    log.trace("Extract 'IE.{0}' from Module (ELF File)", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    elf::Writer elfWriter;
    vpux::OpOrderedMap<elf::writer::Section*> sectionMap;
    std::map<mlir::Operation*, elf::writer::Symbol*> symbolMap;

    // NOTE:
    // TODO(iszilve): normally elf serialization process requires raw data to be serialized first,
    // then symbols, then relocations, simply because of data dependency. However if we could
    // introduce ordering constrains on IR validity, then we could serialize all data in one function
    // and just using a SectionInterface rather than ops themselves.

    log.trace("Serializing 'ELF.{0}' ops", ELF::CreateSectionOp::getOperationName());
    auto sectionOps = netFunc.getOps<ELF::CreateSectionOp>();
    for (auto sectionOp : sectionOps) {
        sectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing 'ELF.{0}' ops", ELF::CreateLogicalsectionOp::getOperationName());
    auto logicalSectionOps = netFunc.getOps<ELF::CreateLogicalsectionOp>();
    for (auto logicalSectionOp : logicalSectionOps) {
        logicalSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing 'ELF.{0}' ops", ELF::CreateSymbolTableSectionOp::getOperationName());
    auto symTabOps = netFunc.getOps<ELF::CreateSymbolTableSectionOp>();
    for (auto symTabOp : symTabOps) {
        symTabOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing 'ELF.{0}' ops", ELF::CreateRelocationSectionOp::getOperationName());
    auto relocSectionOps = netFunc.getOps<ELF::CreateRelocationSectionOp>();
    for (auto relocSection : relocSectionOps) {
        relocSection.serialize(elfWriter, sectionMap, symbolMap);
    }

    return elfWriter.generateELF();
}
