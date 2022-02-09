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

#include <vpux_elf/reader.hpp>

#include <vpux_elf/types/symbol_entry.hpp>

#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Example usage is ./simplereader <path-to-elf>" << '\n';
        return 1;
    }

    std::ifstream stream(argv[1], std::ios::binary);
    std::vector<uint8_t> elfBlob((std::istreambuf_iterator<char>(stream)), (std::istreambuf_iterator<char>()));
    stream.close();

    elf::Reader reader(elfBlob.data(), elfBlob.size());

    std::cout << "Number of sections: " << reader.getSectionsNum() << '\n';
    std::cout << "Number of segments: " << reader.getSegmentsNum() << '\n';

    for (size_t i = 0; i < reader.getSectionsNum(); ++i) {
        auto section = reader.getSection(i);
        const auto sectionHeader = section.getHeader();

        if (sectionHeader->sh_type == elf::SHT_SYMTAB) {
            const auto entriesNum = section.getEntriesNum();
            std::cout << "Found a symbol table " << section.getName() << " with " << entriesNum << " entries" << '\n';

            const auto symbols = section.getData<elf::SymbolEntry>();
            for (size_t j = 0; j < entriesNum; ++j) {
                const auto symbol = symbols[j];
                std::cout << j << ") Symbol's value: " << symbol.st_value << '\n';
            }
        }
    }

    return 0;
}
