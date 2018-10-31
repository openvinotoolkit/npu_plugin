#include "include/mcm/deployer/serializer.hpp"
#include <stdio.h>

namespace mv
{
    Serializer::Serializer(serializer_mode set_output_format)
    {
        output_format = set_output_format;
    }

    unsigned long long Serializer::serialize(mv::ControlModel& graph_2_deploy )
    {

        printf("Serializer\n");
        std::shared_ptr<mv::RuntimeBinary> binaryPointer = graph_2_deploy.getBinaryBuffer() ;
        uint32_t blobSize = 0;
        uint64_t fsize = 0 ;
        switch( output_format )
        {
            case mvblob_mode:
                blobSize = odata.calc(graph_2_deploy);
                if (binaryPointer->getRAMEnabled())
                {
                    graph_2_deploy.allocateBinaryBuffer("RAMBlob", blobSize);
                }
                odata.open(binaryPointer);
                odata.write_elf_header();
                odata.write_mv_header();
                odata.write_stage_section_header();
                odata.write_stages(graph_2_deploy);
                odata.write_buffer_section(graph_2_deploy);
                odata.write_relocation_section(graph_2_deploy);
                fsize = odata.End() ;
            break;
            default:
                std::cout << "ERROR: unsupported deployment output format " << output_format << std::endl;
            break;
        }
        return (fsize);
    }

    void Serializer::print_mode()
    {
        std::cout << "serializer output mode= " << output_format << std::endl;
    }

}
