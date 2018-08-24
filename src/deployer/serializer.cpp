#include "include/mcm/deployer/serializer.hpp"
#include <stdio.h>

namespace mv
{
    Serializer::Serializer(serializer_mode set_output_format){
        output_format = set_output_format;
    }

    uint64_t Serializer::serialize(mv::ControlModel& graph_2_deploy, const char* ofilename )
    {

        printf("Serializer\n");

        uint64_t fsize = 0 ;
        switch( output_format )
        {
            case mvblob_mode:
                odata.calc(graph_2_deploy);
                odata.open(ofilename);
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