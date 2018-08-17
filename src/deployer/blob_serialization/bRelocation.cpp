#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "include/mcm/deployer/blob_serialization/bRelocation.hpp"
#include "include/mcm/deployer/blob_serialization/blob_serializer.hpp"

namespace mv
{

    void RelocationTable::write(Blob_buffer* b)
    {
        printf("Write Reloc Table Now\n");

        auto blob_stats = b->getBlobSumm();

        b->AddBytes(4, blob_stats.relocation_section_size);

        uint32_t mv_stage_section_offset = blob_stats.elf_header_size+blob_stats.mv_header_size+blob_stats.header_pad_size ;
        uint32_t mv_buffer_section_offset = mv_stage_section_offset + blob_stats.stage_section_size ;
        uint32_t mv_relocation_offset = mv_buffer_section_offset + blob_stats.buffer_header_size + blob_stats.buffer_data_size ;


        b->AddBytes(4, mv_relocation_offset + 20);
        b->AddBytes(4, 8*this->constant_entries.size()); // blob_buffer_reloc_size
        b->AddBytes(4, mv_relocation_offset+20+8*this->constant_entries.size());  // Unknown - work_buffer_reloc_offset
        b->AddBytes(4, 8*this->variable_entries.size()); // work_buffer_reloc_size

        std::vector<std::pair<int, bLocation>>::iterator c_it;
        for(c_it = constant_entries.begin(); c_it != constant_entries.end(); c_it++ )    {
            b->AddBytes(4, c_it->first);
            b->AddBytes(4, (int)c_it->second);
        }

        std::vector<std::pair<int, bLocation>>::iterator v_it;
        for(v_it = variable_entries.begin(); v_it != variable_entries.end(); v_it++ )    {
            b->AddBytes(4, v_it->first);
            b->AddBytes(4, (int)v_it->second);
        }
    }

    unsigned int RelocationTable::push_entry(std::pair<int, bLocation> ol){
        /**
         *  Returns index of entry after pushed into relevant table.
         *
        */
        // printf("Push Reloc Table Entry\n");
        switch(ol.second){
            case bLocation::Input:
                this->input_entries.push_back(ol);
                return (unsigned int)this->input_entries.size() -1;
            case bLocation::Output:
                this->output_entries.push_back(ol);
                return (unsigned int)this->output_entries.size() -1;
            case bLocation::Constant:
                this->constant_entries.push_back(ol);
                return (unsigned int)this->constant_entries.size() -1;
            case bLocation::Variable:
                this->variable_entries.push_back(ol);
                return (unsigned int)this->variable_entries.size() -1;
        }
        printf("Relocation Table Entry cannot exist\n");
        assert(0);
    }
}
