#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bRelocation.hpp"
#include <string.h>
#include <assert.h>

namespace mv
{
    void RelocationTable::write(WBuffer* b)
    {
        printf("Write Reloc Table Now\n");

        b->AddBytes(4, 20); // Section Header Size HARDCODED :(
        b->AddBytes(4, 42);  // Unknown - blob_buffer_reloc_offset
        b->AddBytes(4, 8*this->constant_entries.size()); // blob_buffer_reloc_size
        b->AddBytes(4, 42);  // Unknown - work_buffer_reloc_offset
        b->AddBytes(4, 8*this->variable_entries.size()); // work_buffer_reloc_size

        std::vector<std::pair<int, bLocation>>::iterator c_it;
        for(c_it = constant_entries.begin(); c_it != constant_entries.end(); c_it++ )    {
            b->AddBytes(4, c_it->first);
            b->AddBytes(4, (int)c_it->second);
        }

        std::vector<std::pair<int, bLocation>>::iterator v_it;
        for(v_it = constant_entries.begin(); v_it != constant_entries.end(); v_it++ )    {
            b->AddBytes(4, v_it->first);
            b->AddBytes(4, (int)v_it->second);
        }


    }

    unsigned int RelocationTable::push_entry(std::pair<int, bLocation> ol){
        /**
         *  Returns index of entry after pushed into relevant table.
         *
        */
        printf("Push Reloc Table Entry\n");
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
