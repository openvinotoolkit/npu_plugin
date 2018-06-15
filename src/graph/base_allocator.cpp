#include "include/mcm/graph/base_allocator.hpp"
#include <iostream>

mv::base_allocator::callback mv::base_allocator::alloc_fail_callback = &mv::base_allocator::default_alloc_fail;

void mv::base_allocator::default_alloc_fail(int err, char *msg, unsigned len)
{
    std::cerr << "Error " << err << ": " << std::string(msg, msg + len) << std::endl;
    exit(EXIT_FAILURE);
}

mv::base_allocator::~base_allocator()
{
    
}