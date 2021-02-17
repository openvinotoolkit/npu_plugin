#include "include/mcm/utils/helpers.hpp"

void mv::utils::releaseFile(FILE* ptr) {
    if(ptr) {
        fclose(ptr);
    }
}
