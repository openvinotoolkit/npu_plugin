#ifndef MV_HELPERS_HPP_
#define MV_HELPERS_HPP_

#include <fstream>

namespace mv {
namespace utils {

template <typename T, void (*F)(T*)>
struct RaiiWrapper {
    void operator()(T* obj_ptr) {
        if(obj_ptr) {
            F(obj_ptr);
        }
    }
};

void releaseFile(FILE* ptr);

inline size_t unaligned_load(const char* p);
size_t constatnt_string_hash(const std::string str);

}  // namespace utils
}  // namespace mv

#endif  // MV_HELPERS_HPP_
