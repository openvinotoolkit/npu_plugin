/*
* {% copyright %}
*/
#include <graphfile_generated.h>
#include <sw_shave_dispatcher.h>

namespace nn {
namespace shave_lib {
std::shared_ptr<SWShaveDispatcher> SWShaveDispatcher::getInstance() {
    static std::shared_ptr<SWShaveDispatcher> holder(new (memory::cache_aligned) SWShaveDispatcher, memory::cache_aligned_deleter<SWShaveDispatcher>());
    return holder;
};

SWShaveDispatcher::SWShaveDispatcher() {}
SWShaveDispatcher::~SWShaveDispatcher() {}

void SWShaveDispatcher::initSWShaveDispatcher() {}
void SWShaveDispatcher::terminateSWShaveDispatcher() {}
bool SWShaveDispatcher::resizeShavePool(unsigned int) { return false; };
bool SWShaveDispatcher::hasResources() const { return true; };
unsigned char SWShaveDispatcher::getControllerShaveID() const { return 0; };
void SWShaveDispatcher::flushShaveL2DataCache() {}
void SWShaveDispatcher::flushShaveL2InstructionCache() {}

} // namespace shave_lib
} // namespace nn
