#ifndef RUN_MOVISIM_EMULATOR_HPP
#define RUN_MOVISIM_EMULATOR_HPP

#ifdef ENABLE_MOVISIM

#include <inference_engine.hpp>

namespace ms {

InferenceEngine::BlobMap runMoviSimEmulator(InferenceEngine::ExecutableNetwork& exeNet,
                                            std::string pathToNetworkBlob, const std::vector<std::string>& dumpedInputsPaths);

}  // namespace ms
#endif // ENABLE_MOVISIM

#endif // RUN_MOVISIM_EMULATOR_HPP
