set(REMOTE_FLIC_BASE ${vpual_home}/host/source/FLIC)
set(REMOTE_NN_BASE ${vpual_home}/host/source/NN)

include(${CMAKE_CURRENT_LIST_DIR}/Dispatcher.cmake)

set(REMOTE_FLIC_INCLUDE_DIRECTORIES
    ${REMOTE_FLIC_BASE}/top/include
    ${REMOTE_FLIC_BASE}/mvSipp/include
    ${REMOTE_FLIC_BASE}/PlgXlink
    ${REMOTE_FLIC_BASE}/PlgXlink/include
    ${REMOTE_FLIC_BASE}/PlgQuantize
    ${REMOTE_FLIC_BASE}/PlgQuantize/include
    ${REMOTE_FLIC_BASE}/PlgCropNV12
    ${REMOTE_FLIC_BASE}/PlgCropNV12/include
    ${REMOTE_FLIC_BASE}/NNFlicPlg
    ${REMOTE_NN_BASE}/NNFlicPlg/include
    ${REMOTE_FLIC_BASE}/TensorPlg
    ${REMOTE_FLIC_BASE}/GraphManagerPlg
    ${REMOTE_NN_BASE}/GraphManagerPlg/include
    ${REMOTE_NN_BASE}/common/include
    ${REMOTE_NN_BASE}/TensorPlg/include
    ${VPUAL_DISPATCHER_INCLUDE_DIRECTORIES}
)

set(REMOTE_FLIC_SOURCES
    ${REMOTE_FLIC_BASE}/mvSipp/source/PlgMvSippStub.cpp
    ${REMOTE_FLIC_BASE}/PlgXlink/PlgXlinkIn.cpp
    ${REMOTE_FLIC_BASE}/PlgXlink/PlgXlinkOut.cpp
    ${REMOTE_FLIC_BASE}/PlgQuantize/PlgQuantize.cpp
    ${REMOTE_FLIC_BASE}/PlgCropNV12/PlgCropNV12.cpp
    ${REMOTE_NN_BASE}/GraphManagerPlg/GraphManagerPlg.cpp
    ${REMOTE_NN_BASE}/NNFlicPlg/NNFlicPlg.cpp
    ${REMOTE_NN_BASE}/TensorPlg/PlgTensorSource.cpp
    ${REMOTE_NN_BASE}/TensorPlg/PlgStreamResult.cpp

    ${REMOTE_FLIC_BASE}/top/source/Allocator.cpp
    ${REMOTE_FLIC_BASE}/top/source/Flic.cpp
    ${REMOTE_FLIC_BASE}/top/source/Message.cpp
    ${REMOTE_FLIC_BASE}/top/source/Pool.cpp
)

set(REMOTE_FLIC_PUBLIC_HEADERS
    ${REMOTE_FLIC_BASE}/mvSipp/include/PlgMvSipp.h
    ${REMOTE_FLIC_BASE}/PlgXlink/PlgXlinkIn.h
    ${REMOTE_FLIC_BASE}/PlgXlink/PlgXlinkOut.h
    ${REMOTE_FLIC_BASE}/PlgQuantize/PlgQuantize.h
    ${REMOTE_FLIC_BASE}/PlgCropNV12/PlgCropNV12.h
    ${REMOTE_FLIC_BASE}/GraphManagerPlg/GraphManagerPlg.h
    ${REMOTE_FLIC_BASE}/NNFlicPlg/NNFlicPlg.h
    ${REMOTE_FLIC_BASE}/TensorPlg/PlgStreamResult.h
    ${REMOTE_FLIC_BASE}/TensorPlg/PlgTensorSource.h

    ${REMOTE_FLIC_BASE}/top/include/Flic.h
    ${REMOTE_FLIC_BASE}/top/include/MemAllocator.h
    ${REMOTE_FLIC_BASE}/top/include/Message.h
    ${REMOTE_FLIC_BASE}/top/include/Pool.h
    ${REMOTE_FLIC_BASE}/top/include/PoolObj.h
)

add_library(RemoteFlic SHARED
    ${REMOTE_FLIC_SOURCES}
)

target_link_libraries(RemoteFlic VpualDispatcher)
