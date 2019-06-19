set(XLINK_BASE ${vpual_home}/host/source/XLink)

set(XLINK_INCLUDE_DIRECTORIES
    ${XLINK_BASE}/include
    ${XLINK_BASE}/pc
    ${XLINK_BASE}/shared
    ${XLINK_BASE}/swCommon/include
    ${XLINK_BASE}/sshared/include
    ${XLINK_BASE}/pcUsbTool/common/vsc
    ${XLINK_BASE}/pcUsbTool/linux/uvc
    ${XLINK_BASE}/pcUsbTool
    /usr/include/libusb-1.0
    ${XLINK_BASE}/swCommon/pcModel/half
)

set(XLINK_SOURCES
    ${XLINK_BASE}/src/XLinkUAPI.c
)

set(XLINK_PUBLIC_HEADERS
    ${XLINK_BASE}/shared/XLink.h
    ${XLINK_BASE}/include/XLinkUAPI.h
    ${XLINK_BASE}/shared/XLinkPublicDefines.h
)

add_library(XLink2 SHARED
    ${XLINK_SOURCES}
)

target_link_libraries(XLink2 pthread usb-1.0)
