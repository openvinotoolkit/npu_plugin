#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "VpualDispatcher" for configuration "Release"
set_property(TARGET VpualDispatcher APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(VpualDispatcher PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libVpualDispatcher.so"
  IMPORTED_SONAME_RELEASE "libVpualDispatcher.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS VpualDispatcher )
list(APPEND _IMPORT_CHECK_FILES_FOR_VpualDispatcher "${_IMPORT_PREFIX}/lib/libVpualDispatcher.so" )

# Import target "RemoteFlic" for configuration "Release"
set_property(TARGET RemoteFlic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(RemoteFlic PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libRemoteFlic.so"
  IMPORTED_SONAME_RELEASE "libRemoteFlic.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS RemoteFlic )
list(APPEND _IMPORT_CHECK_FILES_FOR_RemoteFlic "${_IMPORT_PREFIX}/lib/libRemoteFlic.so" )

# Import target "NN" for configuration "Release"
set_property(TARGET NN APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(NN PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libNN.so"
  IMPORTED_SONAME_RELEASE "libNN.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS NN )
list(APPEND _IMPORT_CHECK_FILES_FOR_NN "${_IMPORT_PREFIX}/lib/libNN.so" )

# Import target "OSD" for configuration "Release"
set_property(TARGET OSD APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OSD PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libOSD.so"
  IMPORTED_SONAME_RELEASE "libOSD.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS OSD )
list(APPEND _IMPORT_CHECK_FILES_FOR_OSD "${_IMPORT_PREFIX}/lib/libOSD.so" )

# Import target "sipp_custom" for configuration "Release"
set_property(TARGET sipp_custom APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sipp_custom PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsipp_custom.so"
  IMPORTED_SONAME_RELEASE "libsipp_custom.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS sipp_custom )
list(APPEND _IMPORT_CHECK_FILES_FOR_sipp_custom "${_IMPORT_PREFIX}/lib/libsipp_custom.so" )

# Import target "XLink" for configuration "Release"
set_property(TARGET XLink APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XLink PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libXLink.so"
  IMPORTED_SONAME_RELEASE "libXLink.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS XLink )
list(APPEND _IMPORT_CHECK_FILES_FOR_XLink "${_IMPORT_PREFIX}/lib/libXLink.so" )

# Import target "ResMgr" for configuration "Release"
set_property(TARGET ResMgr APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ResMgr PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libResMgr.so"
  IMPORTED_SONAME_RELEASE "libResMgr.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ResMgr )
list(APPEND _IMPORT_CHECK_FILES_FOR_ResMgr "${_IMPORT_PREFIX}/lib/libResMgr.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
