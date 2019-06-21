###########################################################
#                  Find Lemon Library
# https://lemon.cs.elte.hu/trac/lemon
#----------------------------------------------------------

FIND_PATH(LEMON_DIR list_graph.h
    HINTS "${LEMON_ROOT}" "$ENV{LEMON_ROOT}" "${LEMON_INCLUDE_DIR_HINTS}"
    PATHS "$ENV{PROGRAMFILES}/lemon" "$ENV{PROGRAMW6432}/lemon"
    PATH_SUFFIXES lemon
    DOC "Root directory of LEMON includes")

##====================================================
## Include LEMON library
##----------------------------------------------------
IF(EXISTS "${LEMON_DIR}" AND NOT "${LEMON_DIR}" STREQUAL "")
  SET(LEMON_FOUND TRUE)
  # Remove /lemon from path (math.h cannot be exposed all time)
  GET_FILENAME_COMPONENT(LEMON_INCLUDE_DIRS "${LEMON_DIR}" PATH)
  SET(LEMON_DIR "${LEMON_DIR}" CACHE PATH "" FORCE)
  MARK_AS_ADVANCED(LEMON_DIR)
  # Extract Lemon version from config.h
  SET(LEMON_VERSION_FILE ${LEMON_INCLUDE_DIRS}/lemon/config.h)
  IF (NOT EXISTS ${LEMON_VERSION_FILE})
    IF (Lemon_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR
        "Could not find file: ${LEMON_VERSION_FILE} "
        "containing version information in Lemon install located at: "
        "${LEMON_INCLUDE_DIRS}.")
    ELSEIF(NOT Lemon_FIND_QUIETLY)
      MESSAGE(SEND_ERROR
      "Could not find file: ${LEMON_VERSION_FILE} "
      "containing version information in Lemon install located at: "
      "${LEMON_INCLUDE_DIRS}.")
    ENDIF()
  ELSE (NOT EXISTS ${LEMON_VERSION_FILE})
    FILE(READ ${LEMON_VERSION_FILE} LEMON_VERSION_FILE_CONTENTS)
    STRING(REGEX MATCH "#define LEMON_VERSION \"([0-9.]+)\""
    LEMON_VERSION "${LEMON_VERSION_FILE_CONTENTS}")
    STRING(REGEX REPLACE "#define LEMON_VERSION \"([0-9.]+)\"" "\\1"
    LEMON_VERSION "${LEMON_VERSION}")
  ENDIF (NOT EXISTS ${LEMON_VERSION_FILE})
  
  # SET(LEMON_INCLUDE_DIR ${LEMON_DIR})

  FIND_LIBRARY(LEMON_LIBRARY NAMES Lemon lemon emon libemon)

  # locate Lemon libraries
  IF(DEFINED LEMON_LIBRARY)
    SET(LEMON_LIBRARIES ${LEMON_LIBRARY})
  ENDIF()

  MESSAGE(STATUS "Found Lemon ${LEMON_VERSION} (include: ${LEMON_DIR})")
ELSE()

  find_package(Hg)
  if(NOT HG_FOUND)
    message( FATAL_ERROR "Mercurial client requred (hg) to download Lemon sources. Exiting...")
  endif()

  include(ExternalProject)
  set(LEMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/lemon)
  ExternalProject_Add(
      lemon
      HG_REPOSITORY     http://lemon.cs.elte.hu/hg/lemon 
      HG_TAG            "r1.3.1"
      SOURCE_DIR        ${LEMON_SOURCE_DIR}
      BUILD_COMMAND     ${MAKE}
      INSTALL_COMMAND   make install
      VERBATIM
      #CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
  )
  # install(DIRECTORY ${sphinxbase_DESTDIR}/ DESTINATION "/")

  # MESSAGE(FATAL_ERROR "You are attempting to build without Lemon. "
  #         "Please use cmake variable -DLEMON_INCLUDE_DIR_HINTS:STRING=\"PATH\" "
  #         "or LEMON_INCLUDE_DIR_HINTS env. variable to a valid Lemon path. "
  #         "Or install last Lemon version.")
  package_report_not_found(LEMON "Lemon cannot be found")
ENDIF()
##====================================================

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# https://stackoverflow.com/questions/35934112/installing-an-externalproject-with-cmake
# It is better to use binary directory for download or build 3d-party project 
# set(sphinxbase_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib/sphinxbase)
# # This will be used as DESTDIR on subproject's `make install`.
# set(sphinxbase_DESTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib/sphinxbase_install)

# ExternalProject_Add(
#     sphinxbase
#     GIT_REPOSITORY      "https://github.com/cmusphinx/sphinxbase.git"
#     GIT_TAG             "e34b1c632392276101ed16e8a05862e43f038a7c"
#     SOURCE_DIR          ${sphinxbase_SOURCE_DIR}
#     # Specify installation prefix for configure.sh (autogen.sh).
#     CONFIGURE_COMMAND   ./autogen.sh --prefix=${CMAKE_INSTALL_PREFIX}
#     BUILD_COMMAND       ${MAKE}
#     UPDATE_COMMAND      ""
#     # Fake installation: copy installed files into DESTDIR.
#     INSTALL_COMMAND     make DESTDIR=${sphinxbase_DESTDIR} install
#     ...
# )
# Actually install subproject.
# install(DIRECTORY ${sphinxbase_DESTDIR}/ DESTINATION "/")
