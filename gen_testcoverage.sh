#!/bin/bash

# Install any required packages
if [ $(dpkg-query -W -f='${Status}' wmctrl 2>/dev/null | grep -c "ok installed") -eq 0 ] || 
   [ $(dpkg-query -W -f='${Status}' lcov 2>/dev/null | grep -c "ok installed") -eq 0 ];
then
    sudo apt-get install -y lcov wmctrl; # package brings specified window to the front
fi

# change to build dir if exists, and remove any previous build files
if ! [ -d "build" ] 
then
    mkdir build
fi
cd build
rm -rf *

# build libraries + run unit tests
cmake -DCODE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug ..
make -j8
./tests/mcm_unit_tests

# generate line coverage report. Filter out non-relevant projects
lcov --directory . --capture --output-file coverage.info
lcov --remove coverage.info '/usr/include/*' -o coverage_filtered.info
eval lcov --remove coverage_filtered.info '${MCM_HOME}/tests/*' -o coverage_filtered.info
eval lcov --remove coverage_filtered.info \
        '${MCM_HOME}/contrib/flatbuffers/grpc/src/compiler/*' \
        '${MCM_HOME}/contrib/flatbuffers/grpc/src/compiler/*' \
        '${MCM_HOME}/contrib/flatbuffers/include/flatbuffers/*' \
        '${MCM_HOME}/contrib/flatbuffers/src/*' \
        '${MCM_HOME}/contrib/googletest/googletest/include/gtest/*' \
        '${MCM_HOME}/contrib/googletest/googletest/include/gtest/internal/*' \
        '${MCM_HOME}/contrib/googletest/googletest/src/*' \
        '${MCM_HOME}/tests/base/*' \
        '${MCM_HOME}/tests/compiler/*' \
        '${MCM_HOME}/tests/graph/*' \
        '${MCM_HOME}/tests/model/*' \
        '${MCM_HOME}/tests/op/*' \
        '${MCM_HOME}/tests/pass/*' \
        '${MCM_HOME}/tests/resources/*' \
        '${MCM_HOME}/tests/target/kmb/*' \
        '${MCM_HOME}/tests/tensor/*' \
        '${MCM_HOME}/contrib/koala/base/*' \
        '${MCM_HOME}/contrib/koala/coloring/*' \
        '${MCM_HOME}/contrib/koala/container/*' \
        '${MCM_HOME}/contrib/koala/graph/*' \
        '${MCM_HOME}/contrib/koala/io/*' \
        '${MCM_HOME}/contrib/koala/tinyxml/*' \
    -o coverage_filtered.info

genhtml coverage_filtered.info --ignore-errors source --legend --title "MCM Compiler TestCoverage" --output-directory=./ccov

#display report
xdg-open ./ccov/index.html
wmctrl -a "MCM Compiler TestCoverage" # bring to front
