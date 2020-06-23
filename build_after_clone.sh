#!/bin/bash

# Inform user about purposes of this script
echo "This script will make initial build of kmb-plugin after clonning from repository"
read -ep "Would You like to continiue? [Yes/No] " -i "Yes" isContiniue

if [ "$isContiniue" != "Yes" ]; then
	echo "Stop to build kmb-plugin"
	exit 0
else
	echo "Start to build kmb-plugin"
fi

echo "=============== Directory search ==============="
# Set base directories for DLDT and KMB-Plugin
# Set path to KMB-Plugin project
echo "Try to find base directories for dldt and kmb-plugin"
export KMB_PLUGIN_HOME=$(pwd)
echo "kmb-plugin base directory is: " $KMB_PLUGIN_HOME
cd ..

# Set path to DLDT project
# Try to find DLDT in parent directory
if [ -d "dldt" ]
then
	cd dldt
	export DLDT_HOME=$(pwd)
# If "dldt" directory is absent then ask user to set path to DLDT project
else
	echo "Default path to DLDT project is not found"
	read -p "Input path to DLDT project: " DLDT_HOME
fi
echo "dldt base directory is: " $DLDT_HOME

echo "=============== Install dependencies ==============="
# Go to KMB-Plugin directory and instal some prerequisites for KMB-Plugin
cd $KMB_PLUGIN_HOME
# install Swig:
sudo apt install swig
# install python3-dev:
sudo apt install python3-dev
# install python-numpy:
sudo apt install python-numpy
# install metis:
sudo apt install libmetis-dev libmetis5 metis
# boost  
sudo apt install libboost-all-dev

# Begin to make DLDT for KMB-Plugin
echo "=============== Build DLDT ==============="
export BUILD_DIR_NAME=build

echo "Begin to make DLDT for KMB-Plugin"
cd $DLDT_HOME
git submodule init
git submodule update --init --recursive
mkdir -p $DLDT_HOME/$BUILD_DIR_NAME
cd $DLDT_HOME/$BUILD_DIR_NAME
# It is necessary to set -DENABLE_PLUGIN_RPATH=ON because in script in /dld/inference-engine/build-after-clone.sh this parameter is set to OFF
# Path to libraries is necessary for properly work of test script (kmb-plugin/run_tests_after_build.sh)
cmake -DENABLE_TESTS=ON -DENABLE_BEH_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DENABLE_PLUGIN_RPATH=ON -DENABLE_CLDNN=OFF -DENABLE_DLIA=OFF -DENABLE_MKL_DNN=OFF -DENABLE_GNA=OFF -DCMAKE_BUILD_TYPE=Debug ..
make -j8

# try to fix some bug with cmake in DLDT (it is necessary to delete targets_developer.cmake)
rm $DLDT_HOME/$BUILD_DIR_NAME/targets_developer.cmake
cd $DLDT_HOME/$BUILD_DIR_NAME/
cmake ..


echo "=============== Build KMB-Plugin ==============="
# Begin to make KMB-Plugin
echo "Begin to make KMB-Plugin"
cd $KMB_PLUGIN_HOME
export MCM_HOME=$KMB_PLUGIN_HOME/thirdparty/movidius/mcmCompiler
git submodule update --init --recursive
mkdir -p $KMB_PLUGIN_HOME/$BUILD_DIR_NAME
cd $KMB_PLUGIN_HOME/$BUILD_DIR_NAME
cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build ..
make -j8

echo "Work of script is finished. Check logs for errors."

exit 0
