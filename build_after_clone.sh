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

# Go to KMB-Plugin directory and instal some prerequisites for KMB-Plugin
cd $KMB_PLUGIN_HOME

# install Swig: 
sudo apt install swig
# install python3-dev: 
sudo apt install python3-dev
# install python-numpy: 
sudo apt install python-numpy
# install metis: 
sudo wget "http://nnt-srv01.inn.intel.com/builds/inference_engine/vpu_kmb_for_mcm/metis-5.1.0.tar.gz" -O "/tmp/metis-5.1.0.tar.gz" && cd /tmp && \
tar xzf /tmp/metis-5.1.0.tar.gz && (cd "/tmp/metis-5.1.0" && make config && make -j4 && make install) && rm -r /tmp/*



# Begin to make DLDT for KMB-Plugin
echo "Begin to make DLDT for KMB-Plugin"
cd $DLDT_HOME
mkdir $DLDT_HOME/inference-engine/build
cd $DLDT_HOME/inference-engine/build
cmake -DENABLE_TESTS=ON -DENABLE_BEH_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON ..
make -j8

# try to fix some bug with cmake in DLDT (it is necessary to delete targets_developer.cmake)
rm $DLDT_HOME/inference-engine/build/targets_developer.cmake
cd $DLDT_HOME/inference-engine/build/
cmake ..


# Begin to make KMB-Plugin
echo "Begin to make KMB-Plugin"
cd $KMB_PLUGIN_HOME
export MCM_HOME=$KMB_PLUGIN_HOME/thirdparty/movidius/mcmCompiler
git submodule update --init --recursive
mkdir $KMB_PLUGIN_HOME/build
cd $KMB_PLUGIN_HOME/build
cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/inference-engine/build ..
make -j8

echo "Work of script is finished. Check logs for errors."

exit 0

