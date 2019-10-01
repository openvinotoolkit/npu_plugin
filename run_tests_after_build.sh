#!/bin/bash

# Inform user about purposes of this script
echo "This script will run some tests for kmb-plugin"
echo
read -ep "Would You like to continiue? [Yes/No] " -i "Yes" isContiniue

if [ "$isContiniue" != "Yes" ]
then
	echo "Finish to work"
	exit 0
else
	echo "Start to work"
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

# If Inference Engine is built with parameter -DENABLE_PLUGIN_RPATH=OFF then uncomment 5 lines below
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DLDT_HOME/bin/intel64/Release/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DLDT_HOME/inference-engine/temp/opencv_4.1.0_ubuntu18/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DLDT_HOME/inference-engine/temp/tbb/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KMB_PLUGIN_HOME/thirdparty/vsi_cmodel/vpusmm/x86_64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KMB_PLUGIN_HOME/thirdparty/movidius/mcmCompiler/build/lib

export MCM_HOME=$KMB_PLUGIN_HOME/thirdparty/movidius/mcmCompiler
cd $DLDT_HOME/bin/intel64/Release/
echo
echo " Run KmbBehaviorTests with --gtest_filter=*Behavior*orrectLib*kmb*"
echo
./KmbBehaviorTests --gtest_filter=*Behavior*orrectLib*kmb*
echo
echo " Run KmbFunctionalTests"
echo
./KmbFunctionalTests

echo "Script is finished. Check logs for errors"
