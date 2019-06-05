# kmb-plugin

KMBPlugin for Inference Engine


## How to build
* cd $DLDT_HOME
* mkdir $DLDT_HOME/inference-engine/build
* cd $DLDT_HOME/inference-engine/build
* cmake -DENABLE_TESTS=ON -DENABLE_BEH_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON ..
* make -j8
* cd $KMB_PLUGIN_HOME
* export MCM_HOME=$KMB_PLUGIN_HOME/thirdparty/movidius/mcmCompiler/src/mcmCompiler
* mkdir $KMB_PLUGIN_HOME/build
* cd $KMB_PLUGIN_HOME/build
* cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/inference-engine/build ..
* make -j8

## How run tests
* cd $DLDT_HOME/inference-engine/bin/intel64/Release/
* ./KmbBehaviorTests --gtest_filter=\*Behavior\*orrectLib\*kmb\*
* ./KmbFunctionalTests --gtest_filter=\*KmbParsingTest\*
