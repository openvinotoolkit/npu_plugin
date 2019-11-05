set -x
set -e
TESTNAME=$1
cp ${TESTNAME}/${TESTNAME}.in ${INFERENCE_MANAGER_DEMO_HOME}
cp ${TESTNAME}/${TESTNAME}.out ${INFERENCE_MANAGER_DEMO_HOME}
cd ../../build
cmake ..
make -j8
cd tests/layer
./${TESTNAME}
cd output
cp mcm.blob ${INFERENCE_MANAGER_DEMO_HOME}/${TESTNAME}.blob
