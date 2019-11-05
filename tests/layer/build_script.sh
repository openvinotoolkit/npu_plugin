set -x
set -e
TESTNAME=$1
cp ${TESTNAME}_only/${TESTNAME}.in ${INFERENCE_MANAGER_DEMO_HOME}
cp ${TESTNAME}_only/${TESTNAME}.out ${INFERENCE_MANAGER_DEMO_HOME}
cd ../build
cmake ..
make -j8
cd example
./${TESTNAME}_only
cd output
cp mcm.blob ${INFERENCE_MANAGER_DEMO_HOME}/${TESTNAME}.blob
