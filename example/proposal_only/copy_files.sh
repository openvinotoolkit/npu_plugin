set -e
set -x
TESTNAME=$1
cp ${TESTNAME}_in1.bin proposal.in
cp ${TESTNAME}_in2.bin proposal.in2
cp ${TESTNAME}_in3.bin proposal.in3
cp ${TESTNAME}_out.bin proposal.out
