#!/bin/bash

# input params 
#     no params will run all steps
#     tflite - only generates the tflite models
#     comp - only runs compiler step
#     app - only runs blobs through simulator to gen actual results

# set -x # outputs all to screen - for debug

# set required variables
TEST_DIR=~/mdk/testApps/kmb/tapeout_so/tc_kmb_ma2490/ts_nce_bm_blob
TMP_DIR="$TEST_DIR/tmp"
MDK_ROOT=~/mdk
TFLITE_GEN_DIR=~/git/migNetworkZoo/test/kmb_test_generation
TFLITE_MODEL_DIR=~/git/migNetworkZoo/internal/unit_tests/CompilerTestsKmb/layers
FATHOM_DIR="$MDK_ROOT/projects/Fathom/src2"
MV_TOOLS_VERSION="Latest_195458"
MV_TOOLS_DIR="$MDK_ROOT/tools"

#stub directories
TEST_SUITE="$TEST_DIR/tests.csv"
COMP_LOG="$TMP_DIR/compiler.log"
APP_LOG="$TMP_DIR/app.log"
PERF_RES="$TMP_DIR/perf_res.csv"
COVERAGE_RES="$TMP_DIR/coverage.csv"
TEST_CASE_ROOT="${TEST_DIR}/mcm"
TEMPLATE_CASE_DIR="${TEST_CASE_ROOT}/templateTest"

if [ ! -d "$TMP_DIR" ]; then
    mkdir "$TMP_DIR"
fi

# get test ID's, skip csv header
test_names=($( awk -F',' '{print $2}' "$TEST_SUITE" | tail -n +2))

proc=0
if [ "$#" -eq 0 ] ; then   # no params supplied
    proc=1
fi

pipe_start="$1"
echo "$pipe_start"
QUANT_EN=0 # quantized networks not supported yet

#generate all TFLITE models
if [[ "$pipe_start" == "tflite" ]] || [[ $proc -ne 0 ]]  ; then

    #generate all TFLITE models
    pushd "$TFLITE_GEN_DIR" || exit
    cp "$TEST_SUITE" "$TFLITE_GEN_DIR"

    SIMPLIFY=""
    if [[ "${QUANT_EN}" -eq 0 ]] ; then
        SIMPLIFY="--simplify"
    fi

    python3 run_testfile.py ${SIMPLIFY}
    popd || exit
fi

# start up the movisim server simulator
pkill -f -9 moviSim ; sleep 1 ; pkill -f -9 moviSim
nohup ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION}/linux64/bin/moviSim  -cv:ma2x9x -leon -tcpip:30001 -q &
echo $! > "${TEST_DIR}/save_pid_of_nohup.txt"

#iterate over tests from suite
for name in "${test_names[@]}"
do
    EXP_CRC=0
    if [[ "${QUANT_EN}" -eq 0 ]] ; then
        tf_model="$TFLITE_MODEL_DIR/$name/simplified_model.tflite"
    else
	    tf_model="$TFLITE_MODEL_DIR/$name/quantized_model.tflite"
    fi 
    echo "$tf_model"
    
    TENSOR_TMP_DIR="${TMP_DIR}/${name}"
    
    if [[ "$pipe_start" == "comp" ]] || [[ $proc -ne 0 ]]  ; then
        echo "RUNNING COMPILER"
        pushd "$FATHOM_DIR" 1>>/dev/null || exit
        
        if ! [ -d "output" ] ; then 
            rm output/* 
        fi
        DEBUG_INPUT_FILE="${TEST_DIR}/input_dat.npy" 
        # run PoC Compiler with --emulator to generate input.dat, expected_result_sim.dat, blob will be ignored
        python3 Fathom.py generate --image Debug --network-description "$tf_model" \
            --kmb --nDPU 1 --nClusters 1 --cmx 4096 --strategy Clustering --emulator --verbose #> "$COMP_LOG.$name"

        #save compiler results to tmp folder
        mkdir -p "${TENSOR_TMP_DIR}"
        cp output/* "${TENSOR_TMP_DIR}/"

        # run PoC Compiler as a frontend to MCM - we want this blob!
        python3 Fathom.py generate --image Debug --network-description "$tf_model" \
            --kmb --nDPU 1 --nClusters 1 --cmx 4096 --strategy Clustering --cpp --comp-descriptor $MCM_HOME/config/compilation/debug_ma2490.json --verbose > "$COMP_LOG.$name"
        cp blob.bin "${TENSOR_TMP_DIR}/"
         
        TEST_CASE_DIR="${TEST_CASE_ROOT}/test_${name}"
        mkdir -p "${TEST_CASE_DIR}"
        cp "${TENSOR_TMP_DIR}/input.dat" "${TEST_CASE_DIR}/"
        cp "${TENSOR_TMP_DIR}/vpu2.blob" "${TEST_CASE_DIR}/"
        cp "${TENSOR_TMP_DIR}/blob.bin" "${TEST_CASE_DIR}/mcm.blob"
        cp "${TENSOR_TMP_DIR}/expected_result_sim.dat" "${TEST_CASE_DIR}/"
            
        EXP_RESULT="${TEST_CASE_DIR}/expected_result_sim.dat"
        EXP_CRC=$( crc32 "${EXP_RESULT}" )

        cp "${TEMPLATE_CASE_DIR}/Makefile" "${TEST_CASE_DIR}/Makefile"
        sed -i "s/\(.*EXPECTED_CRC\ *=\ * 0x\)[0-9a-zA-Z]*\(.*\)/\1${EXP_CRC}\2/g" "${TEST_CASE_DIR}/Makefile"
        sed -i 's/vpu2.blob/mcm.blob/g' "${TEST_CASE_DIR}/Makefile"

        popd 1>>/dev/null || exit
    fi
    
    if [[ "$pipe_start" == "app" ]] || [[ $proc -ne 0 ]]  ; then
        echo "RUNNING APPLICATION"
        pushd "${TEST_CASE_DIR}" 1>>/dev/null || exit
        rm -rf ./output
        make all -j && make run -j #> "$APP_LOG.$name"
        #make run MV_SOC_REV=ma2490 GRAPH_BIN=vpu2.blob GRAPH_BIN_PATH=."
        #make all -j && make debug srvIP=iirfpga020.ir.intel.com  -j
        
        STATE="fail" 
        OUTPUT_FILE="${TEST_CASE_DIR}/NCE2Task_network_out.bin"
        if [ -f "$OUTPUT_FILE" ] ; then
            ACT_CRC=$(crc32 "${OUTPUT_FILE}")
            echo "ACT_CRC $ACT_CRC"
            echo "EXP_CRC $EXP_CRC"
            if [ "$ACT_CRC" = "$EXP_CRC" ] ; then
                    STATE="pass"
            fi
        fi

        pushd "$FATHOM_DIR" 1>>/dev/null || exit
        python3 Validate.py --reference "$OUTPUT_FILE" --testdata "$EXP_RESULT" --dtype u8
        if [ "$STATE" == "pass" ] ; then
            echo "$name, PASS" >> "$COVERAGE_RES"
        else
            echo "$name, FAIL " >> "$COVERAGE_RES"
        fi
    fi
    break
done

# Test Teardown
kill -9 `cat "${TEST_DIR}/save_pid_of_nohup.txt"`
rm "${TEST_DIR}/save_pid_of_nohup.txt"
[ -f ~/nohup.out ] && cat ~/nohup.out
