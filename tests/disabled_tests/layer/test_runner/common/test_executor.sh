KMB_IP=$1

if [ -z $KMB_IP ]; then
	KMB_IP="localhost"
	echo "KMB IP was not provided. Using localhost as default."
fi

if [[ -z "${INFERENCE_MANAGER_DEMO_HOME}" ]]; then
    echo "Environment variable INFERENCE_MANAGER_DEMO_HOME is not defined"
    exit 1
fi

echo Creating soft links to IM folders
ln -sf "${INFERENCE_MANAGER_DEMO_HOME}/leon" "./"
ln -sf "${INFERENCE_MANAGER_DEMO_HOME}/leon_nn" "./"

echo Building sample app
make all -j8

if (($? != 0)); then
	echo "Build failed!"
	exit $?
fi

checkOutputCRC() {
	OUTPUTS=$1
	LOG=$2
	for ((i=0; i<${#OUTPUTS[@]}; ++i))
	do
		CRC=$(crc32 "${OUTPUTS[i]}")
		echo -e "CRC(${OUTPUTS[i]}): \e[93m$CRC\e[39m"
		if ! grep -q "$CRC" "$LOG"; then
			# If CRC check fails, try file comparison. Sometimes the CRC from the debug log is not correct
			if ! diff output-${i}.bin "${OUTPUTS[i]}"; then
				return 0
			fi
		fi
	done
	return 1
}

CHECK_FUNC=${CHECK_FUNC-checkOutputCRC}

rm -f test_*.log
for ((t=0; t<${#TESTS[@]}; ++t)); do
	rm -f input-?.bin output-?.bin test.blob

	target=${TESTS[t]}
	blob="blobs/${target}.blob"

	if ! [ -e $blob ]; then
		echo "Could not find ${blob}. Skipping the test."
		continue
	fi

	input_count=0
	while [ -e "tests/${target}/in${input_count}.bin" ]; do
	  ln -sf "tests/${target}/in${input_count}.bin" input-${input_count}.bin
	  ((input_count++))
	done

	output_count=0
	while [ -e "tests/${target}/out${output_count}.bin" ]; do
	  outputs[output_count]="tests/${target}/out${output_count}.bin"
	  ((output_count++))
	done

	echo -e "\e[1;33mRunning test #$((t+1)): [in:$input_count out:$output_count] $blob \e[0m"

	ln -s $blob test.blob
	LOG_FILE=test_$t.log
	echo Saving execution log to $LOG_FILE
	make run srvIP="$KMB_IP" 2>&1 | tee $LOG_FILE
	if $CHECK_FUNC "${outputs[@]}" $LOG_FILE; then
		echo -e "\e[91mOutput mismatch!\e[0m" | tee -a $LOG_FILE
	else
		echo -e "\e[32mTest passed!\n\e[0m" | tee -a $LOG_FILE
	fi
done
