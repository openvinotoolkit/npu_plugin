SUMMARY = "NN BLOBS FOR NN/VPUAL TEST APPS"
DESCRIPTION = "Adding the 7 NN Blobs required by NN/VPUAL test apps to 'opt/mobilenet' \
and 'opt/resnet' and 'opt/yolotiny' and 'opt/googlenet' and 'opt/inceptionv3' \
and 'opt/squeezenet' and 'opt/fullyolov2'."
LICENSE = "CLOSED"

RDEPENDS_${PN} = "kernel-module-udmabuf openvino"

SRC_URI = "file://${NN_BLOBS_PATH}/nn-blobs.tar.bz2"

PV="nn_blobs-4734f4c-NN_Compiler_v1.2.13"

S = "${WORKDIR}/nn-blobs"

SOLIBS = ".so"
FILES_SOLIBSDEV = ""

INSANE_SKIP_${PN} = "ldflags"

do_install () {
    install -d ${D}/opt/mobilenet
    install -m 0755 ${S}/mobilenet/expected_result_sim.dat ${D}/opt/mobilenet
    install -m 0755 ${S}/mobilenet/expected_result_tflite.dat ${D}/opt/mobilenet
    install -m 0755 ${S}/mobilenet/input.dat ${D}/opt/mobilenet
    install -m 0755 ${S}/mobilenet/mobilenet.blob ${D}/opt/mobilenet
    install -d ${D}/opt/resnet
    install -m 0755 ${S}/resnet/expected_result_sim.dat ${D}/opt/resnet
    install -m 0755 ${S}/resnet/expected_result_tflite.dat ${D}/opt/resnet
    install -m 0755 ${S}/resnet/input.dat ${D}/opt/resnet
    install -m 0755 ${S}/resnet/resnet.blob ${D}/opt/resnet
    install -d ${D}/opt/yolotiny
    install -m 0755 ${S}/yolotiny/expected_result_sim.dat ${D}/opt/yolotiny
    install -m 0755 ${S}/yolotiny/expected_result_tflite.dat ${D}/opt/yolotiny
    install -m 0755 ${S}/yolotiny/input.dat ${D}/opt/yolotiny
    install -m 0755 ${S}/yolotiny/yolotiny.blob ${D}/opt/yolotiny
    install -d ${D}/opt/googlenet
    install -m 0755 ${S}/googlenet/expected_result_sim.dat ${D}/opt/googlenet
    install -m 0755 ${S}/googlenet/expected_result_tflite.dat ${D}/opt/googlenet
    install -m 0755 ${S}/googlenet/input.dat ${D}/opt/googlenet
    install -m 0755 ${S}/googlenet/googlenet.blob ${D}/opt/googlenet
    install -d ${D}/opt/inceptionv3
    install -m 0755 ${S}/inceptionv3/expected_result_sim.dat ${D}/opt/inceptionv3
    install -m 0755 ${S}/inceptionv3/expected_result_tflite.dat ${D}/opt/inceptionv3
    install -m 0755 ${S}/inceptionv3/input.dat ${D}/opt/inceptionv3
    install -m 0755 ${S}/inceptionv3/inceptionv3.blob ${D}/opt/inceptionv3
    install -d ${D}/opt/squeezenet
    install -m 0755 ${S}/squeezenet/expected_result_sim.dat ${D}/opt/squeezenet
    install -m 0755 ${S}/squeezenet/expected_result_tflite.dat ${D}/opt/squeezenet
    install -m 0755 ${S}/squeezenet/input.dat ${D}/opt/squeezenet
    install -m 0755 ${S}/squeezenet/squeezenet.blob ${D}/opt/squeezenet
    install -d ${D}/opt/fullyolov2
    install -m 0755 ${S}/fullyolov2/expected_result_sim.dat ${D}/opt/fullyolov2
    install -m 0755 ${S}/fullyolov2/expected_result_tflite.dat ${D}/opt/fullyolov2
    install -m 0755 ${S}/fullyolov2/input.dat ${D}/opt/fullyolov2
    install -m 0755 ${S}/fullyolov2/fullyolov2.blob ${D}/opt/fullyolov2
}

FILES_${PN} += " /opt"
