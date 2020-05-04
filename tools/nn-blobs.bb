SUMMARY = "NN BLOBS FOR NN/VPUAL TEST APPS"
DESCRIPTION = "Adding the 7 NN Blobs required by NN/VPUAL test apps to 'opt/mobilenet' \
and 'opt/resnet' and 'opt/yolotiny' and 'opt/googlenet' and 'opt/inceptionv3' \
and 'opt/squeezenet' and 'opt/fullyolov2'."
LICENSE = "CLOSED"

RDEPENDS_${PN} = "kernel-module-udmabuf openvino"

SRC_URI = "file://${NN_BLOBS_PATH}/nn-blobs.tar.bz2"

PV = "nn_blobs_75e0804+NN_Compiler_v1.3.4"

S = "${WORKDIR}/release_blobs"

SOLIBS = ".so"
FILES_SOLIBSDEV = ""

INSANE_SKIP_${PN} = "ldflags"

do_install () {
    install -d ${D}/opt/mobilenet
    install -m 0755 ${S}/mobilenet/expected_result_sim.dat ${D}/opt/mobilenet
    install -m 0755 ${S}/mobilenet/input.dat ${D}/opt/mobilenet
    install -m 0755 ${S}/mobilenet/mobilenet.blob ${D}/opt/mobilenet
    install -d ${D}/opt/resnet
    install -m 0755 ${S}/resnet/expected_result_sim.dat ${D}/opt/resnet
    install -m 0755 ${S}/resnet/input.dat ${D}/opt/resnet
    install -m 0755 ${S}/resnet/resnet.blob ${D}/opt/resnet
    install -d ${D}/opt/yolotiny
    install -m 0755 ${S}/yolotiny/expected_result_sim.dat ${D}/opt/yolotiny
    install -m 0755 ${S}/yolotiny/input.dat ${D}/opt/yolotiny
    install -m 0755 ${S}/yolotiny/yolotiny.blob ${D}/opt/yolotiny
    install -d ${D}/opt/googlenet
    install -m 0755 ${S}/googlenet/expected_result_sim.dat ${D}/opt/googlenet
    install -m 0755 ${S}/googlenet/input.dat ${D}/opt/googlenet
    install -m 0755 ${S}/googlenet/googlenet.blob ${D}/opt/googlenet
    install -d ${D}/opt/inceptionv3
    install -m 0755 ${S}/inceptionv3/expected_result_sim.dat ${D}/opt/inceptionv3
    install -m 0755 ${S}/inceptionv3/input.dat ${D}/opt/inceptionv3
    install -m 0755 ${S}/inceptionv3/inceptionv3.blob ${D}/opt/inceptionv3
    install -d ${D}/opt/squeezenet
    install -m 0755 ${S}/squeezenet/expected_result_sim.dat ${D}/opt/squeezenet
    install -m 0755 ${S}/squeezenet/input.dat ${D}/opt/squeezenet
    install -m 0755 ${S}/squeezenet/squeezenet.blob ${D}/opt/squeezenet
    install -d ${D}/opt/fullyolov2
    install -m 0755 ${S}/fullyolov2/expected_result_sim.dat ${D}/opt/fullyolov2
    install -m 0755 ${S}/fullyolov2/input.dat ${D}/opt/fullyolov2
    install -m 0755 ${S}/fullyolov2/fullyolov2.blob ${D}/opt/fullyolov2

    install -d ${D}/opt/mobilenet_ov
    install -m 0755 ${S}/mobilenet_ov/expected_result_sim_ov.dat ${D}/opt/mobilenet_ov
    install -m 0755 ${S}/mobilenet_ov/input_ov.dat ${D}/opt/mobilenet_ov
    install -m 0755 ${S}/mobilenet_ov/mobilenet.blob ${D}/opt/mobilenet_ov
    install -d ${D}/opt/resnet_ov
    install -m 0755 ${S}/resnet_ov/expected_result_sim_ov.dat ${D}/opt/resnet_ov
    install -m 0755 ${S}/resnet_ov/input_ov.dat ${D}/opt/resnet_ov
    install -m 0755 ${S}/resnet_ov/resnet_ov.blob ${D}/opt/resnet_ov
    install -d ${D}/opt/yolotiny_ov
    install -m 0755 ${S}/yolotiny_ov/expected_result_sim_ov.dat ${D}/opt/yolotiny_ov
    install -m 0755 ${S}/yolotiny_ov/input_ov.dat ${D}/opt/yolotiny_ov
    install -m 0755 ${S}/yolotiny_ov/yolotiny_ov.blob ${D}/opt/yolotiny_ov
    install -d ${D}/opt/googlenet_ov
    install -m 0755 ${S}/googlenet_ov/expected_result_sim_ov.dat ${D}/opt/googlenet_ov
    install -m 0755 ${S}/googlenet_ov/input_ov.dat ${D}/opt/googlenet_ov
    install -m 0755 ${S}/googlenet_ov/googlenet_ov.blob ${D}/opt/googlenet_ov
    install -d ${D}/opt/inceptionv3_ov
    install -m 0755 ${S}/inceptionv3_ov/expected_result_sim_ov.dat ${D}/opt/inceptionv3_ov
    install -m 0755 ${S}/inceptionv3_ov/input_ov.dat ${D}/opt/inceptionv3_ov
    install -m 0755 ${S}/inceptionv3_ov/inceptionv3_ov.blob ${D}/opt/inceptionv3_ov
    install -d ${D}/opt/squeezenet_ov
    install -m 0755 ${S}/squeezenet_ov/expected_result_sim_ov.dat ${D}/opt/squeezenet_ov
    install -m 0755 ${S}/squeezenet_ov/input_ov.dat ${D}/opt/squeezenet_ov
    install -m 0755 ${S}/squeezenet_ov/squeezenet_ov.blob ${D}/opt/squeezenet_ov
    install -d ${D}/opt/fullyolov2_ov
    install -m 0755 ${S}/fullyolov2_ov/expected_result_sim_ov.dat ${D}/opt/fullyolov2_ov
    install -m 0755 ${S}/fullyolov2_ov/input_ov.dat ${D}/opt/fullyolov2_ov
    install -m 0755 ${S}/fullyolov2_ov/fullyolov2_ov.blob ${D}/opt/fullyolov2_ov

    install -d ${D}/opt/inceptionv4_ov
    install -m 0755 ${S}/inceptionv4_ov/expected_result_sim_ov.dat ${D}/opt/inceptionv4_ov
    install -m 0755 ${S}/inceptionv4_ov/input_ov.dat ${D}/opt/inceptionv4_ov
    install -m 0755 ${S}/inceptionv4_ov/inceptionv4_ov.blob ${D}/opt/inceptionv4_ov
    install -d ${D}/opt/resnet-101_ov
    install -m 0755 ${S}/resnet-101_ov/expected_result_sim_ov.dat ${D}/opt/resnet-101_ov
    install -m 0755 ${S}/resnet-101_ov/input_ov.dat ${D}/opt/resnet-101_ov
    install -m 0755 ${S}/resnet-101_ov/resnet-101_ov.blob ${D}/opt/resnet-101_ov
    install -d ${D}/opt/resnet-152_ov
    install -m 0755 ${S}/resnet-152_ov/expected_result_sim_ov.dat ${D}/opt/resnet-152_ov
    install -m 0755 ${S}/resnet-152_ov/input_ov.dat ${D}/opt/resnet-152_ov
    install -m 0755 ${S}/resnet-152_ov/resnet-101_ov.blob ${D}/opt/resnet-152_ov
    install -d ${D}/opt/yolotiny-v1_ov
    install -m 0755 ${S}/yolotiny-v1_ov/expected_result_sim_ov.dat ${D}/opt/yolotiny-v1_ov
    install -m 0755 ${S}/yolotiny-v1_ov/input_ov.dat ${D}/opt/yolotiny-v1_ov
    install -m 0755 ${S}/yolotiny-v1_ov/yolotiny-v1_ov.blob ${D}/opt/yolotiny-v1_ov
    install -d ${D}/opt/facenet_ov
    install -m 0755 ${S}/facenet_ov/expected_result_sim_ov.dat ${D}/opt/facenet_ov
    install -m 0755 ${S}/facenet_ov/input_ov.dat ${D}/opt/facenet_ov
    install -m 0755 ${S}/facenet_ov/facenet_ov.blob ${D}/opt/facenet_ov
}

FILES_${PN} += " /opt"
