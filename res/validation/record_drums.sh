#!/bin/sh

# usage with e.g. 14 rec_{BlockLength}_{RadialLimit}dB 4 127.0.0.1 5005

BASEDIR=$(dirname "$0")
REC_MIC_CH=$1
NAME=$2"_mic"${REC_MIC_CH}
REC_SEC=$3
OSC_IP=$4
OSC_PORT=$5

if ! type "sendosc" >/dev/null 2>&1; then
    echo "\n \"sendosc\" not found. Set noise generator mute ON manually and press enter ..."
    read
else
    echo "\n sending OSC to set noise generator mute ON ...\n"
    sendosc ${OSC_IP} ${OSC_PORT} /generator/mute b true
fi
sleep 1

ecasound -x -f:s32,1 -i jack,ReTiSAR-PreRenderer:output_${REC_MIC_CH} -o ${BASEDIR}/${NAME}"_drums_in.wav" &
ecasound -x -f:s32 -i jack,ReTiSAR-Renderer -o ${BASEDIR}/${NAME}"_drums_out.wav" &
sleep 1 &&
ecasound -i ${BASEDIR}/../source/Drums_48.wav -o jack,ReTiSAR-PreRenderer &
sleep ${REC_SEC}
killall ecasound
sleep 1
