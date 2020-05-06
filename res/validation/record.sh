#!/bin/sh

# usage with e.g. 14 rec_{BlockLength}_{RadialLimit}dB
# dependency: https://github.com/yoggy/sendosc

BASEDIR=$(dirname "$0")
REC_MIC_CH=$1
NAME=$2
REC_SEC=4
OSC_IP="127.0.0.1"
OSC_PORT=5005

AZIMUTH=0
if ! type "sendosc" >/dev/null 2>&1; then
    echo "\n \"sendosc\" not found. Set head azimuth $AZIMUTH deg manually and press enter ..."
    read
else
    echo "\n sending OSC to set head azimuth $AZIMUTH deg ...\n"
    sendosc ${OSC_IP} ${OSC_PORT} /tracker/azimuth f ${AZIMUTH}
fi

${BASEDIR}/record_noise.sh ${REC_MIC_CH} ${NAME}"_"${AZIMUTH}"deg" ${REC_SEC} ${OSC_IP} ${OSC_PORT}
${BASEDIR}/record_drums.sh ${REC_MIC_CH} ${NAME}"_"${AZIMUTH}"deg" ${REC_SEC} ${OSC_IP} ${OSC_PORT}

AZIMUTH=-90
if ! type "sendosc" >/dev/null 2>&1; then
    echo "\n \"sendosc\" not found. Set head azimuth $AZIMUTH deg manually and press enter ..."
    read
else
    echo "\n sending OSC to set head azimuth $AZIMUTH deg ...\n"
    sendosc ${OSC_IP} ${OSC_PORT} /tracker/azimuth f ${AZIMUTH}
fi

${BASEDIR}/record_noise.sh ${REC_MIC_CH} ${NAME}"_"${AZIMUTH}"deg" ${REC_SEC} ${OSC_IP} ${OSC_PORT}
${BASEDIR}/record_drums.sh ${REC_MIC_CH} ${NAME}"_"${AZIMUTH}"deg" ${REC_SEC} ${OSC_IP} ${OSC_PORT}

echo "\n ... recording finished.\n"
