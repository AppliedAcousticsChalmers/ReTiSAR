#!/bin/sh

# usage e.g.
# ./res/validation/record_ir.sh
#
# dependency: http://www.eca.cx/ecasound/download.php
# dependency: https://github.com/yoggy/sendosc

run_and_record() {
    NAME=${1}
    CONFIG="${2}"

    # shellcheck disable=SC2086
    python -m ReTiSAR ${CONFIG} &
    sleep "${STARTUP_SEC}"

    for AZIMUTH in 0 40 80 120 160; do
        set_azimuth "${AZIMUTH}"
        sleep .2
        record_target "${NAME}_${AZIMUTH}deg"
    done

    pkill -f ReTiSAR
    sleep 1
}

set_azimuth() {
    AZIMUTH=${1}

    if ! type "sendosc" >/dev/null 2>&1; then
        printf "\nlibrary \"sendosc\" not found. Set head azimuth %s deg manually and press enter ..." "${AZIMUTH}"
        read -r
    else
        printf "\nsending OSC to set head azimuth %s deg ...\n" "${AZIMUTH}"
        sendosc "${OSC_IP}" "${OSC_PORT}" /tracker/azimuth f "${AZIMUTH}"
    fi
}

record_target() {
    BASEDIR=$(dirname "${0}")
    REC_NAME=${1}

    # start playback of impulse slightly delayed!
    ecasound -q -x -t:"${REC_SEC}" \
        -a:1 -f:s32,1 -i playat,0.1,"${PWD}/res/source/Impulse_48.wav" -o jack,ReTiSAR-PreRenderer \
        -a:2 -f:s32,2 -i jack,ReTiSAR-Renderer -o "${BASEDIR}/${REC_NAME}_target_out.wav"
    sleep .3
}

NAME_PREFIX="rec_Impulse_"
STARTUP_SEC=17
REC_SEC=2.0
OSC_IP="127.0.0.1"
OSC_PORT=5005

SECONDS=0 # measure execution time
if ! type "ecasound" >/dev/null 2>&1; then
    printf "library \"ecasound\" not found, see http://www.eca.cx/ecasound/download.php.\n ... interrupted.\n"
    exit
fi

pkill -f ecasound
pkill -f ReTiSAR

CONFIG_NAME="4096_DP"
CONFIG="--STUDY_MODE -b=4096 -SP=FALSE -sh=8 -arr=0 -sht=NONE -irt=0 -tt=NONE -s=NONE -sp=[(37,0)] -gt=NONE -ar=res/ARIR/CR1_VSA_110RS_L.sofa -art=ARIR_SOFA -arl=0 -hr=res/HRIR/KU100_THK/HRIR_L2702.sofa -hrt=HRIR_SOFA -hrl=0 -hp=NONE"
run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}"

CONFIG_NAME="1024_DP"
CONFIG="--STUDY_MODE -b=1024 -SP=FALSE -sh=8 -arr=0 -sht=NONE -irt=0 -tt=NONE -s=NONE -sp=[(37,0)] -gt=NONE -ar=res/ARIR/CR1_VSA_110RS_L.sofa -art=ARIR_SOFA -arl=0 -hr=res/HRIR/KU100_THK/HRIR_L2702.sofa -hrt=HRIR_SOFA -hrl=0 -hp=NONE"
run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}"

CONFIG_NAME="4096_SP"
CONFIG="--STUDY_MODE -b=4096 -SP=TRUE -sh=8 -arr=0 -sht=NONE -irt=0 -tt=NONE -s=NONE -sp=[(37,0)] -gt=NONE -ar=res/ARIR/CR1_VSA_110RS_L.sofa -art=ARIR_SOFA -arl=0 -hr=res/HRIR/KU100_THK/HRIR_L2702.sofa -hrt=HRIR_SOFA -hrl=0 -hp=NONE"
run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}"

CONFIG_NAME="1024_SP"
CONFIG="--STUDY_MODE -b=1024 -SP=TRUE -sh=8 -arr=0 -sht=NONE -irt=0 -tt=NONE -s=NONE -sp=[(37,0)] -gt=NONE -ar=res/ARIR/CR1_VSA_110RS_L.sofa -art=ARIR_SOFA -arl=0 -hr=res/HRIR/KU100_THK/HRIR_L2702.sofa -hrt=HRIR_SOFA -hrl=0 -hp=NONE"
run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}"

# shellcheck disable=SC2039
echo ${SECONDS} | awk '{printf "\n ... finished in "int($1/60/60)"h "int($1/60)"m "int($1%60)"s.\n"}'
