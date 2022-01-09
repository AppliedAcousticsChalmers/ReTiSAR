#!/bin/sh

# usage e.g.
# ./res/validation/record_snr.sh
#
# dependency: http://www.eca.cx/ecasound/download.php
# dependency: https://github.com/yoggy/sendosc

run_and_record() {
    NAME=${1}
    CONFIG="${2}"
    CONFIG_REC_CH=${3}
    IS_ALL_CH_MODE=${4}

    # shellcheck disable=SC2086
    python -m ReTiSAR ${CONFIG}${CONFIG_OVERWRITE} &
    sleep "${STARTUP_SEC}"

    if ! ${IS_ALL_CH_MODE}; then
        printf "\nmanually disconnect all connections from noise generator to renderer except for port %s\n and press enter to continue ..." "${CONFIG_REC_CH}"
        read -r
    fi

    for AZIMUTH in 0 45 90; do
        set_azimuth "${AZIMUTH}"
        record_target "${NAME}_${AZIMUTH}deg" "${CONFIG_REC_CH}"
        record_noise "${NAME}_${AZIMUTH}deg" "${CONFIG_REC_CH}"
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
    REC_CH=${2}

    if ! type "sendosc" >/dev/null 2>&1; then
        printf "\nlibrary \"sendosc\" not found. Set noise generator mute ON manually and press enter ..."
        read -r
    else
        printf "\nsending OSC to set noise generator mute ON ...\n"
        sendosc "${OSC_IP}" "${OSC_PORT}" /generator/mute b true
    fi
    sleep .2

    ecasound -q -x -t:"${REC_SEC}" \
        -a:1 -f:s32,1 -i "${PWD}/res/source/NoiseWhite_48.wav" -o jack,ReTiSAR-PreRenderer \
        -a:2 -f:s32,1 -i jack,ReTiSAR-PreRenderer:output_"${REC_CH}" -o "${BASEDIR}/${REC_NAME}_target_in.wav" \
        -a:3 -f:s32,2 -i jack,ReTiSAR-Renderer -o "${BASEDIR}/${REC_NAME}_target_out.wav"
    sleep .3
}

record_noise() {
    BASEDIR=$(dirname "${0}")
    REC_NAME=${1}
    REC_CH=${2}

    if ! type "sendosc" >/dev/null 2>&1; then
        printf "\nlibrary \"sendosc\" not found. Set noise generator mute OFF manually and press enter ..."
        read -r
    else
        printf "\nsending OSC to set noise generator mute OFF ...\n"
        sendosc "${OSC_IP}" "${OSC_PORT}" /generator/mute b false
    fi
    sleep .2

    ecasound -q -x -t:"${REC_SEC}" \
        -a:1 -f:s32,1 -i jack,ReTiSAR-Generator:output_"${REC_CH}" -o "${BASEDIR}/${REC_NAME}_noise_in.wav" \
        -a:2 -f:s32,2 -i jack,ReTiSAR-Renderer -o "${BASEDIR}/${REC_NAME}_noise_out.wav"
    sleep .3
}

NAME_PREFIX="rec_"
STARTUP_SEC=17
REC_SEC=2.0
OSC_IP="127.0.0.1"
OSC_PORT=5005
CONFIG_OVERWRITE=" -tt=NONE -s=NONE -gm=FALSE -arm=FALSE -hrm=FALSE -hp=NONE -irt=0 -sht=NONE --STUDY_MODE"

SECONDS=0 # measure execution time
if ! type "ecasound" >/dev/null 2>&1; then
    printf "library \"ecasound\" not found, see http://www.eca.cx/ecasound/download.php.\n ... interrupted.\n"
    exit
fi

pkill -f ecasound
pkill -f ReTiSAR

## ######################### #
#CONFIG_REC_CH=1
#
#CONFIG_NAME="230ch_8cm_sh12_0dB"
#CONFIG="-b=4096 -SP=TRUE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/DRIR_sim_LE230_PW_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=12"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#CONFIG_NAME="338ch_8cm_sh12_0dB"
#CONFIG="-b=4096 -SP=TRUE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/DRIR_sim_GL338_PW_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=12"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
## ######################### #

# ######################### #
CONFIG_REC_CH=1

CONFIG_NAME="32ch_4cm_sh4_0dB_incoherent"
CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=4"
run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
CONFIG_NAME="32ch_4cm_sh4_12dB_incoherent"
CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=12 -sh=4"
run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
CONFIG_NAME="32ch_4cm_sh4_0dB_coherent"
CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=4"
run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH} false
CONFIG_NAME="32ch_4cm_sh4_12dB_coherent"
CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=12 -sh=4"
run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH} false
# ######################### #

## ######################### #
#CONFIG_NAME="110ch_8cm_sh4_0dB"
#CONFIG_REC_CH=4
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/110RS_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=4"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="110ch_8cm_sh8_0dB"
#CONFIG_REC_CH=4
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/110RS_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=8"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="110ch_8cm_sh8_0dB_EQ"
#CONFIG_REC_CH=4
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/110RS_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_eq_CR1_VSA_110RS_L_struct.mat -arr=0 -sh=8"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="110ch_8cm_sh8_18dB"
#CONFIG_REC_CH=4
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/110RS_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=18 -sh=8"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="32ch_4cm_sh4_0dB"
#CONFIG_REC_CH=1
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=4"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="32ch_4cm_sh4_0dB_EQ"
#CONFIG_REC_CH=1
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_eq_Eigenmike_struct.mat -arr=0 -sh=4"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="32ch_4cm_sh4_18dB"
#CONFIG_REC_CH=1
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=18 -sh=4"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="32ch_8cm_sh4_0dB"
#CONFIG_REC_CH=1
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_r875_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=4"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="32ch_8cm_sh8_0dB"
#CONFIG_REC_CH=1
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/Eigenmike_synthetic_r875_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=8"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="110ch_8cm_sh8_0dB_1024"
#CONFIG_REC_CH=4
#CONFIG="-b=1024 -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/110RS_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=8"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH}
#
#CONFIG_NAME="110ch_8cm_sh8_0dB_1ch"
#CONFIG_REC_CH=4
#CONFIG="-b=4096 -SP=FALSE -art=ARIR_MIRO -hrt=HRIR_MIRO -gt=NOISE_IIR_PINK -gl=-20 -arl=-20 -hrl=0 -ar=res/ARIR/110RS_synthetic_struct.mat -hr=res/HRIR/KU100_THK/L2702_struct.mat -arr=0 -sh=8"
#run_and_record ${NAME_PREFIX}${CONFIG_NAME} "${CONFIG}" ${CONFIG_REC_CH} false
## ######################### #

# shellcheck disable=SC2039
echo ${SECONDS} | awk '{printf "\n ... finished in "int($1/3600)"h "int($1%3600/60)"m "int($1%60)"s.\n"}'
