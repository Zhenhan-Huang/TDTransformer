#!/bin/bash
GPU=0

MODEL=$1

BASEDIR="output"

CONFIG=configs/${MODEL}/dev.yaml

for DATASET in 40975 31 44 1017 188 1494 40687 42178 40664 42734 1464 50 1504 1510 1489 1063 1467 1480 1068 1049 1050 312 38 24 40701 803 1462 40983 40900 871 1558 976 1056 807 1020 819 1471 1046 1053 1461 1220 1491 1492 1493 6 42 16 14 12 18 28 40966 1552 1548 185 22 182 1475 1497 183 4538 1459 1476 26 184 679 473 46 1560 1529 1540 1538 1568 1525 40474 40475; do
    for SEED in 1 2 3; do
        bash ./scripts/launch.sh $DATASET $MODEL $SEED $BASEDIR $CONFIG $GPU

        if [  $? -ne 0 ]; then
            echo "Something wrong with the program execution, stop the execution"
            exit 1
        fi
    done
done
