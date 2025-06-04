#!/bin/bash

export OMP_NUM_THREADS=8

for DATASET in Cornell
do
    echo "DATASET=${DATASET}"
    for LAYERS in 4 5 6 7 8
    do
        echo "LAYER_NUM=${LAYERS}"
        echo "LAYER_NUM=${LAYERS}" >> results/${DATASET}.txt
        for LR in 0.003 0.01 0.03 0.1
        do
            echo "LR=${LR}"
            echo "LR=${LR}" >> results/${DATASET}.txt
            for WD in 0 1e-5 1e-4 1e-3
            do
                echo "WD=${WD}"
                echo "WD=${WD}" >> results/${DATASET}.txt
                for i in $(seq 0 9)
                do
                    echo "Split=${i}"
                    echo "Split=${i}" >> results/${DATASET}.txt
                    python train.py \
                        --dataset ${DATASET} \
                        --dropout 0.5 \
                        --eps 0.4 \
                        --lr ${LR} \
                        --hidden 32 \
                        --patience 100 \
                        --epochs 500 \
                        --weight_decay ${WD} \
                        --layer_num ${LAYERS} \
                        --seed $((i * 10)) \
                        --dataset_file_path ../../dataset/ \
                        --seed $((i * 10)) \
                        --result_file_path ../../stage2/inputs/ >> results/${DATASET}.txt
                done
            done
        done
    done
done