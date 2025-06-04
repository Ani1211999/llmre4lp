#!/bin/bash

export OMP_NUM_THREADS=8

for DATASET in Cornell
do
    echo "DATASET=${DATASET}"
    
    for NUM_HOPS in 2 3
    do
        echo "NUM_HOPS=${NUM_HOPS}"
        echo "NUM_HOPS=${NUM_HOPS}" >> results/${DATASET}_mixhop.txt
        
        for COMBINE in concat sum
        do
            echo "COMBINE=${COMBINE}"
            echo "COMBINE=${COMBINE}" >> results/${DATASET}_mixhop.txt
            
            for LR in 0.003 0.01 0.03 0.1
            do
                echo "LR=${LR}"
                echo "LR=${LR}" >> results/${DATASET}_mixhop.txt
                
                for WD in 0 1e-5 1e-4 1e-3
                do
                    echo "WD=${WD}"
                    echo "WD=${WD}" >> results/${DATASET}_mixhop.txt
                    
                    for i in $(seq 0 9)
                    do
                        echo "Split=${i}"
                        echo "Split=${i}" >> results/${DATASET}_mixhop.txt
                        
                        python train_mixhop.py \
                            --dataset ${DATASET} \
                            --dropout 0.5 \
                            --lr ${LR} \
                            --hidden 32 \
                            --patience 100 \
                            --epochs 500 \
                            --weight_decay ${WD} \
                            --num_hops ${NUM_HOPS} \
                            --combine ${COMBINE} \
                            --seed $((i * 10)) \
                            --dataset_file_path ../../dataset/ \
                            --result_file_path ../../stage2/inputs/ >> results/${DATASET}_mixhop.txt
                    done
                done
            done
        done
    done
done