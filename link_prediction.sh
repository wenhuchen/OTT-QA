#!/bin/bash

for i in {0..7}
	do
	echo "Starting process", ${i}
	CUDA_VISIBLE_DEVICES=$i python link_prediction.py --shard ${i}@8 --do_all --load_from link_generator/model-ep9.pt --dataset data/all_plain_tables.json --batch_size 256 &
done
