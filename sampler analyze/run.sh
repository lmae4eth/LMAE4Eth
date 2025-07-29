#!/bin/bash

sampler_options=("neighbor" "labor" "ladies" "poisson-ladies")

for sampler in "${sampler_options[@]}"
do
    echo "run --sampler=${sampler}..."
    python train_lightning.py --sampler "${sampler}"
    echo "--sampler=${sampler} ended"
done