#!/bin/bash
#BSUB -J TFC
#BSUB -o TFC_LOG_%J.out  
#BSUB -q hpc
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -N
# all  BSUB option comments should be above this line!

# execute our command

cd ..
set -e
source .venv/bin/activate
python3 main.py