#!/bin/bash
#BSUB -J TFC
#BSUB -o TFC_LOG_%J.out  
#BSUB -q hpc
#BSUB -W 48:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 28
#BSUB -R "span[hosts=1]"
# all  BSUB option comments should be above this line!

# execute our command

set -e
source ../.venv/bin/activate
#pip3 freeze
#module load scipy/1.9.1-python-3.9.14
python3 main.py