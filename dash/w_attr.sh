#!/bin/bash   
#SBATCH --job-name=LGNN
#SBATCH --time=6:00:00
#SBATCH --partition=rtx8000,pascal  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc
#SBATCH --mem=60G
#SBATCH --output=out/output.o%j
#SBATCH --error=out/output.e%j

source $HOME/FFF/bin/activate
python src/main.py --seed $1 --log $2 --data-name $3 --drnl 0 --init-attribute None --init-representation $4 --embedding-dim 32 