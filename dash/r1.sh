#!/bin/bash   
#SBATCH --job-name=LGNN
#SBATCH --time=6:00:00
#SBATCH --partition=rtx8000,pascal  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc
#SBATCH --mem=60G
#SBATCH --output=out/output.o%j
#SBATCH --error=out/output.e%j

source $HOME/FFF/bin/activate
python src/main.py --data-split-num $1 --log d0_$2 --data-name $3 --drnl 0 --init-attribute ones --init-representation None --embedding-dim 16 
python src/main.py --data-split-num $1 --log d1_$2 --data-name $3 --drnl 1 --init-attribute ones --init-representation None --embedding-dim 16
