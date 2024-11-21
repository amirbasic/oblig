#!/bin/bash

#SBATCH --job-name=torfor-test-norbert
#SBATCH --account=ec30
#SBATCH --time=04:00:00
#SBATCH --partition=ifi_accel    # To use the accelerator nodes
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-task=10

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
module load nlpl-transformers/4.20.1-foss-2021a-Python-3.9.5
module load nlpl-gensim/4.2.0-foss-2021a-Python-3.9.5
module load nlpl-transformers/4.24.0-foss-2021a-Python-3.9.5


MODEL=${1} # can be /fp/projects01/ec30/models/bert-base-cased on Fox
TRAIN=${2}
TEST=${3}


python3 IN5550/oblig3/predict_on_test.py
