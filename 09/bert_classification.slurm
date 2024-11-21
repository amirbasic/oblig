#!/bin/bash

#SBATCH --job-name=in5550
#SBATCH --account=ec30
#SBATCH --mail-type=FAIL
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=16

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
module load nlpl-transformers/4.20.1-foss-2021a-Python-3.9.5

MODEL='/fp/projects01/ec30/models/bert-base-cased' # can be /fp/projects01/ec30/models/bert-base-cased on Fox
TRAIN='sst_train.tsv'
TEST='sst_test.tsv'


python3 bert_classification.py --model ${MODEL} --trainset ${TRAIN} --devset ${TEST}
