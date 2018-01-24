#!/bin/bash

## Simply invoke the R script that implements our model.
## This R script assumes that it runs in the root directory of the Docker image and that
## the test data are mounted in /test-data,
## the output should be written to /output,
## and that the entire directory structure of the submitted Docker image
## (e.g., an R object encapsulating trained modeler state) is mounted at /
mkdir dataPatientUnmapped/
mkdir dataPatient/
mkdir output/
mkdir resultTemp/
# 2 fichiers à tester 
	# GSE15695 [HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Arra
	# Hose	

#ls /test-data/

./scriptAnalyseAffyMetrix.sh test-data/GSE15695entrezIDlevel* GSE15695
./scriptAnalyseAffyMetrix.sh test-data/HoseentrezIDlevel* Hose


./scriptAnalyseSingleCell.sh test-data/dfci.2009_entrezID* DFCI
./scriptAnalyseSingleCell.sh test-data/m2gen_entrezID* M2Gen


#cat resultTemp/dfci.2009_entrezID_TPM_hg19_LBR1.csv | sed s/"_[1,2]_[P,B]M"/""/g | grep "M\t"

cat resultTemp/* | grep "patient" | sort | uniq > output/predictions.tsv 
cat resultTemp/* | sort | uniq | grep -v "patient" >> output/predictions.tsv


cat output/predictions.tsv 
