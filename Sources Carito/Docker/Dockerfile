FROM ubuntu
RUN apt-get update && apt-get install -y r-base

## Install dependencies
RUN Rscript -e 'source("http://bioconductor.org/biocLite.R"); biocLite("plyr")'
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
RUN Rscript -e "install.packages('data.table')"
RUN Rscript -e "install.packages('randomForest')"

## Copy files over to Docker image
COPY score_sc2.sh /score_sc2.sh
COPY scriptAnalyseAffyMetrix.sh /scriptAnalyseAffyMetrix.sh
COPY normalisation.r /normalisation.r
COPY scoringAffy.r /scoringAffy.r
COPY scoringSingleCell.r /scoringSingleCell.r
COPY scriptAnalyseSingleCell.sh /scriptAnalyseSingleCell.sh

COPY tools/mappingGenesNormalises.py /tools/mappingGenesNormalises.py
COPY tools/mapping.csv /tools/mapping.csv
COPY tools/MSComputing.py /tools/MSComputing.py
COPY tools/componentsAffy.csv /tools/componentsAffy.csv
COPY tools/resultat_GSE19784HOVON65.csv /tools/resultat_GSE19784HOVON65.csv
COPY tools/resultat_GSE24080UAMS.csv /tools/resultat_GSE24080UAMS.csv 


