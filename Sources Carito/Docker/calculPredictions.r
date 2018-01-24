
# prend en parametre :
	# 1 : la matrice de validation
	# 2...n : les modèles à apprendre




args = commandArgs(trailingOnly=TRUE)
taille=(length(args))
library(randomForest)
######
# Fonction de génération d'une random forest
######
apprentissage<-function(file){
	modele<-read.table(file, row.names=1)
	colnames(modele)[ncol(modele)]="HR"
	ForetAlea<- randomForest(HR~.,data=modele, prox=TRUE, ntree=nrow(modele))
	return(ForetAlea)

}

####
## Fin fonction de génération random forsest
#######"
validation=validation=read.table("resultat_MS.csv", row.names=1)

prediction=0
for (i in args){
	#print(i)
	modele<-read.table(i, row.names=1)
	colnames(modele)[ncol(modele)]="HR"
	ForetAlea<- randomForest(HR~.,data=modele, prox=TRUE, ntree=nrow(modele))
	prediction=prediction+predict(ForetAlea, validation)
}
do.trace
prediction=prediction/taille
predictionsimplifie=prediction>0.5

result=matrix(ncol=4,nrow=nrow(validation))

result[,2]=row.names(validation)

MMRF	MMRF_1030	7.42126240254235	FALSE
