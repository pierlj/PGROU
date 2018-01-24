
# prend en parametre :
	# 1 : la matrice de validation
	# 2...n : les modèles à apprendre




args = commandArgs(trailingOnly=TRUE)
library(randomForest)
output=(args[1])
nomEtude=(args[2])
validation=validation=read.table("resultat_MS.csv", row.names=1)



modele<-read.table("tools/resultat_MMRF.csv", row.names=1)
colnames(modele)[ncol(modele)]="HR"
ForetAlea<- randomForest(HR~.,data=modele, prox=TRUE, ntree=nrow(modele))


prediction=predict(ForetAlea, validation)






result=matrix(ncol=4,nrow=nrow(validation))
colnames(result)=c("study","patient","predictionscore","highriskflag")
result[,1]=nomEtude
result[,2]=row.names(validation)
result[,3]=prediction
result[,4]=prediction>0.5


write.table(result,output, sep="\t",quote=F,row.names=F)









