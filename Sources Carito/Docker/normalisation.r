
args = commandArgs(trailingOnly=TRUE)
file=(args[1])
print(file)
##############
# Fonction de normalisation (moyenne des 2 normalisation)
#############
normalisation<-function(matrice){
	data=matrice
	
matrix1=matrix(0,ncol=ncol(data),nrow=nrow(data))
matrix2=matrix(0, ncol=ncol(data),nrow=nrow(data))

print("normalisation par lignes")
# Normalisation par lignes
for (i in c(1:nrow(data))){
min=min(data[i,])
max=max(data[i,])		

matrix1[i,]=as.numeric(data[i,]-min)/(max-min)
}



dataTemp=t(data)
matrixTemp2=t(matrix(0, ncol=ncol(data),nrow=nrow(data)))

# Normalisation par colonne
print("normalisation par colonnes")
for (i in c(1:nrow(dataTemp))){
min=min(dataTemp[i,])
max=max(dataTemp[i,])

matrixTemp2[i,]=as.numeric((dataTemp[i,]-min)/(max-min))
}

matrix2=t(matrixTemp2)
dataNormalise<-(matrix1+matrix2)/2
colnames(dataNormalise)=colnames(data)
rownames(dataNormalise)=rownames(data)

print("Ecriture résultat")
for (i in c(1:ncol(dataNormalise))){
	nomFichier=paste("dataPatientUnmapped",colnames(dataNormalise)[i],sep="/")
	write.table(file=nomFichier,dataNormalise[,i],col.names=FALSE, quote=F)
}


}
##############
# FIN DE LA FONCTION de normalisation (moyenne des 2 normalisation)
#############



data<-read.table(file,h=T,sep=",", row.names=1)
dataNormalise=normalisation(data)

