#####
# Calcul du MS + prise intégration données cliniques
#####

# Clinic/GSE19784HOVON65
rm resultat_GSE19784HOVON65.csv
for i in dataPatientGSE19784HOVON65/*
do
nom=$(echo $i | cut -f2 -d"/")
echo $i
cat $i | grep -v " = NA" > test
cat test > $i

# Calcul du MS
python tools/MSComputing.py $i components.csv>test
clinique=$(cat Clinic/GSE19784HOVON65_Clinic | grep $nom | cut -f2 -d"," )


cat test | sed s/"$"/" "$clinique/g | grep -v "CENSORED" >> resultat_GSE19784HOVON65.csv

done
rm test



# Clinic/GSE24080UAMS_Clinic

rm resultat_GSE24080UAMS.csv
for i in dataPatientGSE24080UAMS/*
do
nom=$(echo $i | cut -f2 -d"/")
echo $i
cat $i | grep -v " = NA" > test
cat test > $i

python tools/MSComputing.py $i components.csv>test
clinique=$(cat Clinic/GSE24080UAMS_Clinic | grep $nom | cut -f2 -d"," )

cat test | sed s/"$"/" "$clinique/g | grep -v "CENSORED" >> resultat_GSE24080UAMS.csv

done
rm test




####
# Partie modélisation prédictive
#####

R
# Chargement librairie => Foret arbre de décision
library(randomForest)

##############.

# Génération de la foret pour les données HOVON
modele<-read.table("resultat_GSE19784HOVON65.csv", row.names=1)
colnames(modele)[ncol(modele)]="HR"
ForetAleaHOVON<- randomForest(HR~.,data=modele, prox=TRUE, ntree=nrow(modele))

# Génération de la foret pour les données UAMS
modele<-read.table("resultat_GSE24080UAMS.csv", row.names=1)
colnames(modele)[ncol(modele)]="HR"
ForetAleaUAMS<- randomForest(HR~.,data=modele, prox=TRUE, ntree=nrow(modele))


# Test du modele HOVON sur données UAMS
validation=read.table("resultat_GSE24080UAMS.csv", row.names=1)
colnames(validation)[ncol(modele)]="HR"
prediction=predict(ForetAleaHOVON, validation)
prediction=prediction>0.5
matriceConfusion=table(validation$HR, prediction)
precision=sum(diag(matriceConfusion))/sum(matriceConfusion)
print(precision)


# Test du modele UAMS sur données HOVON
validation=read.table("resultat_GSE19784HOVON65.csv", row.names=1)
colnames(validation)[ncol(modele)]="HR"
prediction=predict(ForetAleaUAMS, validation)
prediction=prediction>0.5
matriceConfusion=table(validation$HR, prediction)
precision=sum(diag(matriceConfusion))/sum(matriceConfusion)
print(precision)


