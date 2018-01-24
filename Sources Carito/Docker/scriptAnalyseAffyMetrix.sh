


file=$1
etude=$2
echo $1
echo "normalisation"
output="resultTemp/"$(echo $file | cut -f2 -d"/")
#cat $file | head -1

echo $etude

Rscript normalisation.r $file

#Mapping en protéines (fichier de mapping)
# Mapper données observées
#ls dataPatientUnmapped/*
echo "mapping"
for i in dataPatientUnmapped/*
do
nom=$(echo $i | cut -f2 -d"/")
python tools/mappingGenesNormalises.py tools/mapping.csv $i $nom 
done
rm dataPatientUnmapped/*


#Calcul du MS (component)
rm resultat_MS.csv
echo "calcul du MS"
for i in dataPatient/*
do
nom=$(echo $i | cut -f2 -d"/")
cat $i | grep -v " = NA" > test
cat test > $i
python tools/MSComputing.py $i tools/componentsAffy.csv >> resultat_MS.csv
done
rm test

rm dataPatient/*

echo "prediction"
#Génération du modèle d’apprentissage (MS calculés)
Rscript scoringAffy.r $output $etude
