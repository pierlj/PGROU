

file=$1
etude=$2
echo $1
echo "normalisation"
output="resultTemp/"$(echo $file | cut -f2 -d"/")
#cat $file | head -1

echo $etude

Rscript normalisation.r $file

#ls dataPatientUnmapped/*
echo "mapping"
for i in dataPatientUnmapped/*
do
nom="Patient"$(echo $i | cut -f2 -d"/" | cut -f1,2 -d"_")
nomOutput=$(echo $nom | sed s/"PatientX"/""/g | sed s/"Patient"/""/g)
python tools/mappingGenesNormalises.py tools/mapping.csv $i $nomOutput 
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
