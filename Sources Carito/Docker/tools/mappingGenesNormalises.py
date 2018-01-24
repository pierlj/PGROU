# -*-coding:Latin-1 -*
import os
import sys

Dico={}
# Charger la table de mapping
file=open(sys.argv[1],"rb")
data=file.readlines()
file.close()


for i in data:
#	print(i)
	gene=i.split("\t")[0]
	entrez=i.split("\t")[1]
	#print(gene+" "+entrez)
	Dico[entrez]=gene


# Charger liste gènes selectionnées et aficher la conversion
file=open(sys.argv[2],"rb")
data=file.readlines()
file.close()

# Ouvrir fichier sortie
file=open("dataPatient/"+sys.argv[3],"w")


for i in data:
	nom=i.split(" ")[0]
	valeur=i.split(" ")[1]
	if(nom in Dico.keys()):
		file.write(Dico[nom]+" = "+valeur)

file.close()
