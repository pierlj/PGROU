# -*-coding:Latin-1 -*

import os
import sys


nom=sys.argv[1].split("/")
nom=nom[len(nom)-1]
ligne=nom
# Charger liste des Observations
Observations={}

file=open(sys.argv[1],"rb")
data=file.readlines()
file.close()


for i in data:
#	print(i.split(" "))
	node=i.split(" ")[0]
	signe=i.split(" ")[2].split("\n")[0]
	#print(node+" "+signe)
	Observations[node]=float(signe)*1.0

# Chargement des Tuples
file=open(sys.argv[2],"rb")
data=file.readlines()
file.close()


# Pour chaque tuple
for i in data:
	listeSousTuples=[]
	listeSignes={}
	nbreMatching=0
	sommematching=0.0
	tuples=i.split("\n")[0].split(",")
	#print(tuples)
	# Pour chaque sous tuple
	for sousTuple in tuples:
		node=sousTuple.split(" ")[0]
		signe=sousTuple.split(" ")[1]
		listeSousTuples.append(node)
		listeSignes[node]=signe
#	print(len(listeSousTuples))
	# Pour chaque observation :
	for node in Observations.keys():
		# Si le noeud est présent
		if(node in listeSousTuples):
			#print(node+" "+listeSignes[node]+" "+str(Observations[node]))
			# Incrémenter matching
			nbreMatching=nbreMatching+1
			# incrémenter en fonction la somme matching
				# Si signe à + => ajouter valeur
			if(listeSignes[node]=="+"):
				sommematching=sommematching+Observations[node]
			# Si signe à - => 1-valeur
			else:
				sommematching=sommematching+(1-Observations[node])
			
	#print(sommematching)
	#print(nbreMatching)
	MS=0.5
	if(nbreMatching!=0):
		MS=sommematching/nbreMatching
	MS=max(MS, 1-MS)
	ligne=ligne+" "+str(MS)


print(ligne)
