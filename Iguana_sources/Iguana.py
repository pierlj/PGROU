# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:57:38 2017

@author: Khalil Boulkenafet, Pierre Le Jeune, Jinhui Liu, Jules Paris and Justin Voïnéa
"""

#importation des modules nécessaires à l'application 
import sys
import os
from PyQt5 import QtWidgets, QtCore
from py2cytoscape.data.cyrest_client import CyRestClient
import psutil
import networkx as nx
from sklearn.metrics import confusion_matrix
import numpy
import re
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random



#import du fichier génrant l'interface graphique
import interface_ui

global dir_path

dir_path = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(dir_path)

#classe principale permettant de lancer l'application
class Pappl(QtWidgets.QWidget, interface_ui.Ui_Form):
    
    fname = ""
    grapheLoc=[]
    grapheLoc2=[]
    table=[]
    
    dataTraining=""
    dataTest=[]
    
    trainFeatures=[]
    trainLabels=[]
    
    testFeatures=[]
    testLabels=[]
    
    clf=[]
    
    patientsDatasForSimilDirURL = ""
    clinicDatasForSimilFileURL = ""
    componentsFileUrl = ""
    grapheGenesFileURL = ""
    
    nbr = 0
    
    dataPrediction=""
    
    
    def __init__(self):
        super(Pappl, self).__init__()
        self.setupUi(self)
        self.connectActions()
        self.setWindowFlags(QtCore.Qt.Window |
        QtCore.Qt.CustomizeWindowHint |
        QtCore.Qt.WindowTitleHint |
        QtCore.Qt.WindowCloseButtonHint |
        #QtCore.Qt.WindowStaysOnTopHint |
        QtCore.Qt.WindowContextHelpButtonHint)
        
    #fonction permettant de lier les boutons de l'interface avec les fonctions correspondantes
    def connectActions(self):
        self.load.clicked.connect(self.loading)
        self.load_2.clicked.connect(self.loading)
        self.load_3.clicked.connect(self.loading)
        self.launch.clicked.connect(self.lancement)
        self.display.clicked.connect(self.affichage)
        self.display.setEnabled(False)
        self.reduc.clicked.connect(self.reduction)
        self.reduc.setEnabled(False)
        self.color.clicked.connect(self.colorations)
        # self.color.setEnabled(False)
        self.color_1.clicked.connect(self.afficheColor)
        # self.color_1.setEnabled(False)
        self.nCompo.clicked.connect(self.nComposantesAux)
        # self.nCompo.setEnabled(False)"
        self.train.clicked.connect(self.training)
        self.train.setEnabled(False)
        self.chercheFichier.clicked.connect(self.loadingDataTrain)
        self.test.clicked.connect(self.testData)
        self.test.setEnabled(False)
        self.pushButton.clicked.connect(self.loadingDataTest)
        self.buttonDonnee.clicked.connect(self.loadPatientsDatasForSimil)
        self.buttonClinique.clicked.connect(self.loadClinicDatasForSimil)
        self.buttonCsv.clicked.connect(self.loadComponentsFileForSimil)
        self.buttonSif.clicked.connect(self.loadGrapheGenesFileForSimil)
        self.graphSimi.toggled.connect(self.modifyDisplayGraphState)
        self.dataPred.toggled.connect(self.modifyButtonClinique)
        self.chercher.clicked.connect(self.loadDataPred)
        self.calcul.clicked.connect(self.runSimilAlgorithm)
        self.lancer.clicked.connect(self.prediction)
        
    #ouverture d'une fenetre d'erreur lorsque cytoscape n'est pas lancé    
    def alerte(a):
        msglabel = QtWidgets.QLabel("Attention Cytoscape.exe n'est pas lancé ! \nVeuillez lancer l'application avant d'afficher un graphe.")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Attention")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    #affichage d'un message lorsque la réduction est terminée
    def doneR(a):
        msglabel = QtWidgets.QLabel("\t\tLa réduction est finie.\t\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Fini")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    
    #affichage d'un message lorsque l'identification des colorations est terminée
    def doneI(a):
        msglabel = QtWidgets.QLabel("\t\tL'identification des composants est finie.\t\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Fini")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    #affichage d'un message lorsque la coloration du graphe est terminée   
    def doneC(a):
        msglabel = QtWidgets.QLabel("\t\tLa coloration est finie.\t\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Fini")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    #affichage d'un message lorsque l'exportation des composants est terminée
    def doneN(a):
        msglabel = QtWidgets.QLabel("\t\tL'exportation des composants est terminée.\t\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Fini")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    def doneClass(a):
        msglabel = QtWidgets.QLabel("\t\tCréation du classificateur terminée.\t\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Fini")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
        
    def donePred(a):
        msglabel = QtWidgets.QLabel("\t\tPrédiction terminée.\t\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Fini")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
        
    def doneSimil(a):
        msglabel = QtWidgets.QLabel("\t\tCalcul de similarité terminé.\t\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Fini")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    #affichage d'un message lorsque l'on veut effectuer une tâche sans que les fichiers nécessaires soient présents dans le répertoire correspondant
    def pb(a):
        msglabel = QtWidgets.QLabel("Attention : il manque les fichiers necéssaires pour effectuer cette opération.\nMerci d'effectuer d'abord la réduction du graphe.")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Attention")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    def missingFiles(a):
        msglabel = QtWidgets.QLabel("\tAttention : il manque des fichiers necéssaires pour effectuer cette opération.\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Attention")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    def missingTestFiles(a):
        msglabel = QtWidgets.QLabel("\tAttention : veuillez sélectionner un fichier à tester.\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Attention")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    def noClass(a):
        msglabel = QtWidgets.QLabel("\tAttention : classificateur manquant.\t\n Veuillez créer un classificateur depuis l'onglet de classification.")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Attention")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    def wrongPatient(self):
        msglabel = QtWidgets.QLabel("\tAttention : le patient recherché n'existe pas.\t\n \tVeuillez réessayer avec un autre nom.\t")
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Attention")
        ok = QtWidgets.QPushButton('OK', dialog)
        ok.clicked.connect(dialog.accept)
        ok.setDefault(True)
        dialog.layout = QtWidgets.QGridLayout(dialog)
        dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
        dialog.layout.addWidget(ok, 1, 1)
        dialog.exec_()
    
    #fonction determinant si Cytoscape est lancé
    def isRunning(s):
        for pid in psutil.pids():
            p=psutil.Process(pid)
            if p.name()=="Cytoscape.exe":
                return True
        return False
    
    #chargement des graphes en ouvrant un explorateur de fichier
    def loading(self):
        nom=QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', dir_path + '\\', '*.sif')
        if (len(nom[0])>0):
            self.grapheLoc.append(nom[0])
            self.grapheLoc2.append(nom[0])
            self.graph.addItem(os.path.basename(str(nom[0])))
            self.graph_2.addItem(os.path.basename(str(nom[0])))
            self.graph_3.addItem(os.path.basename(str(nom[0])))
            self.display.setEnabled(True)
            self.reduc.setEnabled(True)
            self.color.setEnabled(True)
            
    #lancement de la réduction, appel de l'algorithme Iggy-poc avec les bons fichiers
    def reduction(self):
        self.fname = self.graph_2.currentItem().text()
        l=len(self.grapheLoc2[self.graph_2.currentRow()])
        reduced=self.grapheLoc2[self.graph_2.currentRow()][:l-4]+"-reduced.sif"
        self.compaction(self.grapheLoc2[self.graph_2.currentRow()],self.grapheLoc2[self.graph_2.currentRow()][:l-4]+"-reduced.sif", self.grapheLoc2[self.graph_2.currentRow()][:l-4]+"-reduced-hash.txt", self.grapheLoc2[self.graph_2.currentRow()][:l-4]+"-reduced-logic.txt")
        self.grapheLoc.append(reduced)
        self.graph.addItem(os.path.basename(str(reduced)))
        self.doneR()

    #lancement, via commande dos du script asp avec clingo
    #dépend des options sélectionnées dans l'application (temps d'exécution)
    def colorations(self):
        nbColor=0 # possibilite de changer le nombre de reponse:  0 -> all answers n -> n answers
        name = self.grapheLoc2[self.graph_3.currentRow()]
        input=os.path.splitext(name)[0]+"-reduced-logic.txt"
        output=dir_path+"\ASPout.txt"
        #option apres input : --opt-mode=optN --enum-mode=cautious --quiet=1
        print(input)
        if (self.time.value()==0):
            command=dir_path+"\clingo.exe " +str(nbColor)+" "+dir_path +"\optimizationComponent.lp "+input+" --opt-mode=optN --enum-mode=cautious --parallel-mode=2  > "+ os.path.splitext(input)[0] +"-colorations.txt"
        else:
            command=dir_path+"\clingo.exe " +str(nbColor)+" "+dir_path +"\optimizationComponent.lp "+input+" --time-limit="+str(self.time.value())+" --opt-mode=optN --enum-mode=cautious > "+ os.path.splitext(input)[0] +"-colorations.txt"
        print(command)
    
        os.system(command)
        try:
            self.processASP(os.path.splitext(name)[0]+"-reduced-logic-colorations.txt")
            self.identificationColor(os.path.splitext(name)[0]+"-reduced-logic-colorations-processed.txt")
            self.doneI()
        except (IndexError, EnvironmentError):
            self.pb()
    
        #self.processASP(os.path.splitext(name)[0]+"-reduced-logic-colorations.txt")
        #self.identificationColor(os.path.splitext(name)[0]+"-reduced-logic-colorations-processed.txt")
        #self.doneI()
    #lancement de la fonction d'affichage        
    def afficheColor(self):
        self.colorGraphe(self.table)
    
        
    def nComposantesAux(self):
        self.nComposantes(self.table)
        self.doneN()
    
    
    #lancement de cytoscape --> si cytoscape n'est pas installé dans le bon répertoire, il y aura une erreur
    #Il faudrait proposer une option dans l'application pour donner le chemin d'accès de cytoscape et le changer ici. 
    def lancement(self):
        os.startfile(r'C:\Program Files\Cytoscape_v3.5.1\Cytoscape.exe')
    
    
    

    def training(self):
        model = 0
        if self.nomFichier.item(0).text() != "":
            r=[]
            for i in range (100):
                seed = random.randint(1,10000)
                test_size = self.spinBox.value()
                X_train, X_test, y_train, y_test = train_test_split(self.trainFeatures, self.trainLabels, test_size=test_size, random_state=seed)
                #print(X_test)
                # fit model no training data
                model = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
                model.fit(X_train, y_train)
            
            
                y_pred = model.predict(X_test)
            #predictions = [round(value) for value in y_pred]
            
                accuracy = accuracy_score(y_test, y_pred)
                r.append(accuracy)
            
            self.clf.append(model)
            if (not self.test.isEnabled()):
                self.comboBox.clear()
                self.classificateurPred.clear()
            self.taux.setText(str(sum(r)/100*100.0)+"%")
            self.comboBox.addItem("Classificateur "+str(len(self.clf)))
            self.classificateurPred.addItem("Classificateur "+str(len(self.clf)))
            self.test.setEnabled(True)
            self.doneClass()
        else:
            self.missingFiles()
        
    def testData(self):
        if  self.listWidget.count() != 0:
            path=self.dataTest[self.listWidget.currentRow()]
            (features,labels)=self.importData(path)
            (p,matrix)=self.validation(features,labels)
            self.fichier.setText(os.path.basename(path))
            self.precision.setText(str('{:01.2f}'.format(p)))
            self.matrice.setText("\tPositive\tNegative\nPositive\t"+str(matrix[0][0])+"\t"+str(matrix[0][1])+"\nNegative\t"+str(matrix[1][0])+"\t"+str(matrix[1][1]))
        else:
            self.missingTestFiles()
        
    def validation(self,X,y):
        #print("Prediction:")
        pred=self.clf[self.comboBox.currentIndex()].predict(X)
        
        confusionMatrix=confusion_matrix(y,pred)
        
        
        precision=sum(numpy.diagonal(confusionMatrix))/sum(sum(confusionMatrix))
        return (precision,confusionMatrix) 
        
        
        
    def loadingDataTrain(self):
        
        nom=QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', dir_path + '\\', '*.csv')
        if (len(nom[0])>0):
            
            self.dataTrain=nom[0]
            self.nomFichier.clear()
            self.nomFichier.addItem(str(nom[0]))
            self.train.setEnabled(True)
            
        (self.trainFeatures,self.trainLabels)=self.importData(self.dataTrain)
    
    def importData(self,chemin):
        path=chemin.replace('\\','\\\\')
        
        file=open(chemin,'r')
        data=file.readlines()
        
        names=[]
        X=[]
        y=[]
        for i in data:
            tab=i.split(" ")
            names.append(tab[0])
            X.append(tab[1:-1])
            y.append(tab[-1][:-1])
        
        for i in range (len(X)):
            for j in range (len (X[i])):
                X[i][j]=float(X[i][j])
        file.close()
        return(numpy.array(X),numpy.array(y))
    
    def importDataPred(self,chemin):
        path=chemin.replace('\\','\\\\')
        print(chemin)
        file=open(chemin,'r')
        data=file.readlines()
        names=[]
        X=[]
        
        for i in data:
            tab=i.split(" ")
            names.append(tab[0])
            X.append(tab[1:])
        
        for i in range (len(X)):
            for j in range (len (X[i])):
                X[i][j]=float(X[i][j])
        file.close()
        return(numpy.array(X),names)
            
    
    def prediction(self):
        if self.classificateurPred.itemText(0) == "--Classificateur--":
            self.noClass()
        
        elif  self.patients.item(0).text() != "" and self.patients.item(0).text() != "Données_patients":
            print(self.classificateurPred.itemText(0))
            (X,names)=self.importDataPred(self.dataPrediction)
            y=self.clf[self.classificateurPred.currentIndex()].predict(X)
            res=[]
            n=len(X)
            for i in range(n):
                res.append(names[i]+" \t"+y[i])
            path=self.dataPrediction[:-4]+"-predictions.csv"
            file=open(path,'w')
            for line in res:
                file.write(line)
                file.write("\n")
            file.close()
            self.donePred()
        
        else:
            self.missingFiles()
        
        
    
    def loadDataPred(self):
        nom=QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', dir_path + '\\', '*.csv')
        self.dataPrediction=nom[0]
        if (len(nom[0])>0):
            self.patients.clear()
            self.patients.addItem(nom[0])
    
    
            
    def loadingDataTest(self):
        print("loadTest")
        nom=QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', dir_path + '\\', '*.csv')
        if (len(nom[0])>0):
            self.dataTest.append(nom[0])
            
            self.listWidget.addItem(os.path.basename(str(nom[0])))
            self.test.setEnabled(True)
            
            

    
    
    #affichage du graphe selectionné dans cytoscape
    def affichage(self):
        if not self.isRunning():
            self.alerte()
        else:
            cy = CyRestClient()
        
            style1 = cy.style.create('sample_style1')
            #feuille de style cytoscape
            new_defaults = {
                    
            # Node defaults
            'NODE_FILL_COLOR': '#ff5500',
            'NODE_SIZE': 20,
            'NODE_BORDER_WIDTH': 0,
            'NODE_TRANSPARENCY': 120,
            'NODE_LABEL_COLOR': 'white',
            
            # Edge defaults
            'EDGE_WIDTH': 3,
            'EDGE_STROKE_UNSELECTED_PAINT': '#aaaaaa',
            #'EDGE_LINE_TYPE': 'LONG_DASH',
            'EDGE_TRANSPARENCY': 120,
            
            
            # Network defaults
            'NETWORK_BACKGROUND_PAINT': 'white'
            }
            
            # Update
            style1.update_defaults(new_defaults)
            
            kv_pair = {
                '-1': 'T',
                '1': 'Arrow'
            }
            style1.create_discrete_mapping(column='interaction', 
                                        col_type='String', vp='EDGE_SOURCE_ARROW_SHAPE', mappings=kv_pair)
            
            self.fname = self.grapheLoc[self.graph.currentRow()]
            net1 = cy.network.create_from(self.grapheLoc[self.graph.currentRow()])
            cy.style.apply(style1,net1)

    def InversionTuple(self,node):
        Dico={}
        Dico["+"]="-"
        Dico["-"]="+"
        tupleInverse=""
        for sousTuple in node.split(","):
            #print(sousTuple)
            sousNode=sousTuple.split(" ")[0]
            sousSigne=sousTuple.split(" ")[1]
            #print(sousNode)
        #    print(sousSign
            tupleInverse=tupleInverse+","+sousNode+" "+Dico[sousSigne]    
        tupleInverse=(tupleInverse[1:len(tupleInverse)+1])
        return tupleInverse
    
    
    
    # Renvoie le nom du tuple fusionnant les prÃ©dÃ©cesseurs fusionnables (en prenant en compte le type d'arc)
    def FusionTuples(self,PredecesseursFusionnables,node,G):
        nouveauTuple=""
        for sousTuple in PredecesseursFusionnables:
            #print(sousTuple)
            tupleCalcule=sousTuple
            arc=G[sousTuple][node][0]['edge_type']
            if(arc == "-1"):
    #            print("inversion")
                tupleCalcule=self.InversionTuple(tupleCalcule)
                
            nouveauTuple=nouveauTuple+","+tupleCalcule
        nouveauTuple=(nouveauTuple[1:len(nouveauTuple)+1])
        return(nouveauTuple)     
    
    
    
    # Renvoie la liste des prÃ©dÃ©cesseurs qui peuvent fusionner
    def IdentificationPredecessorsFusionnables(self,G,ListePredecesseurs,Compacte):
        fusionnable=[]
        for node in ListePredecesseurs:
            listeSuccesseurs=G.successors(node)        
            listePredecessors=G.predecessors(node)
            if((len(listeSuccesseurs)==1) and (len(listePredecessors)==0) and (len(G[node][listeSuccesseurs[0]])==1) and (node not in Compacte) and (listeSuccesseurs[0] not in Compacte) ):
                fusionnable.append(node)
        return(fusionnable)
    
    
    # Renvoie false si 
        # la target a un arc vers la source de signe diffÃ©rent
    def FusionPossible(self,source,target, G, arc):
        resultat=True
        listeTarget=G.successors(target)
        # Parcours de successeurs de la cible
        if(source in listeTarget):
            for edge in G[target][source]:
                if(G[target][source][edge]['edge_type'] != arc):
                    return False
    
        return True
    
    def FusionNodes(self,graphe,dico,inversionArc):
        # CrÃ©er un nouveau graphe H
        H=nx.MultiDiGraph()
        ListeArcs=[]
        inversion={}
        inversion["1"]="-1"
        inversion["-1"]="1"
        # Pour chaque noeud du graphe
        for node in graphe.nodes():
            H.add_node(dico[node])
        # Pour chaque arc du graphe G
        for edges in graphe.edges():
            #print(edges)
            source=edges[0]
            target=edges[1]
            #print("ci")
            for edge in graphe[source][target]:
                #print(graphe[source][target][edge])
                arc=graphe[source][target][edge]['edge_type']
                if(source in inversionArc):
                    arc=inversion[arc]
                    #print(dico[source]+" => "+str(arc)+" => "+dico[target])
                edge=dico[source]+dico[target]+str(arc)
                #print(edge)
                # Si l'arc n'existe pas encore
                if(edge not in ListeArcs):
                    #print("non existant")
                    ListeArcs.append(edge)
                    # Ajouter un arc en mappant les noeud du dico
                    H.add_edge(dico[source],dico[target],edge_type=arc)
        # Renvoyer nouveauGraphe
        return(H)
    
    
    # Renvoie le tuple avec les signes inversÃ©s
    def inversionTuple(self,tuple):
        inversion={}
        inversion["+"]="-"
        inversion["-"]="+"
        TupleInverse=""
        for sousTuple in tuple.split(","):
            node=sousTuple.split(" ")[0]
            signe=inversion[sousTuple.split(" ")[1]]
            TupleInverse=TupleInverse+","+node+" "+signe
        # Enlever l'entÃªte
        TupleInverse=TupleInverse[1:len(TupleInverse)+1]
        #print(TupleInverse)
        return(TupleInverse)
            
    
    # Renvoie le signe du noeud du tuple connectant le noeud au tuple initialement
    def ArcOriginel(self,TuplePred,noeudSource,grapheOriginel):
        noeudTarget=noeudSource.split(" ")[0]+" +"
        # Pour chaque noeud du Tuple    
        for sousTuple in TuplePred.split(","):
            node=sousTuple.split(" ")[0]+" +"
            signe=sousTuple.split(" ")[1]
            if(grapheOriginel.has_edge(node,noeudTarget)):
                return(signe)
            # S'il existe un arc entre ce noeud et le noeudSource
                # Renvoyer signe
    
    # Fonction de generation des Tuples
    def generationTuple(self,predecesseur, noeudSource, arc,grapheOriginel):
        #print("Fusion de "+predecesseur+" avec "+noeudSource)
        if(arc=="1"):
            arc="+"
        elif(arc=="-1"):
            arc="-"
        else:
            print("erreur d'arc : "+str(arc))
        if(arc=="-"):
            # Noeud source inverse
            #print("inversion")
            noeudSource=self.inversionTuple(noeudSource)
        # Gestion des cas de doubles inhibition
        SigneLastNode=self.ArcOriginel(predecesseur,noeudSource,grapheOriginel)
        #print(SigneLastNode)
        #if(SigneLastNode=="-"):
            #print("avant : "+noeudSource)
            #noeudSource=inversionTuple(noeudSource)
            #print("apres : "+noeudSource)
        nouveauTuple=predecesseur+","+noeudSource
        return(nouveauTuple)

    #fonction effectuant la réduction
    def compaction(self,input,out1,out2,out3):
        file=open(input,"r")
        data=file.readlines()
        file.close()
        #print(data) 
        G=nx.MultiDiGraph()
        GOrigine=nx.MultiDiGraph()
        #print(data)
        separateur="\t"
        node_type={}
        for row in data:
            if (len(row.split(separateur)) == 3):
                sourceOrigine=row.split(separateur)[0]
                source=row.split(separateur)[0]+" +"
                modele=row.split(separateur)[1]
                targetOrigine=row.split(separateur)[2].split("\n")[0].split("\r")[0]
                target=row.split(separateur)[2].split("\n")[0].split("\r")[0]+" +"
                #G.add_edge(source,target,edge_type=str(modele))
                if(modele=="inhibitor" or modele=="-1"):
                    G.add_edge(source,target,edge_type="-1")
                    GOrigine.add_edge(sourceOrigine,targetOrigine,edge_type="-1")
                else:
                    G.add_edge(source,target,edge_type="1")
                    GOrigine.add_edge(sourceOrigine,targetOrigine,edge_type="1")
                
        #print("graphe initial de "+str(len(G.nodes()))+" nodes")
        #print("graphe initial de "+str(len(G.edges()))+" arcs")
        Copie=G.copy()
        nbreTuples=len(G.nodes())
        nbreArcs=len(G.edges())
        nouveauNbreTuples=0
        nouveauNbreArcs=0
        
        NxNombreTuplesGlobal=0
        NombreTuplesGlobal=nbreTuples
        NxNombreArcsGlobal=0
        NombreArcsGlobal=len(G.edges())
        listeIsole=[]
        
        print("graph with "+str(len(G.nodes()))+" nodes and "+str(len(G.edges()))+" edges")
        while(NxNombreTuplesGlobal!=NombreTuplesGlobal or NombreArcsGlobal!=NxNombreArcsGlobal):
            #print("cycle "+str(NombreTuplesGlobal)) 
            NxNombreTuplesGlobal=NombreTuplesGlobal
            NxNombreArcsGlobal=NombreArcsGlobal
            # REDUCTION SUR LA COHERENCE
            suppression=[]
            # Tant que le nbre de Tuples varie
            while(nbreTuples!=nouveauNbreTuples):
                nbreTuples=nouveauNbreTuples
                # Reinitialisation des fusions
                Fusion=[]
                Dico={}
                InversionArc=[]
                for node in G.nodes():
                    Dico[node]=node
                # Pour chaque Noeud
                for node in G.nodes():
                
                    predecesseurs=G.predecessors(node)
                    #print (predecesseurs)
                    # Si nbre Predecesseur == 1 ET predec ne fusionne pas ET node ne fusionne pas
                    if(len(predecesseurs)==1 and predecesseurs[0] not in Fusion and node not in Fusion and len(G[predecesseurs[0]][node])==1 and self.FusionPossible(predecesseurs[0],node, G, G[predecesseurs[0]][node][0]['edge_type'])):
                        

                    
                        #print(node+ " fusion avec "+predecesseurs[0])
                        # Si le pred et noeud partagent 2 arcs diffÃ©rent
                        #print(node+" avec "+predecesseurs[0])
                        Fusion.append(node)
                        Fusion.append(predecesseurs[0])
                        #print(G[predecesseurs[0]][node])
                        NouveauTuple=self.generationTuple(predecesseurs[0], node,G[predecesseurs[0]][node][0]['edge_type'],Copie)
                        Dico[node]=NouveauTuple
                        Dico[predecesseurs[0]]=NouveauTuple
                        if(G[predecesseurs[0]][node][0]['edge_type']=="-1"):
                            InversionArc.append(node)
                        # Fusionner(noeud, pred, arc)
                # Mise Ã  jour du nbre de Tuples    
                G=self.FusionNodes(G,Dico,InversionArc)
            
                G.remove_edges_from(G.selfloop_edges())
                nouveauNbreTuples=len(G.nodes())
        
            
            #print("reduction par perfection")
            suppression=[]
            reduction=False
            rename={}
            # RÃ©initialiser la liste des noeuds Ã  compacter
            Compacte=[]
            # Pour chaque noeud
            for node in G.nodes():
                
                rename[node]=node
                # rÃ©cupÃ©rer liste prÃ©decesseurs
                ListePredecesseurs=G.predecessors(node)
                PredecesseursFusionnables=self.IdentificationPredecessorsFusionnables(G,ListePredecesseurs,Compacte)
                
                # Si plus d'un prÃ©dÃ©cesseur fusionnable
                if(len(PredecesseursFusionnables)>1):
                    # FUsion de ces noeuds
                    # CrÃ©er un nom de tuple commun et mettre en dico rename
                    reduction=True
                    NouveauTuple=self.FusionTuples(PredecesseursFusionnables,node,G)
                    
                    G.add_edge(NouveauTuple,node,edge_type="1")
                    # Stocker les autres noeud en suppression
                    for sousNode in PredecesseursFusionnables:
                        suppression.append(sousNode)
            G.remove_nodes_from(suppression)
        
            Copie=G.copy()
            nbreTuples=len(G.nodes())
            nbreArcs=len(G.edges())
            nouveauNbreTuples=0
            nouveauNbreArcs=0
        
            nbreNoeudsCycle=0
            NxnbreNoeudsReduction=nbreTuples
            listeConsistent=[]
        
        
            while(nbreNoeudsCycle!=NxnbreNoeudsReduction):
                nbreNoeudsCycle=len(G.nodes())
                suppression=[]
                rename={}
                # RÃ©initialiser la liste des noeuds Ã  compacter
                Compacte=[]
                # Pour chaque noeud
                for node in G.nodes():
                    rename[node]=node
                for node in G.nodes():
                    successeur=G.successors(node)
                    predecesseur=G.predecessors(node)
                    # Si noeud sans predecesseur, 1 successeur et un seul signe entre les 2
                    if(len(successeur)==1 and len(predecesseur)==0 and len(G[node][successeur[0]])==1):
                        tete=successeur[0].split(",")[0]
                        if(tete not in listeConsistent):
                            listeConsistent.append(tete)
                        suppression.append(node)
                        nouveauTuple=node
                        # Si arc inhibiteur => Inversion du tuple
                        if(G[node][successeur[0]][0]['edge_type']=="-1"):
                            nouveauTuple=self.InversionTuple(nouveauTuple)
                
                        rename[successeur[0]]=rename[successeur[0]]+","+nouveauTuple
                G.remove_nodes_from(suppression)
                NxnbreNoeudsReduction=len(G.nodes())
                G=nx.relabel_nodes(G,rename)
                #listeConsistent
                
            DicoNodes={}
            DicoInverse={}
            nbreNode=0
            for node in G.nodes(): 
                DicoInverse[node]="node"+str(nbreNode)
                DicoNodes[DicoInverse[node]]=node
                nbreNode=nbreNode+1
                
            Copie=nx.MultiDiGraph()
            Copie.add_nodes_from(G.nodes())
            #print("reduction des arcs")
            for edge in G.edges():
                poidsActivation=0
                poidsInhibition=0
                source=edge[0]
                target=edge[1]
                if(len(source.split("\"")) > 1 ):
                    source=source.split("\"")[1]
                if(len(target.split("\"")) > 1 ):
                    target=target.split("\"")[1]
                for sousTuple1 in source.split(","):
                    # Pour chaque sousNoeud de tuple2
                    for sousTuple2 in target.split(","):
                        node1=sousTuple1.split(" ")[0]
                        signeSource=sousTuple1.split(" ")[1]
                        #print(sousTuple1+ " to "+node1+" "+signeSource)
                        node2=sousTuple2.split(" ")[0]
                        if(GOrigine.has_edge(node1,node2)):
                            
                            for edgeOrigine in GOrigine[node1][node2]:
                                arc=(GOrigine[node1][node2][edgeOrigine]['edge_type'])
                            #    print(node1+" to "+node2+" "+arc)
                                if(arc == "1"):
                                    if(signeSource=="+"):
                                        poidsActivation=poidsActivation+1
                                    else:
                                        poidsInhibition=poidsInhibition+1
                                elif(arc=="-1"):
                                    if(signeSource=="+"):
                                        poidsInhibition=poidsInhibition+1
                                    else:
                                        poidsActivation=poidsActivation+1
        
                poidsMin=min(poidsActivation,poidsInhibition)  
                retour=True
        #        poidsMin=0 
                if(poidsActivation-poidsMin >0):
                    Copie.add_edge(source,target,edge_type="1")  
                if(poidsInhibition-poidsMin >0):
                    Copie.add_edge(source,target,edge_type="-1") 
                # Cas isolement d'un arc 
                if(poidsActivation==poidsInhibition and poidsActivation !=0):
                    # Stocker tuple pour prÃ©ciser target : consistent + imperfect
                    tete=edge[1].split(",")[0]
                    if(tete not in listeIsole):
                        listeIsole.append(tete)
        
        #    print(Copie.edges())
        #    print(G.edges())
            G=Copie
        
            nouveauNbreTuples=len(G.nodes())
            nouveauNbreArcs=len(G.edges())
            NombreArcsGlobal=nouveauNbreArcs
            NombreTuplesGlobal=nouveauNbreTuples
            print("Reduction to "+str(len(G.nodes()))+" nodes and "+str(len(G.edges()))+" edges")
        
        # Listing des arcs
        listeArcs=[]
        for i in G.edges():
            #print(i)
            source=i[0]
            target=i[1]
            for arc in (G[source][target]):
                edge="\""+source+"\""+"\t"+str(G[source][target][arc]['edge_type'])+"\t"+"\""+target+"\""
                if(edge not in listeArcs):
                    listeArcs.append(edge)
            if(source==target):
                print("Frappe "+source+" => "+str(G[source][target][arc]['edge_type']))
        
        
        file=open(out1,"w") #ecriture format sif
        for i in listeArcs:
            file.write(i+"\n")
        #    print(G[
        file.close()
        
        
        # Ecriture du Dictionnaire
        file=open(out2,"w")
        for node in G.nodes():
            file.write("\""+node+"\" : "+DicoInverse[node]+"\n")
        
        file.close()
        
        
        
        
        fileOutput=open(out3,"w")
        # Ecriture du graphe Mis en forme
        NodeUtilise=G.nodes()
        for edge in G.edges():
            poidsActivation=0
            poidsInhibition=0
            source=edge[0]
            target=edge[1]
            if(len(source.split("\"")) > 1 ):
                source=source.split("\"")[1]
            if(len(target.split("\"")) > 1 ):
                    target=target.split("\"")[1]
            for sousTuple1 in source.split(","):
                # Pour chaque sousNoeud de tuple2
                for sousTuple2 in target.split(","):
                    node1=sousTuple1.split(" ")[0]
                    signeSource=sousTuple1.split(" ")[1]
                    #print(sousTuple1+ " to "+node1+" "+signeSource)
                    node2=sousTuple2.split(" ")[0]
                    # print("test "+node1+" to "+node2)
                    if(GOrigine.has_edge(node1,node2)):
                        for edgeOrigine in GOrigine[node1][node2]:
                            arc=(GOrigine[node1][node2][edgeOrigine]['edge_type'])
                        #    print(node1+" to "+node2+" "+arc)
                            if(arc == "1"):
                                if(signeSource=="+"):
                                    poidsActivation=poidsActivation+1
                                else:
                                    poidsInhibition=poidsInhibition+1
                            elif(arc=="-1"):
                                if(signeSource=="+"):
                                    poidsInhibition=poidsInhibition+1
                                else:
                                    poidsActivation=poidsActivation+1
            poidsMin=min(poidsActivation,poidsInhibition)
            if(poidsActivation-poidsMin!=0):
                if(edge[0] in NodeUtilise):
                    NodeUtilise.remove(edge[0])
                if(edge[1] in NodeUtilise):
                    NodeUtilise.remove(edge[1])
        
        
            #print("("+source+ ","+target+",1,"+str(poidsActivation)+").")
                fileOutput.write("edge("+DicoInverse[edge[0]]+ ","+DicoInverse[edge[1]]+",1,"+str(poidsActivation-poidsMin)+").\n")
            if(poidsInhibition-poidsMin!=0):
                if(edge[0] in NodeUtilise):
                    NodeUtilise.remove(edge[0])
                if(edge[1] in NodeUtilise):
                    NodeUtilise.remove(edge[1])
                #print("("+source+ ","+target+",-1,"+str(poidsInhibition)+").")
                fileOutput.write("edge("+DicoInverse[edge[0]]+ ","+DicoInverse[edge[1]]+",-1,"+str(poidsInhibition-poidsMin)+").\n")
        
            # Checker si besoin d'afficher les composants prÃ©-identifiÃ©s : En liste "NodeUtilise"    
        
            
        # Recuperation des cibles imparfaites
        for component in listeIsole:
            #print(component)
            for node in G.nodes():
                if(node.find(component)!=-1):
                    fileOutput.write("imperfectcoloring("+DicoInverse[node]+").")
                    fileOutput.write("consistentTarget("+DicoInverse[node]+").\n")                          
        
        
        
        
        # Recuperation des cibles cohÃ©rentes
        for component in listeConsistent:
            #print(component)
            for node in G.nodes():
                if(node.find(component)!=-1):    
                    fileOutput.write("consistentTarget("+DicoInverse[node]+").\n")                          
        
        
        
        
        
        
        fileOutput.close()
        

    
    def InversionTupleID(self,tuple):
    
        signe=tuple.split(" ")[1]
        node=tuple.split(" ")[0]
        if(signe=="+"):
            return (node+" -")
        else:
            return(node+" +")
    
    def FusionTuplesID(self,node1,node2,signe):
        nouveauTuple=node1
        #print(node1)
        #print(node2)
        #print(signe)
        for sousTuple in node2.split(","):
            #print(sousTuple)
            tupleCalcule=sousTuple
            if(signe == -1):
                #print("inversion")
                tupleCalcule=self.InversionTupleID(tupleCalcule)
            
            #print(tupleCalcule)	
            nouveauTuple=nouveauTuple+","+tupleCalcule
        #nouveauTuple=(nouveauTuple[1:len(nouveauTuple)+1])
        #print(nouveauTuple)
        return(nouveauTuple) 	
    
    
    # Renvoie a chaque fin d'appel un tuple contenant le nouveau noeud s'il est dans la liste
    # Appel nouveauTuple=rechercheTuple(G,node,tupleCourant,listeNodes,signe):
        # G : graphe
        # Noeud : noeud en cours d'exploration
        # tupleCourange : tuple en cours de construction
        # ListeNodes : liste des noeuds
        # signe = signe de la correlation
        
    def rechercheTupleID(self,G,node,tupleCourant,listeNodes,signe):
        # Pour le noeud courant
        #print(node)
        signeInitial=signe
        listeVoisins=G.neighbors(node)
        for voisin in listeVoisins:
            signe=signeInitial
            if(voisin in listeNodes):	
                
                listeNodes.remove(voisin)
                #print(node+" "+voisin)
                signeInteraction=G[voisin][node]['edge_type']
                # Calcul du signe l'Ã©lÃ©ment selon signe
                #print(signe)
                signe=signe*signeInteraction
            #print(voisin+" : "+str(signe)+" "+str(signeInteraction)+" from "+node)
                element=voisin
                # Si l'interaction est nÃ©gative
                #print("fusion :"+voisin+" : "+str(signe))
                tupleCourant=self.FusionTuplesID(tupleCourant,voisin,signe)
                # Relancer algo sur noeud ajoutÃ©
                tupleCourant=self.rechercheTupleID(G,voisin,tupleCourant,listeNodes,signe)
                
        return tupleCourant
    
    
    
    
    # Fonction effectuant l'identifcation des colorations
    def identificationColor(self,path):
        Dico={}
        DicoInverse={}
        name = self.grapheLoc2[self.graph_3.currentRow()]
        input=os.path.splitext(name)[0]+"-reduced-hash.txt"
        # Charger Dico des nodes
        
        file=open(input,'r')
        data=file.readlines()
        file.close()    
        for i in data:
            #print(i)
            node=i.split(" : ")[0].split("\"")[1]
            #print(node)
            conversion=i.split(" : ")[1].split("\n")[0]
            #print(conversion)
            Dico[conversion]=node
            DicoInverse[node]=conversion
        
        # Charger le graphe
        
        
        # convertir le graphe en graphe networkX
        file=open(path,'r')
        data=file.readlines()
        file.close()
        # print(data) 
        G=nx.Graph()
           
        # Pour chaque ligne :
        for i in data:
            # Identifier source & target : convertir
            source=Dico[i.split("(")[1].split(",")[0]]
            target=Dico[i.split(",")[1].split(")")[0]]
        
            # Identifier le type de correlation : signe de l'arc
            # cas positif
            if(len(i.split("Positif"))==2):
                G.add_edge(source,target,edge_type=1)
            # cas nÃ©gatif
            else:
                G.add_edge(source,target,edge_type=-1)
        
        
        
        listeTuples=[]
        
        listeNodes=[]
        
        #Identifier les noeuds hors graphe de corrÃ©lation : tuples simples ou non listÃ©s
        
        for i in DicoInverse.keys():
            if(i in G.nodes()):
                listeNodes.append(i)
            else:
                listeTuples.append(i)
            
    
        # Pour chaque noeud :
            # Fonction recursive d'exploration : parametre : graphe, tuple actuel, 
        while (len(listeNodes)!=0):
            for node in listeNodes:
                #print(node)
                nouveauTuple=node
                listeNodes.remove(node)
                # Appel fonction de recherche
                nouveauTuple=self.rechercheTupleID(G,node,nouveauTuple,listeNodes,1)
                listeTuples.append(nouveauTuple)
        
            
        # Affichage des tuples
        #for tuple in listeTuples:
        #	print(tuple) 
        file=open(os.path.splitext(self.grapheLoc2[self.graph_3.currentRow()])[0]+"-composants.csv",'w')
        for line in listeTuples:
            file.write(line+'\n')
        file.close()
        file=open(os.path.splitext(self.grapheLoc2[self.graph_3.currentRow()])[0]+"-coloration-table.csv",'w')
        tableTuple=[]
        for i in range(len(listeTuples)):
            tuple=listeTuples[i]
            tuple=tuple.split(",")
            for node in tuple:
                if(node.split(" ")[1] == '-'):
                    tableTuple.append([node.split(" ")[0],i,-1])
                else:
                    tableTuple.append([node.split(" ")[0],i,1])
        
        self.table=tableTuple
        file.write("\"Node\", \"Coloration\", \"Signe\""+"\n")
        for tuple in tableTuple:
            file.write("\""+tuple[0]+"\", "+str(tuple[1])+", "+str(tuple[2])+"\n")
        file.close()
    
    #affichage des colorations
    def colorGraphe(self,tableTuple):
        if not self.isRunning():
            self.alerte()
        else:
            cy = CyRestClient()
            net1 = cy.network.create_from(self.grapheLoc2[self.graph_3.currentRow()])
            net1.create_node_column("composant", data_type='Integer',is_immutable=False)
            net1.create_node_column("signe",data_type='Integer',is_immutable=False)
            table=net1.get_node_table()
            nodeTable={}
            nodes=net1.get_nodes()
            for node in nodes:
                nodeTable[net1.get_node_value(node)['name']]=node
            
            
            
            for line in tableTuple:
            
                table.set_value(nodeTable[line[0]],"composant",line[1])
                table.set_value(nodeTable[line[0]],"signe",line[2])
                
            net1.update_node_table(table,network_key_col='name', data_key_col='name')
            style1 = cy.style.create('sample_style1')
        
            points = [{
            'value': '0.0',
            'lesser':'blue',
            'equal':'blue',
            'greater': 'blue'
            },{
            'value': '1.0',
            'lesser':'red',
            'equal':'red',
            'greater': 'red'
            }]
            
            points[1]["value"]=self.nbComposantes(tableTuple)-1
            style1.create_continuous_mapping(column='composant', col_type='Integer', vp='NODE_FILL_COLOR',points=points)
            cy.style.apply(style1,net1)
            cy.layout.apply(name='organic',network=net1)
            self.doneC()
    
    #fonction qui transforme la sortie du script ASP afin qu'il soit compréhensible pour le script d'identification
    def processASP(self,path):
        file=open(path,'r')
        lines=file.readlines()

        
        correle=self.lastAns(lines)
        correleSep=correle.split()
        file.close()
        name=os.path.splitext(path)[0]+"-processed"+os.path.splitext(path)[1]
        file=open(name,'w')
        for line in correleSep:
            file.write(line+"\n")
        file.close()

    #renvoie le nombre de composants obtenus
    def nbComposantes(self,table):
        nb=0
        for line in table:
            if (line[1]>nb):
                nb=line[1]
        return (nb+1)
    
    #renvoie la dernière réponse (optimum ou non) du script ASP    
    def lastAns(self,lines):
        res=0
        line=lines[0]
        for k in range(len(lines)):
            if( "Answer" in lines[k]):
                res=res+1
                line=lines[k+1]
        return line
    
    #extraction des n composants et export des fichiers correspondants
    def nComposantes(self,tableTuple):
        n=self.nbComposantes(tableTuple)
        listeComposante=[[] for _ in range(n)]
        for line in tableTuple:
            listeComposante[line[1]-1].append(line[0])
        file=open(self.grapheLoc2[self.graph_3.currentRow()],'r')
        base=file.readlines()
        file.close()
        graphesComposantes=[]
        for compo in listeComposante:
            current=[]
            if (len(compo)==1):
                current.append(compo[0])
            else:
                for line in base:
                    
                    lineSliced=line[:-1]
                    edge=lineSliced.split("\t")
                    
                    if (edge[0] in compo and edge[2] in compo ):
                        current.append(line)
            graphesComposantes.append(current)
        name = self.grapheLoc2[self.graph_3.currentRow()]
        directory=os.path.dirname(name)+"//Composants_"+os.path.basename(os.path.splitext(name)[0])
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i in range(len(graphesComposantes)):
            strOpen=directory+"//composant"+str(i+1)+".sif"
            file=open(strOpen,'w')
            for line in graphesComposantes[i]:
                file.write(line)
                file.write("\n")
    
    #Permet de sélectionner le dossier pour le jeu de données               
    def loadPatientsDatasForSimil(self):
        self.patientsDatasForSimilDirURL = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Directory", dir_path + '\\')
        self.donnees.item(0).setText(self.patientsDatasForSimilDirURL)
    
    #Permet de sélectionner le fichier clinique pour le calcul
    def loadClinicDatasForSimil(self):
        self.clinicDatasForSimilFileURL = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', dir_path + '\\')
        self.clinique.item(0).setText(self.clinicDatasForSimilFileURL[0])
        
    def loadComponentsFileForSimil(self):
        self.componentsFileURL = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', dir_path + '\\', '*.csv')
        self.csv.item(0).setText(self.componentsFileURL[0])
        
    def loadGrapheGenesFileForSimil(self):
        self.grapheGenesFileURL = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', dir_path + '\\', '*.sif')
        self.sif.item(0).setText(self.grapheGenesFileURL[0])
        
    #Définition de la fonction MSComputing en accord avec le script de Bertrand
    def computing(self,p,c,f):
        #Chargement du fichier test
        test=open(f,'w')
        
        nom=p.split("\\")
        nom=nom[-1]
        ligne=nom
        
        # Charger liste des Observations
        Observations={}
        
        file=open(p,"r")
        data=file.readlines()
        file.close()
        
        
        
        for i in data:
            #print(i.split(" "))
            node=i.split(" ")[0]
            signe=i.split(" ")[2].split("\n")[0]
            
                #print(node+" "+signe)
            Observations[node]=float(signe)*1.0
        
        # Chargement des Tuples
        file=open(c,"r")
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
        
        #print(ligne)
        test.write(ligne)
        test.close()
        
        
    def grapheSimilarite(self, rfile):
        resultFile = open(rfile,'r')
        
        strings=[]
        
        lines=resultFile.readlines()
        for line in lines:
            temp=line.split(' ')
            
            if temp[0]==self.nomPatient.toPlainText():
                strings=temp.copy()
            print(temp)
        if strings != []:
            patientName = strings[0]
            similVector = strings[1:-1]
            resultFile.close()
            
            fileComponents = open(self.componentsFileURL[0], 'r')
            hashMap = {}
            n = 0
            listeComposantes = []
            for line in fileComponents:
                if line != '\n':
                    composante = line.split(',')
                    for i in range (len(composante)):
                        if (i == len(composante) - 1):
                            composante[i] = composante[i][:-3]
                        else:
                            composante[i] = composante[i][:-2]
                    listeComposantes.append(composante)
                    n += 1
                    hashMap["NOEUD" + str(n)] = similVector[n-1]
            fileComponents.close()
                    
            fileGraphe = open(self.grapheGenesFileURL[0],'r')
            base = fileGraphe.readlines()
            fileGraphe.close()
            
            resultFile = open(rfile,'r')
            strings=[]
            
            lines=resultFile.readlines()
            for line in lines:
                temp=line.split(' ')
                
                if temp[0]==self.nomPatient.toPlainText():
                    strings=temp.copy()
                    
            patientName = strings[0]
            similVector = strings[1:-1]
            resultFile.close()
            
            grapheComposantes=[]
            nodeAlone = [True for _ in range(n)]
            
            for i in range(n):
                composante1 = listeComposantes[i]
                for j in range(i+1,n):
                    composante2 = listeComposantes[j]
                    for line in base:
                        lineSliced = line[:-1]
                        edge = lineSliced.split("\t")
                            
                        if (edge[0] in composante1 and edge[2] in composante2):
                            arc = "NOEUD" + str(i+1) + "\t" + edge[1] + "\t" + "NOEUD" + str(j+1) + "\n"
                            if arc not in grapheComposantes:
                                grapheComposantes.append(arc)
                                nodeAlone[i] = False
                                nodeAlone[j] = False
                                
                        if (edge[2] in composante1 and edge[0] in composante2):
                            arc = "NOEUD" + str(j+1) + "\t" + edge[1] + "\t" + "NOEUD" + str(i+1) + "\n"
                            if arc not in grapheComposantes:
                                grapheComposantes.append(arc)
                                nodeAlone[i] = False
                                nodeAlone[j] = False
            grapheComposantes = sorted(grapheComposantes, key = lambda composante: int(composante.split("\t")[0][5:]))
                                    
            dir = os.path.dirname(rfile)
            
            
            fileGrapheSimilURL = dir + "\\Graphe_" + patientName + ".sif"
            fileGrapheSimil = open(fileGrapheSimilURL, 'w')
            print(fileGrapheSimilURL)
            for arc in grapheComposantes:
                fileGrapheSimil.write(arc)
                
            for i in range(n):
                if nodeAlone[i] == True:
                    fileGrapheSimil.write("NOEUD" + str(i+1) + "\n")
                
            fileGrapheSimil.close()
            
            return [fileGrapheSimilURL, hashMap]
        else:
            self.wrongPatient()
            return ["error",{}]

    def displayGrapheSimilarite(self, grapheURL, hashMap):
        if not self.isRunning():
            self.alerte()
        else:
            cy = CyRestClient()            
            net1 = cy.network.create_from(grapheURL)
            net1.create_node_column("Similarite", data_type='Integer',is_immutable=False)
            
            table=net1.get_node_table()
            nodes=net1.get_nodes()
            
            print (len(hashMap))
            
            for node in nodes:
                
                table.set_value(node, "Similarite", int(float(hashMap[net1.get_node_value(node)['name']])*100))
            
            net1.update_node_table(table,network_key_col='name', data_key_col='name')
            
            style1 = cy.style.create('sample_style1')
        
        
            mini=100
            maxi=0
            for node in nodes:
                
                a=int(float(hashMap[net1.get_node_value(node)['name']])*100)
                print(a)
                if a<mini:
                    mini=a
                if a>maxi:
                    maxi=a
        
        
            print(mini)
            print(maxi)
            points = [{
            'value': '0.0',
            'lesser':'red',
            'equal':'red',
            'greater': 'red'
            },{
            'value': '100',
            'lesser': 'green',
            'equal': 'green',
            'greater': 'green'}]
            
            points[0]["value"]=mini
            points[1]["value"]=maxi
            
            kv_pair = {
                '-1': 'T',
                '1': 'Arrow'
            }
            style1.create_discrete_mapping(column='interaction',col_type='String',vp='EDGE_SOURCE_ARROW_SHAPE', mappings=kv_pair)
            
            style1.create_continuous_mapping(column='Similarite',col_type='Integer',vp='NODE_FILL_COLOR',points=points)

            cy.style.apply(style1,net1)
            cy.layout.apply(name='organic',network=net1)
            
            
    def modifyDisplayGraphState(self):
        self.afficherGraph.setEnabled(self.graphSimi.isChecked())
        self.nomPatient.setEnabled(self.graphSimi.isChecked())
        if (not self.graphSimi.isChecked()):
            self.afficherGraph.setChecked(False)
            self.nomPatient.setText("")
        
    def modifyButtonClinique(self):
        self.buttonClinique.setEnabled(not self.dataPred.isChecked())
        
    def similIsRunnable(self):
        cond1 = self.donnees.item(0).text() != "" and self.donnees.item(0).text() != "Jeu_de_données" and self.clinique.item(0).text() != "" and self.clinique.item(0).text() != "Fichier_clinique" and self.csv.item(0).text() != "" and self.csv.item(0).text() != "Fichier_CSV"
        cond2 = self.donnees.item(0).text() != "" and self.donnees.item(0).text() != "Jeu_de_données" and self.dataPred.isChecked() and self.csv.item(0).text() != "" and self.csv.item(0).text() != "Fichier_CSV"
        cond3 = not self.graphSimi.isChecked()
        cond4 = (self.sif.item(0).text() != "" and self.sif.item(0).text() != "Fichier_Sif" and self.graphSimi.isChecked())
        cond5 = not (self.graphSimi.isChecked() and self.dataPred.isChecked())
        
        return (cond1 or cond2) and (cond3 or cond4) and cond5
    
    #Lance l'algorithme de calcul de similarité
    def runSimilAlgorithm(self):
        #On vérifie que l'utilisateur a bien sélectionné les informations demandées
        if self.similIsRunnable():
            rfile = os.path.dirname(self.patientsDatasForSimilDirURL) + '\\resultat_' + self.patientsDatasForSimilDirURL.split("/")[-1] + '.csv'
            results=open(rfile,'w')
            
            #Traitement des données
            print(rfile)
            for i in os.listdir(self.patientsDatasForSimilDirURL):
                
                #Ouverture du fichier et affichage
                f = open(self.patientsDatasForSimilDirURL + '\\' + i, 'r')
                nom=os.path.basename(f.name) 
                #print('dataPatientGSE19784HOVON65/'+nom)
                
                #Suppression des gènes "NA" en passant par le fichier test
                test = open(dir_path+'test','w')
                for line in f:
                    if re.search(r' = NA', line) is None:
                        #Écriture dans le fichier test
                        test.write(line)
                f.close()
                test.close()
                
                #On repasse le fichier texte dans le fichier de gènes (qui ne contient
                # alors plus de gènes NA)
                f = open(self.patientsDatasForSimilDirURL + '\\' + i,'w')
                test = open(dir_path + 'test','r')
                for line in test:
                    f.write(line)
                f.close()
                test.close()
                
                # Manque à éxecuter le script MSComputing (dont le résultat sera foutu dans test)
                # et faire des mini-traitements
                # puis faire pareil sur l'autre dossier
                pathtest=dir_path+ '\\' + 'testpython'
                self.computing(self.patientsDatasForSimilDirURL + '\\' + i,self.componentsFileURL[0],pathtest)
                ftest=open(pathtest,'r')
                
                clinique = ""
                if (not self.dataPred.isChecked()):
                    cl=open(self.clinicDatasForSimilFileURL[0],'r')
                    for line in cl:
                        if re.match(nom,line) is not None:
                            clinique=line.split(',')[1]
                    cl.close()
                
                for line in ftest:
                    if clinique!="CENSORED\n":
                        results.write(line)
                        if (not self.dataPred.isChecked()):
                            
                            results.write(" "+clinique)
                        else:
                            results.write("\n")
                ftest.close()
                
                

            results.close()
            os.remove(pathtest)
            
            if (self.graphSimi.isChecked()):
                print("Name: "+rfile)
                [fileURL, hashMap] = self.grapheSimilarite(rfile)
                if (self.afficherGraph.isChecked() and fileURL != "error"):
                    self.displayGrapheSimilarite(fileURL, hashMap)
            
            self.doneSimil()
        else:
            self.missingFiles()
            
            

        
    def main(self):
        print(self.table)
        self.show()
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pappl = Pappl()
    pappl.main()
    sys.exit(app.exec_())
