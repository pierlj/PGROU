# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface.ui'
#
# Created by: PyQt5 UI code generator 5.8.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowModality(QtCore.Qt.NonModal)
        Form.setEnabled(True)
        Form.resize(721, 493)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Iguane.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        Form.setWindowOpacity(1.0)
        Form.setStyleSheet("")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setStyleSheet("")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setStyleSheet("")
        self.tab.setObjectName("tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setMinimumSize(QtCore.QSize(0, 30))
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 1, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.launch = QtWidgets.QPushButton(self.tab)
        self.launch.setObjectName("launch")
        self.verticalLayout_3.addWidget(self.launch)
        self.gridLayout_3.addLayout(self.verticalLayout_3, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graph = QtWidgets.QListWidget(self.tab)
        self.graph.setObjectName("graph")
        self.horizontalLayout.addWidget(self.graph)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.load = QtWidgets.QPushButton(self.tab)
        self.load.setObjectName("load")
        self.verticalLayout_5.addWidget(self.load)
        self.display = QtWidgets.QPushButton(self.tab)
        self.display.setEnabled(True)
        self.display.setObjectName("display")
        self.verticalLayout_5.addWidget(self.display)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.gridLayout_3.addLayout(self.horizontalLayout, 2, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem1 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setMinimumSize(QtCore.QSize(0, 30))
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.graph_2 = QtWidgets.QListWidget(self.tab_2)
        self.graph_2.setObjectName("graph_2")
        self.horizontalLayout_2.addWidget(self.graph_2)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.load_2 = QtWidgets.QPushButton(self.tab_2)
        self.load_2.setObjectName("load_2")
        self.verticalLayout_13.addWidget(self.load_2)
        self.reduc = QtWidgets.QPushButton(self.tab_2)
        self.reduc.setEnabled(True)
        self.reduc.setObjectName("reduc")
        self.verticalLayout_13.addWidget(self.reduc)
        self.horizontalLayout_2.addLayout(self.verticalLayout_13)
        self.verticalLayout_6.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_6.addLayout(self.verticalLayout_6)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setEnabled(True)
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.tab_3)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem2 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem2)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setMinimumSize(QtCore.QSize(0, 30))
        self.label_3.setObjectName("label_3")
        self.verticalLayout_7.addWidget(self.label_3)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.time = QtWidgets.QSpinBox(self.tab_3)
        self.time.setMaximum(86400)
        self.time.setObjectName("time")
        self.horizontalLayout_3.addWidget(self.time)
        self.verticalLayout_7.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.graph_3 = QtWidgets.QListWidget(self.tab_3)
        self.graph_3.setObjectName("graph_3")
        self.horizontalLayout_4.addWidget(self.graph_3)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.load_3 = QtWidgets.QPushButton(self.tab_3)
        self.load_3.setObjectName("load_3")
        self.verticalLayout_15.addWidget(self.load_3)
        self.color = QtWidgets.QPushButton(self.tab_3)
        self.color.setEnabled(True)
        self.color.setObjectName("color")
        self.verticalLayout_15.addWidget(self.color)
        self.color_1 = QtWidgets.QPushButton(self.tab_3)
        self.color_1.setObjectName("color_1")
        self.verticalLayout_15.addWidget(self.color_1)
        self.nCompo = QtWidgets.QPushButton(self.tab_3)
        self.nCompo.setObjectName("nCompo")
        self.verticalLayout_15.addWidget(self.nCompo)
        self.horizontalLayout_4.addLayout(self.verticalLayout_15)
        self.verticalLayout_7.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_7.addLayout(self.verticalLayout_7)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.afficherGraph = QtWidgets.QCheckBox(self.tab_4)
        self.afficherGraph.setEnabled(False)
        self.afficherGraph.setObjectName("afficherGraph")
        self.gridLayout_2.addWidget(self.afficherGraph, 14, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_4)
        self.label_5.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setMinimumSize(QtCore.QSize(0, 40))
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 4)
        self.label_14 = QtWidgets.QLabel(self.tab_4)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 8, 1, 1, 1)
        self.calcul = QtWidgets.QPushButton(self.tab_4)
        self.calcul.setMinimumSize(QtCore.QSize(125, 0))
        self.calcul.setMaximumSize(QtCore.QSize(200, 16777215))
        self.calcul.setObjectName("calcul")
        self.gridLayout_2.addWidget(self.calcul, 17, 1, 1, 3, QtCore.Qt.AlignHCenter)
        self.buttonClinique = QtWidgets.QPushButton(self.tab_4)
        self.buttonClinique.setMaximumSize(QtCore.QSize(30, 20))
        self.buttonClinique.setObjectName("buttonClinique")
        self.gridLayout_2.addWidget(self.buttonClinique, 4, 3, 1, 1)
        self.buttonDonnee = QtWidgets.QPushButton(self.tab_4)
        self.buttonDonnee.setMaximumSize(QtCore.QSize(30, 20))
        self.buttonDonnee.setObjectName("buttonDonnee")
        self.gridLayout_2.addWidget(self.buttonDonnee, 2, 3, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.tab_4)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 1, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout_2.addItem(spacerItem3, 7, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout_2.addItem(spacerItem4, 10, 1, 1, 1)
        self.sif = QtWidgets.QListWidget(self.tab_4)
        self.sif.setMaximumSize(QtCore.QSize(16777215, 20))
        self.sif.setObjectName("sif")
        item = QtWidgets.QListWidgetItem()
        self.sif.addItem(item)
        self.gridLayout_2.addWidget(self.sif, 13, 1, 1, 2)
        self.graphSimi = QtWidgets.QCheckBox(self.tab_4)
        self.graphSimi.setObjectName("graphSimi")
        self.gridLayout_2.addWidget(self.graphSimi, 11, 1, 1, 2)
        self.dataPred = QtWidgets.QCheckBox(self.tab_4)
        self.dataPred.setObjectName("dataPred")
        self.gridLayout_2.addWidget(self.dataPred, 9, 1, 1, 2)
        self.nomPatient = QtWidgets.QTextEdit(self.tab_4)
        self.nomPatient.setEnabled(False)
        self.nomPatient.setMinimumSize(QtCore.QSize(400, 23))
        self.nomPatient.setMaximumSize(QtCore.QSize(500, 23))
        self.nomPatient.setObjectName("nomPatient")
        self.gridLayout_2.addWidget(self.nomPatient, 14, 2, 1, 1)
        self.csv = QtWidgets.QListWidget(self.tab_4)
        self.csv.setMaximumSize(QtCore.QSize(16777215, 20))
        self.csv.setObjectName("csv")
        item = QtWidgets.QListWidgetItem()
        self.csv.addItem(item)
        self.gridLayout_2.addWidget(self.csv, 6, 1, 1, 2)
        self.label_16 = QtWidgets.QLabel(self.tab_4)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 3, 1, 1, 2)
        self.buttonSif = QtWidgets.QPushButton(self.tab_4)
        self.buttonSif.setMaximumSize(QtCore.QSize(30, 20))
        self.buttonSif.setObjectName("buttonSif")
        self.gridLayout_2.addWidget(self.buttonSif, 13, 3, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.tab_4)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 5, 1, 1, 2)
        self.clinique = QtWidgets.QListWidget(self.tab_4)
        self.clinique.setMinimumSize(QtCore.QSize(350, 0))
        self.clinique.setMaximumSize(QtCore.QSize(16777215, 20))
        self.clinique.setObjectName("clinique")
        item = QtWidgets.QListWidgetItem()
        self.clinique.addItem(item)
        self.gridLayout_2.addWidget(self.clinique, 4, 1, 1, 2)
        self.buttonCsv = QtWidgets.QPushButton(self.tab_4)
        self.buttonCsv.setMaximumSize(QtCore.QSize(30, 20))
        self.buttonCsv.setObjectName("buttonCsv")
        self.gridLayout_2.addWidget(self.buttonCsv, 6, 3, 1, 1)
        self.donnees = QtWidgets.QListWidget(self.tab_4)
        self.donnees.setEnabled(True)
        self.donnees.setMinimumSize(QtCore.QSize(350, 0))
        self.donnees.setMaximumSize(QtCore.QSize(16777215, 20))
        self.donnees.setObjectName("donnees")
        item = QtWidgets.QListWidgetItem()
        self.donnees.addItem(item)
        self.gridLayout_2.addWidget(self.donnees, 2, 1, 1, 2)
        self.label_18 = QtWidgets.QLabel(self.tab_4)
        self.label_18.setObjectName("label_18")
        self.gridLayout_2.addWidget(self.label_18, 12, 1, 1, 2)
        spacerItem5 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout_2.addItem(spacerItem5, 16, 1, 1, 2)
        self.verticalLayout_4.addLayout(self.gridLayout_2)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.tab_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem6 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem6)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.gridLayout.setObjectName("gridLayout")
        self.label_19 = QtWidgets.QLabel(self.tab_5)
        self.label_19.setObjectName("label_19")
        self.gridLayout.addWidget(self.label_19, 3, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.tab_5)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.gridLayout.addWidget(self.comboBox, 8, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.tab_5)
        self.label_11.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 12, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.tab_5)
        self.label_10.setMaximumSize(QtCore.QSize(55, 20))
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 11, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.tab_5)
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 10, 2, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.tab_5)
        self.pushButton.setMaximumSize(QtCore.QSize(70, 22))
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 8, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.tab_5)
        self.label_9.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 7, 0, 1, 4)
        self.listWidget = QtWidgets.QListWidget(self.tab_5)
        self.listWidget.setMinimumSize(QtCore.QSize(0, 220))
        self.listWidget.setMaximumSize(QtCore.QSize(400, 16777215))
        self.listWidget.setObjectName("listWidget")
        self.gridLayout.addWidget(self.listWidget, 9, 0, 4, 1)
        self.test = QtWidgets.QPushButton(self.tab_5)
        self.test.setMaximumSize(QtCore.QSize(75, 22))
        self.test.setObjectName("test")
        self.gridLayout.addWidget(self.test, 9, 2, 1, 1, QtCore.Qt.AlignHCenter)
        self.label_12 = QtWidgets.QLabel(self.tab_5)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 5, 2, 1, 1, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        spacerItem7 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem7, 0, 0, 1, 5)
        self.fichier = QtWidgets.QLabel(self.tab_5)
        self.fichier.setText("")
        self.fichier.setObjectName("fichier")
        self.gridLayout.addWidget(self.fichier, 10, 3, 1, 2)
        self.label_6 = QtWidgets.QLabel(self.tab_5)
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 1, 0, 1, 5)
        self.chercheFichier = QtWidgets.QPushButton(self.tab_5)
        self.chercheFichier.setMaximumSize(QtCore.QSize(30, 20))
        self.chercheFichier.setObjectName("chercheFichier")
        self.gridLayout.addWidget(self.chercheFichier, 4, 4, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.tab_5)
        self.label_8.setMinimumSize(QtCore.QSize(100, 0))
        self.label_8.setMaximumSize(QtCore.QSize(35, 20))
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 5, 0, 1, 1, QtCore.Qt.AlignRight)
        self.train = QtWidgets.QPushButton(self.tab_5)
        self.train.setEnabled(True)
        self.train.setMaximumSize(QtCore.QSize(70, 22))
        self.train.setSizeIncrement(QtCore.QSize(0, 0))
        self.train.setBaseSize(QtCore.QSize(0, 0))
        self.train.setObjectName("train")
        self.gridLayout.addWidget(self.train, 6, 0, 1, 1)
        self.taux = QtWidgets.QLabel(self.tab_5)
        self.taux.setText("")
        self.taux.setObjectName("taux")
        self.gridLayout.addWidget(self.taux, 5, 3, 1, 2)
        self.spinBox = QtWidgets.QSpinBox(self.tab_5)
        self.spinBox.setMaximumSize(QtCore.QSize(50, 20))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(45)
        self.spinBox.setProperty("value", 10)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 5, 1, 1, 1)
        self.nomFichier = QtWidgets.QListWidget(self.tab_5)
        self.nomFichier.setMaximumSize(QtCore.QSize(16777215, 20))
        self.nomFichier.setObjectName("nomFichier")
        self.gridLayout.addWidget(self.nomFichier, 4, 0, 1, 4)
        self.matrice = QtWidgets.QLabel(self.tab_5)
        self.matrice.setMinimumSize(QtCore.QSize(150, 60))
        self.matrice.setText("")
        self.matrice.setObjectName("matrice")
        self.gridLayout.addWidget(self.matrice, 12, 3, 1, 2)
        self.precision = QtWidgets.QLabel(self.tab_5)
        self.precision.setMaximumSize(QtCore.QSize(16777215, 20))
        self.precision.setText("")
        self.precision.setObjectName("precision")
        self.gridLayout.addWidget(self.precision, 11, 3, 1, 2)
        spacerItem8 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem8, 2, 0, 1, 5)
        self.horizontalLayout_5.addLayout(self.gridLayout)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.tab_6)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        spacerItem9 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem9)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.chercher = QtWidgets.QPushButton(self.tab_6)
        self.chercher.setMaximumSize(QtCore.QSize(30, 20))
        self.chercher.setObjectName("chercher")
        self.gridLayout_4.addWidget(self.chercher, 5, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.tab_6)
        self.label_13.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_13.setObjectName("label_13")
        self.gridLayout_4.addWidget(self.label_13, 1, 0, 1, 2)
        spacerItem10 = QtWidgets.QSpacerItem(20, 60, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout_4.addItem(spacerItem10, 2, 0, 1, 2)
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem11, 7, 0, 1, 2)
        self.classificateurPred = QtWidgets.QComboBox(self.tab_6)
        self.classificateurPred.setMaximumSize(QtCore.QSize(110, 16777215))
        self.classificateurPred.setObjectName("classificateurPred")
        self.classificateurPred.addItem("")
        self.gridLayout_4.addWidget(self.classificateurPred, 3, 0, 1, 1)
        self.lancer = QtWidgets.QPushButton(self.tab_6)
        self.lancer.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lancer.setObjectName("lancer")
        self.gridLayout_4.addWidget(self.lancer, 6, 0, 1, 2, QtCore.Qt.AlignHCenter)
        spacerItem12 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout_4.addItem(spacerItem12, 0, 0, 1, 2)
        self.patients = QtWidgets.QListWidget(self.tab_6)
        self.patients.setMaximumSize(QtCore.QSize(16777215, 20))
        self.patients.setObjectName("patients")
        self.gridLayout_4.addWidget(self.patients, 5, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.tab_6)
        self.label_20.setObjectName("label_20")
        self.gridLayout_4.addWidget(self.label_20, 4, 0, 1, 1)
        self.horizontalLayout_8.addLayout(self.gridLayout_4)
        self.tabWidget.addTab(self.tab_6, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Iguana"))
        self.label.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'trebuchet MS,sans-serif\'; font-size:15px; font-variant:small-caps;\">Affichage de graphe grâce à Cytoscape</span></p></body></html>"))
        self.launch.setWhatsThis(_translate("Form", "Lance le programme Cytoscape si celui-ci est bien installé auchemin suivant : \"C:Program FilesCytoscape_v3.5.1\"."))
        self.launch.setText(_translate("Form", "Lancer Cytoscape"))
        self.load.setWhatsThis(_translate("Form", "Permet de charger un graphe dans l\'application pour pouvoir l\'utiliser."))
        self.load.setText(_translate("Form", "Charger un graphe"))
        self.display.setWhatsThis(_translate("Form", "Affiche le graphe dans Cytoscape si celui-ci est bien lancé"))
        self.display.setText(_translate("Form", "Afficher le graphe"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Affichage de graphe"))
        self.label_2.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'trebuchet MS,sans-serif\'; font-size:15px; font-variant:small-caps;\">Réduction de graphes</span></p></body></html>"))
        self.load_2.setWhatsThis(_translate("Form", "Permet de charger un graphe dans l\'application pour pouvoir l\'utiliser."))
        self.load_2.setText(_translate("Form", "Charger un graphe"))
        self.reduc.setWhatsThis(_translate("Form", "Lance la réduction du graphe sélectionné dans la fenêtre ci-contre."))
        self.reduc.setText(_translate("Form", "Réduire le graphe"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "Réduction d\'un graphe"))
        self.label_3.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'trebuchet MS,sans-serif\'; font-size:15px; font-variant:small-caps;\">Coloration du graphe réduit grâce à IGGY-poc</span></p></body></html>"))
        self.label_4.setWhatsThis(_translate("Form", "Permet de définir un temps maximal d\'exécution (en seconde) durant lequel le programme va rechercher les colorations du graphe. Si ce temps est à 0, il n\'y aura pas de limite de temps."))
        self.label_4.setText(_translate("Form", "<html><body><p align=\"center\">Temps d\'exécution maximal (en secondes) :</p></body></html>"))
        self.time.setWhatsThis(_translate("Form", "Permet de définir un temps maximal d\'exécution (en seconde) durant lequel le programme va rechercher les colorations du graphe. Si ce temps est à 0, il n\'y aura pas de limite de temps."))
        self.load_3.setWhatsThis(_translate("Form", "Permet de charger un graphe dans l\'application pour pouvoir l\'utiliser."))
        self.load_3.setText(_translate("Form", "Charger un graphe"))
        self.color.setWhatsThis(_translate("Form", "Lance l\'identification des colorations et respect le temps donné en argument au dessus. Si ce temps est à 0, il n\'y aura pas de limite appliquée."))
        self.color.setText(_translate("Form", "Identifier les colorations"))
        self.color_1.setWhatsThis(_translate("Form", "Colore le graphe et l\'affiche dans Cytoscape. Il faut que Cytoscape soit lancé pour pouvoir effectuer cette action."))
        self.color_1.setText(_translate("Form", "Colorer le graphe"))
        self.nCompo.setWhatsThis(_translate("Form", "Exporte les N fichiers correspondant aux N composants identifiés du graphe sélectionné ci-contre."))
        self.nCompo.setText(_translate("Form", "N Composantes"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Form", "Coloration d\'un graphe"))
        self.afficherGraph.setWhatsThis(_translate("Form", "Permet d\'afficher le graphe de similarité du patient dont le nom est écrit. Il faut avoir démarré Cytoscape."))
        self.afficherGraph.setText(_translate("Form", "Afficher le graphe de similarité du patient :"))
        self.label_5.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'trebuchet MS,sans-serif\'; font-size:15px; font-variant:small-caps;\">Calcul de similarités</span></p></body></html>"))
        self.label_14.setText(_translate("Form", "Options :"))
        self.calcul.setWhatsThis(_translate("Form", "Cliquer sur ce bouton pour effectuer le calcul selon les options établies ci-dessus."))
        self.calcul.setText(_translate("Form", "Effectuer le calcul"))
        self.buttonClinique.setWhatsThis(_translate("Form", "Permet de charger le fichier clinique."))
        self.buttonClinique.setText(_translate("Form", "..."))
        self.buttonDonnee.setWhatsThis(_translate("Form", "Permet de choisir le dossier de données."))
        self.buttonDonnee.setText(_translate("Form", "..."))
        self.label_15.setWhatsThis(_translate("Form", "Dossier contenant l\'ensemble des fichiers des activations de gènes des patients. "))
        self.label_15.setText(_translate("Form", "Jeu de données :"))
        self.sif.setWhatsThis(_translate("Form", "Fichier .sif des gènes, il doit être issu du module coloration de graphes de Iguana."))
        __sortingEnabled = self.sif.isSortingEnabled()
        self.sif.setSortingEnabled(False)
        self.sif.setSortingEnabled(__sortingEnabled)
        self.graphSimi.setWhatsThis(_translate("Form", "Permet de générer le graphe des composants. Si cette option est cochée, il faut ajouter le graphe des gènes dans l\'espace ci-dessous."))
        self.graphSimi.setText(_translate("Form", "Générer le graphe des composants"))
        self.dataPred.setWhatsThis(_translate("Form", "Si cette option est cochée, le fichier généré servira pour la prédiction. Sinon, ce sera un fichier utile pour la création du classificateur et pour les tests."))
        self.dataPred.setText(_translate("Form", "Créer des données pour la prédiction"))
        self.nomPatient.setWhatsThis(_translate("Form", "Permet d\'afficher le graphe de similarité du patient dont le nom est écrit. Il faut avoir démarré Cytoscape."))
        self.csv.setWhatsThis(_translate("Form", "Fichier contenant la liste des gènes de chaque composant représenté dans les expressions de gène des patients. C\'est un fichier .csv utilisant des tabulations comme séparateur. Nécessaire que dans le cas où l\'on souhaite construire le graphe de similarité pour un patient."))
        __sortingEnabled = self.csv.isSortingEnabled()
        self.csv.setSortingEnabled(False)
        self.csv.setSortingEnabled(__sortingEnabled)
        self.label_16.setWhatsThis(_translate("Form", "Fichier clinique donnant le résultat pour chaque patient (état critique ou non). Pas nécessaire si on souhaite faire des données de prédiction."))
        self.label_16.setText(_translate("Form", "Fichier cliniques correspondant aux données :"))
        self.buttonSif.setWhatsThis(_translate("Form", "Permet de charger le graphe .sif des gènes."))
        self.buttonSif.setText(_translate("Form", "..."))
        self.label_17.setWhatsThis(_translate("Form", "Fichier contenant la liste des gènes de chaque composant représenté dans les expressions de gène des patients. C\'est un fichier .csv utilisant des tabulations comme séparateur. Nécessaire que dans le cas où l\'on souhaite construire le graphe de similarité pour un patient."))
        self.label_17.setText(_translate("Form", "Fichier CSV des gènes par composants :"))
        self.clinique.setWhatsThis(_translate("Form", "Fichier clinique donnant le résultat pour chaque patient (état critique ou non). Pas nécessaire si on souhaite faire des données de prédiction."))
        __sortingEnabled = self.clinique.isSortingEnabled()
        self.clinique.setSortingEnabled(False)
        self.clinique.setSortingEnabled(__sortingEnabled)
        self.buttonCsv.setWhatsThis(_translate("Form", "Permet de charger le fichier des gènes."))
        self.buttonCsv.setText(_translate("Form", "..."))
        self.donnees.setWhatsThis(_translate("Form", "Dossier contenant l\'ensemble des fichiers des activations de gènes des patients. "))
        __sortingEnabled = self.donnees.isSortingEnabled()
        self.donnees.setSortingEnabled(False)
        self.donnees.setSortingEnabled(__sortingEnabled)
        self.label_18.setWhatsThis(_translate("Form", "Fichier .sif des gènes, il doit être issu du module coloration de graphes de Iguana."))
        self.label_18.setText(_translate("Form", "Fichier .sif correspondant au graphe des gènes :"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("Form", "Calcul de similarités"))
        self.label_19.setWhatsThis(_translate("Form", "Correspond au fichier .csv permettant à l\'algorithme de machine learning d\'apprendre. Ce fichier doit spécifier le résultat attendu pour chaque patient."))
        self.label_19.setText(_translate("Form", "Fichier CSV permettant l\'apprentissage :"))
        self.comboBox.setWhatsThis(_translate("Form", "Permet de choisir le classificateur que l\'on souhaite utiliser pour le test."))
        self.comboBox.setItemText(0, _translate("Form", "--Classificateur--"))
        self.label_11.setWhatsThis(_translate("Form", "Matrice montrant le nombre de données évaluées par le classificateur par rapport au résultat attendu. On a ainsi la proportion de données ayant mal été prédites et dans quel cas (ex : donnée fausse prédite vraie sera sur l\'antidiagonale). Pour plus de détails se référer au guide d\'utilisation."))
        self.label_11.setText(_translate("Form", "<html><head/><body><p align=\"center\">Matrice de confusion :</p></body></html>"))
        self.label_10.setWhatsThis(_translate("Form", "La précision de ce classificateur sur ce jeu de données. Permet de savoir la proportion de données de validation aux quelle l\'algorithme a bien répondu (ex: l\'algorithme a prédit vrai et la réponse attendue était vrai)."))
        self.label_10.setText(_translate("Form", "<html><head/><body><p>Précision :</p></body></html>"))
        self.label_7.setWhatsThis(_translate("Form", "Affiche le nom du fichier testé."))
        self.label_7.setText(_translate("Form", "Fichier testé :"))
        self.pushButton.setWhatsThis(_translate("Form", "Pour charger un fichier .csv, du même format que l\'apprentissage, pour tester un classificateur."))
        self.pushButton.setText(_translate("Form", "Charger"))
        self.label_9.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'trebuchet MS,sans-serif\'; font-size:15px; font-variant:small-caps;\">Module de test</span></p></body></html>"))
        self.test.setWhatsThis(_translate("Form", "Lance le test si le fichier .csv et le classificateur ont bien été choisis."))
        self.test.setText(_translate("Form", "Test"))
        self.label_12.setWhatsThis(_translate("Form", "Le taux de réussite de ce modèle. Permet de savoir la proportion de données de validation aux quelle l\'algorithme a bien répondu (ex: l\'algorithme a prédit vrai et la réponse attendue était vrai)."))
        self.label_12.setText(_translate("Form", "Taux de réussite :"))
        self.fichier.setWhatsThis(_translate("Form", "Affiche le nom du fichier testé."))
        self.label_6.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'trebuchet MS,sans-serif\'; font-size:15px; font-variant:small-caps;\">Création du classificateur</span></p></body></html>"))
        self.chercheFichier.setWhatsThis(_translate("Form", "Permet de charger le fichier de données des patients."))
        self.chercheFichier.setText(_translate("Form", "..."))
        self.label_8.setWhatsThis(_translate("Form", "Permet de choisir la proportion de données destinées à être utilisées pour la validation du modèle."))
        self.label_8.setText(_translate("Form", "<html><head/><body><p>Validation (%) :</p></body></html>"))
        self.train.setWhatsThis(_translate("Form", "Pour lancer l\'entrainement permettant de créer le classificateur"))
        self.train.setText(_translate("Form", "Train"))
        self.taux.setWhatsThis(_translate("Form", "Le taux de réussite de ce modèle. Permet de savoir la proportion de données de validation aux quelle l\'algorithme a bien répondu (ex: l\'algorithme a prédit vrai et la réponse attendue était vrai)."))
        self.spinBox.setWhatsThis(_translate("Form", "Permet de choisir la proportion de données destinées à être utilisées pour la validation du modèle. Les valeurs sont usualement comprises entre 10% et 20%."))
        self.nomFichier.setWhatsThis(_translate("Form", "Correspond au fichier .csv permettant à l\'algorithme de machine learning d\'apprendre. Ce fichier doit spécifier le résultat attendu pour chaque patient."))
        self.matrice.setWhatsThis(_translate("Form", "Matrice montrant le nombre de données évaluées par le classificateur par rapport au résultat attendu. On a ainsi la proportion de données ayant mal été prédites et dans quel cas (ex : donnée fausse prédite vraie sera sur l\'antidiagonale). Pour plus de détails se référer au guide d\'utilisation."))
        self.precision.setWhatsThis(_translate("Form", "La précision de ce classificateur sur ce jeu de données. Permet de savoir la proportion de données de validation aux quelle l\'algorithme a bien répondu (ex: l\'algorithme a prédit vrai et la réponse attendue était vrai)."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("Form", "Classificateur"))
        self.chercher.setWhatsThis(_translate("Form", "Permet de charger le fichier de données des patients."))
        self.chercher.setText(_translate("Form", "..."))
        self.label_13.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'trebuchet MS,sans-serif\'; font-size:15px; font-variant:small-caps;\">Prédire l\'état des patients</span></p></body></html>"))
        self.classificateurPred.setWhatsThis(_translate("Form", "Permet de choisir le classificateur que l\'on souhaite utiliser pour la prédiction."))
        self.classificateurPred.setItemText(0, _translate("Form", "--Classificateur--"))
        self.lancer.setWhatsThis(_translate("Form", "Lancer la prédiction sur le fichier chargé."))
        self.lancer.setText(_translate("Form", "Lancer"))
        self.patients.setWhatsThis(_translate("Form", "Fichier de données des patients dont on souhaite prédire l\'état."))
        self.label_20.setWhatsThis(_translate("Form", "Fichier de données des patients dont on souhaite prédire l\'état."))
        self.label_20.setText(_translate("Form", "Données des patients dont on souhaite prédire l\'état :"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("Form", "Prédiction"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

