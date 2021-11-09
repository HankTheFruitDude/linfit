#!/bin/python
# Geschrieben 10/2020, Henry Korhonen henryk@ethz.ch, basierend auf Matlabskripten von Martin Willeke. Hinweise, Bemerkungen und Vorschläge bitte an henryk@ethz.ch.

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from math import sqrt
import matplotlib.pyplot as plt

xy = pd.read_table('test_lin.txt', names=['x','y'], sep=r'\s+') # Lesen der Daten, erstellen eines Dataframes. Als Separator kommt hier eine unbestimmte Anzahl Leerschläge in Frage. Andernfalls "sep" anpassen.

y = xy['y'] # Relevante Daten aus dem Dataframe extrahieren. Achtung: "names" in pd.read_table gibt der ersten Spalte den Namen x und der zweiten y. Unbedingt sicherstellen, dass die richtigen Daten extrahiert werden!
x = xy['x']

N = len(y) # Anzahl Datenpunkte ermitteln.

def func(x, a, b): # Funktion definieren, die gefittet werden soll. Hier wird mit einer Polynomfunktion ersten Grades gearbeitet.

    return a*x +b

popt, pcov = curve_fit(func, x, y) # fitten der Daten
# popt, pcov = curve_fit(func, x, y, bounds=([alower, blower], [aupper, bupper])) # fitten der Daten mit Eingrenzung der Regressionskoeffizienten
pstd = np.sqrt(np.diag(pcov)) # Standardabweichung der Regressionskoeffizienten. Nota bene: auf der Diagonalen von pcov stehen die Varianzen der Regressionskoeffizienten.

alpha = 0.05 # m%-Vertrauensintervall: m = 100*(1-alpha)
p = len(popt)
dof = max(0,N-p) # Anzahl Freiheitsgrade (nota bene: das hängt von der Anzahl Regressionskoeffizienten in der Fitfunktion ab (siehe def func(...) oben)
tinv = stats.t.ppf(1.0-alpha/2., dof) # Student-T-Faktor ermitteln


print('Anzahl Freiheitsgrade: {0}\nAnzahl Messungen: {1}\n=================================='.format(dof, N))
for i, regkoeff,var in zip(range(N), popt, np.diag(pcov)): # Hier werden alle Regressionskoeffizienten mit den entsprechenden Vertrauensintervallen ausgegeben.
    sigma = var**0.5
    print('Parameter {0}: {1} \n Vertrauensintervall: [{2}  {3}] \n Standardabweichung: {4} \n =================================='.format(i+1, regkoeff, regkoeff - sigma*tinv, regkoeff + sigma*tinv, sigma))

obere = popt+np.diag(pcov)**0.5*tinv
untere = popt-np.diag(pcov)**0.5*tinv

plt.plot(x,y,'b.',label='data') # Daten plotten
plt.plot(x, func(x, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt)) # Gefittete Funktion plotten. Auch hier muss angepasst werden, wenn mehr als nur a und b genutzt werden sollen (siehe def func(...) oben)
# Man beachte, dass *popt der Funktion func() (die ja weiter oben definiert wurde) sämtliche Parameter weitergibt, die beim fitten bestimmt wurden. Das können auch mehrere sein; so viele, wie bei der Definition angegeben wurden.
plt.plot(x, func(x, *obere), 'b-', label='obere grenze: a=%5.3f, b=%5.3f' % tuple(popt+pstd))
plt.plot(x, func(x, *untere), 'g-', label='untere grenze: a=%5.3f, b=%5.3f' % tuple(popt-pstd))

plt.grid(linestyle=':') # grid zeichnen

plt.xlabel('x') # Labels setzen
plt.ylabel('y')
plt.legend() # Legende generieren

plt.savefig("dateiname1.pdf") # Plot als PDF-Datei speichern.
plt.savefig("dateiname1.png") # Plot als PNG-Datei speichern.

plt.show() # Plot anzeigen


plt.grid(linestyle=':')
plt.plot((x[0],x[N-1]),(0,0),'g-',label='Nulllinie')
plt.plot(x, y-func(x, *popt),'b.',label='Abweichungen')

plt.xlabel('x') # Labels setzen
plt.ylabel('y')
plt.legend() # Legende generieren

plt.savefig("dateiname2.pdf") # Plot als PDF-Datei speichern.
plt.savefig("dateiname2.png") # Plot als PNG-Datei speichern.

plt.show() # Plot anzeigen
