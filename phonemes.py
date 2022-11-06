import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from tqdm import tqdm

def histogram_fonemov(y, fonemi):
    histogram = [(y==i).sum() for i in range(len(fonemi))]
    plt.bar(np.arange(len(fonemi)), histogram)
    plt.xticks(np.arange(len(fonemi)), fonemi)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def pripravi_zbirko(filename):
    """Prebere podatkovno zbirko iz datoteke .arff ter jo vrne v obliki,
    primerni za učenje in preizkušanje sklearn modelov.
    vhod: datotečno ime .arff datoteke
    izhodi: X - matrika značilk vzorcev
            y - matrika številčnih oznak vzorcev
            fonemi - seznam fonemov, ki dekodira številčne oznake"""
    with open(filename, "r") as f:
        vsebina = f.read()
    vrstice = vsebina.split("\n")[:-1]

    stevilo_vzorcev = 28801
    stevilo_znacilk = 13

    X = np.zeros((stevilo_vzorcev, stevilo_znacilk))
    y = np.zeros((stevilo_vzorcev))

    fonemi = vrstice[15][22:87].split(",")


    # TODO: preberi značilke vzorcev in jih shrani v matriko X
    # TODO: preberi foneme in jih shrani v seznam fonemi
    # TODO: preberi oznake vzorcev, jih kodiraj tako, da indeksirajo
    #       seznam fonemi, in shrani v vektor y

    for i in range(stevilo_vzorcev-18):

        znacilke = vrstice[18+i].split(",")
        oznaka = znacilke.pop()

        X[i] = znacilke
        y[i] = fonemi.index(oznaka)


    # prikaz porazdelitve fonemov
    histogram_fonemov(y, fonemi)

    return X, y, fonemi

def premesaj_vrstice(X, y):
    inds = np.random.permutation(X.shape[0])
    return X[inds], y[inds]

def konfuzijska_matrika(y_test, y_hat, fonemi):
    """Prikaže konfuzijsko matriko razpoznavalnika na podlagi
    pravilnih oznak (y_test), izračunanih oznak (y_hat) ter seznama fonemov."""
    matrika = np.zeros((len(fonemi), len(fonemi)), dtype="int32")

    #TODO: sestavi konfuzijsko matriko

    for i in range(len(y_test)):
        matrika[int(y_test[i])][int(y_hat[i])] += 1

    plt.imshow(matrika)
    plt.xticks(np.arange(len(fonemi)), fonemi)
    plt.yticks(np.arange(len(fonemi)), fonemi)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    temp = [matrika[j][j]/sum(matrika[j]) for j in range(len(fonemi))]
    #print(sum(matrika[temp.index(max(temp))]))
    #print(matrika[temp.index(max(temp))][temp.index(max(temp))])
    m = fonemi[temp.index(max(temp))]
    n = fonemi[temp.index(min(temp))]
    print("Najbolje razpoznan fonem: ",m)
    print("Najslabse razpoznan fonem: ",n)

    temp2 = matrika
    for j in range(len(fonemi)):
        temp2[j][j] = 0
    naj_napaka = np.unravel_index(temp2.argmax(), temp2.shape)
    print("Najpogostejsa napaka: ", fonemi[naj_napaka[0]], "-> ", fonemi[naj_napaka[1]])



def navzkrizno_preverjanje(X, y, N, razvrscevalnik, fonemi):
    """Navzkrižno preverjanje natančnosti razpoznavalnika
    na podatkovni zbirki, podani z vektorji značilk X in vektorjem oznak y."""
    # Najprej naključno premešamo matriko X in vektor y:
    X_premesan, y_premesan = premesaj_vrstice(X, y)

    # podatkovno zbirko enakomerno razdelimo na N kosov:
    X_deli = []
    y_deli = []
    for i in range(N):
        i0 = (X.shape[0] // N) * i
        i1 = (X.shape[0] // N) * (i + 1)

        X_del = X_premesan[i0 : i1]
        y_del = y_premesan[i0 : i1]

        X_deli.append(X_del)
        y_deli.append(y_del)

    # Seznam, kamor bomo shranjevali uspešnosti poskusov
    uspesnosti = []

    # N-krat ponovimo postopek učenja in testiranja, pri čemer v i-tem poskusu
    # testiramo na i-tem delu razdeljene zbirke, učimo pa na vseh ostalih:
    for i in range(N):
        X_train = [X_deli[ind] for ind in range(N) if ind != i]
        y_train = [y_deli[ind] for ind in range(N) if ind != i]

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        X_test = X_deli[i]
        y_test = y_deli[i]

        razvrscevalnik.fit(X_train, y_train)

        y_hat = razvrscevalnik.predict(X_test)

        # uspešnost merimo kot delež predvidenih oznak na testni zbirki, ki s
        # ujemajo z dejanskimi:
        uspesnost = np.mean(y_hat == y_test)
        uspesnosti.append(uspesnost)

        if i == 0:
            konfuzijska_matrika(y_test, y_hat, fonemi)

    return uspesnosti

if __name__ == "__main__":
    X, y, fonemi = pripravi_zbirko("posnetek.arff")

    razvrscevalnik1 = RidgeClassifier(alpha=1, fit_intercept= True, normalize=False)
    razvrscevalnik2 = LogisticRegression()

    uspesnost1 = navzkrizno_preverjanje(X, y, 5, razvrscevalnik1, fonemi)
    uspesnost2 = navzkrizno_preverjanje(X, y, 5, razvrscevalnik2, fonemi)

    print("Povprecna uspesnost - najmanjsi kvadrati: ", sum(uspesnost1)/len(uspesnost1))
    print("Povprecna uspesnost - logisticna regresija: ", sum(uspesnost2)/len(uspesnost2))
