from znacilke import *
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier as KNN
import sys, os
from tqdm import tqdm
from sklearn.model_selection import cross_validate, KFold

def csm(v1, v2):
    """Kosinusna mera podobnosti med vektorjema v1 in v2."""

    cos_pdb = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return cos_pdb # TODO: spiši funkcijo

def evk(v1, v2):
    """Evklidska razdalja med vektorjema v1 in v2."""

    evk_raz = np.linalg.norm(v1 - v2)

    return evk_raz # TODO: spiši funkcijo

class CSMKNN():
    """Razpoznavalnik s prileganjem K najbližjih sosedov,
    ki za prileganje lahko uporablja poljubno mero razdalje ali podobnosti."""

    def __init__(self, k=5, mera=csm, nacin="max"):
        """Inicializira razpoznavalnik.
        k: število najbližjih sosedov, ki jih upoštevamo pri razvrščanju
        mera: funkcija, ki kot vhod prejme dva vektorja značilk in vrne
              njuno razdaljo ali podobnost.
        nacin: 'max', če uporabljamo mero podobnosti, oz.
               'min' za mere razdalje."""
        self.k = k
        self.mera = mera
        self.nacin = nacin

    def fit(self, X, y):
        """ Metoda za učenje razpoznavalnika.
        X: Vektorji značilk, array velikosti (N_slik, N_znacilk)
        y: oznake slik, array velikosti (N_slik,)
        """
        # oznake grejo od 0 do (N_razredov - 1)
        # "učenje" razpoznavalnika sestoji samo iz koraka kopiranja učne zbirke
        self.N_razredov = y.max() + 1
        self.X = np.copy(X)
        self.y = np.copy(y)

        #print(X.shape)

    def predict(self, X):
        """ Metoda za zaganjanje razpoznavalnika na testnih podatkih,
        ki jih želimo razvrščati.
        X: vektorji značilk, ki jih želimo razvrstiti v razrede učne množice,
           array velikosti (N_slik, N_znacilk) - N_slik je lahko različen od
           učne množice, N_znacilk pa mora biti enak
        izhod: oznake razredov, ki jih razpoznavalnik pripiše vsakemu izmed
               vektorjev značilk
        """
        # vektor, kamor bomo shranili izračunane oznake vektorjev X
        y = np.zeros((X.shape[0],), dtype="int32")

        # Izračunamo matriko podobnosti med vsakim vektorjem značilk v učni
        # zbirki in vsakim vektorjem značilk v testni zbirki
        # self.X : naša učna zbirka
        # X : naša testna zbirka
        M = np.zeros((X.shape[0], self.X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.X.shape[0]):
                M[i, j] = self.mera(X[i], self.X[j])

        # i-ta vrstica matrike M zdaj vsebuje podobnosti med i-tim vektorjem v
        # testni zbirki ter vsakim izmed vektorjev v učni zbirki.
        # s funkcijo argsort najdemo indekse K najbolj podobnih:

        for i in range(M.shape[0]):
            indeksi = np.argsort(M[i])

            # zanimajo nas najbolj podobni vektorji, ki so za mere podobnosti
            # na koncu razvrščenega seznama, za razdalje pa na začetku:
            if self.nacin == "max":
                sosedi = indeksi[-self.k:]
            elif self.nacin == "min":
                sosedi = indeksi[:self.k]
            else:
                raise ValueError("Nacin mora biti 'min' ali 'max'.")

            # poiščemo oznako vsakega od najbližjih sosedov:
            oznake = [self.y[indeks] for indeks in sosedi]

            # preštejemo, kolikokrat se v vektorju oznak pojavi vsaka izmed
            # oznak razredov:
            pojavi = []
            for j in range(self.N_razredov):
                pojavi_j = np.sum([oznaka == j for oznaka in oznake])
                pojavi.append(pojavi_j)

            # i-temu vektorju v testni zbirki priredimo oznako tistega razreda,
            # ki se med njegovimi K najbližjimi sosedi pojavi največkrat:
            y[i] = np.argmax(pojavi)

        return y

def premesaj_vrstice(X, y):
    """Premeša matriko vektorjev X ter vektor oznak y, z uporabo
    iste naključne permutacije."""
    X_premesan = np.zeros_like(X)
    y_premesan = np.zeros_like(y)

    indeksi = np.random.permutation(X.shape[0])

    for i, indeks in enumerate(indeksi):
        X_premesan[i] = X[indeks]
        y_premesan[i] = y[indeks]

    return X_premesan, y_premesan

def navzkrizno_preverjanje(X, y, N, izbiralnik, razvrscevalnik):
    """Navzkrižno preverjanje natančnosti razpoznavalnika
    na podatkovni zbirki, podani z vektorji značilk X in vektorjem oznak y."""
    # Najprej naključno premešamo matriko X in vektor y:
    X_premesan, y_premesan = premesaj_vrstice(X, y)

    # podatkovno zbirko enakomerno razdelimo na N kosov, ki jih shranimo v
    # spodnja seznama:
    X_deli = []
    y_deli = []
    for i in range(N):
         # TODO: enakomerno razdeli podatkovno zbirko
         a = int(len(X_premesan) / N);
         X_deli.append((X_premesan[i*a:(i+1)*a]))
         y_deli.append((y_premesan[i*a:(i+1)*a]))

    # Seznam, kamor bomo shranjevali uspešnosti poskusov
    uspesnosti = []

    # N-krat ponovimo postopek učenja in testiranja, pri čemer v i-tem poskusu
    # testiramo na i-tem delu razdeljene zbirke, učimo pa na vseh ostalih:
    for i in range(N):
        #TODO: določi učno in testno množico za i-ti poskus
        X_train = np.delete(X_deli, i, 0);
        X_train = np.array([j for i in X_train for j in i])

        y_train = np.delete(y_deli, i, 0);
        y_train = np.array([j for i in y_train for j in i])

        X_test = np.array(X_deli[i])
        y_test = np.array(y_deli[i])


        # Izbira značilk na podlagi učne množice:
        izbiralnik.fit(X_train, y_train)

        X_train = izbiralnik.transform(X_train)
        X_test  = izbiralnik.transform(X_test)

        # Učenje razvrščevalnika na učni množici
        razvrscevalnik.fit(X_train, y_train)

        # Razvrščanje testne množice
        y_hat = razvrscevalnik.predict(X_test)

        # uspešnost merimo kot delež predvidenih oznak na testni zbirki, ki se
        # ujemajo z dejanskimi:
        uspesnost = np.mean(y_hat == y_test)
        uspesnosti.append(uspesnost)

    return uspesnosti

def shrani_znacilke():
    """Funkcija, prebere značilke slik ter njihov razred, in jih shrani v
    obliki numpy arrayev."""
    dn = "slike/"
    fns = sorted(os.listdir(dn))

    slike = []
    oznake = []

    for fn in fns:
        # TODO: Iz datotečnega imena preberi oznako objekta na sliki.
        #       Oznako, ki naj bo 0-4, shrani v seznam oznake.
        # TODO: preberi sliko in jo shrani v seznam slike

        oznake.append(int(fn[3]) - 1);
        slike.append(cv2.imread(dn + fn));

    vektorji = []
    for slika in tqdm(slike):
        vektor = doloci_ffk(slika, 4, 4)
        vektorji.append(vektor)

    X = np.array(vektorji)
    y = np.array(oznake)
    np.save("X.npy", X)
    np.save("y.npy", y)
    return (X, y)

def nalozi_znacilke():
    X = np.load("X.npy")
    y = np.load("y.npy")
    return (X, y)

if __name__ == "__main__":
    if "X.npy" in os.listdir(".") and "y.npy" in os.listdir("."):
        X, y = nalozi_znacilke()
    else:
        X, y = shrani_znacilke()

    # TODO: preveri uspešnost razpoznavanja za različne kombinacije
    #       števila značilk, števila najbližjih sosedov, in mere,
    #       ki se uporablja za prileganje

    f1 = open("evk.txt", "w")
    f2 = open("csm.txt", "w")

    f1.write("       1        3        5        7 \n 1 ");
    f2.write("       1        3        5        7 \n 1 ");

    for i in range(1,25):
        for k in [1,3,5,7]:

            izbiralnik1 = SelectKBest(k=i)
            razvrscevalnik1 = CSMKNN(k=k, mera=evk, nacin="min")

            izbiralnik2 = SelectKBest(k=i)
            razvrscevalnik2 = CSMKNN(k=k, mera=csm, nacin="max")

            uspesnosti1 = navzkrizno_preverjanje(X, y, 6, izbiralnik1, razvrscevalnik1)
            uspesnosti2 = navzkrizno_preverjanje(X, y, 6, izbiralnik2, razvrscevalnik2)

            f1.write("%f " % ( (sum(uspesnosti1) / len(uspesnosti1))))
            f2.write("%f " % ( (sum(uspesnosti2) / len(uspesnosti2))))
        if (i < 24):
            f1.write("\n %d "% (i+1))
            f2.write("\n %d "% (i+1))

    f1.close()
    f2.close()
