import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

def izracunajHistogram(sivaSlika):
    """Izračun histograma sivinske slike.
    @param sivaSlika : uint8 2D numpy array poljubne velikosti
    izhod: histogram slike"""
    histogram = np.zeros((256,), dtype="float64")
    for sivi_nivo in range(256):
        st_pikslov = np.sum(sivaSlika == sivi_nivo)
        histogram[sivi_nivo] = st_pikslov
    return histogram

def narisiHistogram(histogram):
    """Izris izračunanega histograma.
    @param histogram : 1D numpy array poljubne velikosti in tipa"""
    n_elementov = histogram.shape[0]
    plt.figure()
    plt.title("histogram")
    plt.bar(np.arange(n_elementov), histogram)
    plt.xlabel("svetilnost")
    plt.ylabel("st. pikslov")
    plt.grid(True)
    return 0

def dolociPrag(sivaSlika):
    """Določitev praga sive slike po metodi maksimizacjie informacije.
    @param sivaSlika : slika, ki ji določamo prag
    izhod: vrednost praga, ki maksimizira informacijo"""
    histogram = izracunajHistogram(sivaSlika)

    # izračun števila pikslov v sliki
    n = sivaSlika.shape[0] * sivaSlika.shape[1]

    # izračun porazdelitve relativnih frekvenc, P
    P = histogram / n

    # inicializacija vektorja, kamor bomo shranjevali informacijo
    # za vsako možno vrednost praga
    informacija = np.zeros_like(histogram)

    #TODO: izračunaj informacijo pri vsaki možni vrednosti praga
    #TODO: določi vrednost praga, ki maksimizira informacijo

    for t in range(256):      # za vsak možen prag

        p0 = np.sum(P[0:(t+1)])   
        p1 = 1 - p0

        if (p0 == 0 or p1 == 0):
            H = 0
        else:

            H0 = 0
            H1 = 0

            for i in range(t+1):
                if P[i] > 0:
                    H0 = H0 - (P[i] / p0) * np.log2(P[i] / p0)  # količina informacije do praga

            for j in range((t+1), 256):
                if P[j] > 0:
                    H1 = H1 - (P[j] / p1) * np.log2(P[j] / p1) # količina informacije po pragu

            H = H0 + H1  # skupna količina informacije

        informacija[t] = H  # shranimo v vektor z indeksom , ki je enak pragu

    # izračun informacije pri vsakem možnem pragu:
    # iskanje vrednosti praga, kjer je informacija maksimalna
    prag = np.argmax(informacija)

    return prag

if __name__ == "__main__":
    # Preberi sliko s podanega datotečnega imena
    try:
        filename = sys.argv[1]
    except IndexError:
        print("Uporaba programa: python vaja1a.py <datotečno ime slike>")
        sys.exit(1)

    # Branje slike in pretvorba barvnega prostora
    slika = cv2.imread(filename)
    slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    sivaSlika = cv2.cvtColor(slika, cv2.COLOR_RGB2GRAY)
    
    # TODO: Obdelava sivinske slike, npr. s postopki
    # izravnave histograma, filtrom mediane in Gaussovim glajenjem
    # gl. funkcije: cv2.equalizeHist, cv2.GaussianBlur, cv2.medianBlur

    obdelanaSlika = cv2.medianBlur(sivaSlika, 13)
    #obdelanaSlika = cv2.equalizeHist(obdelanaSlika)

    # Izračun in prikaz histograma ter slik

    histogram = izracunajHistogram(obdelanaSlika)
    narisiHistogram(histogram)

    plt.figure()
    plt.title("Barvna slika")
    plt.imshow(slika)

    plt.figure()
    plt.title("Sivinska slika")
    plt.imshow(sivaSlika, cmap="gray")

    plt.figure()
    plt.title("Obdelana slika")
    plt.imshow(obdelanaSlika, cmap="gray")

    # TODO: dopolnite funkcijo za izračun praga, dolociPrag
    # TODO: uporabite funkcijo dolociPrag za določitev praga obdelane sive slike
    # TODO: uporabite izračunano vrednost praga za upragovljanje sivinske slike
    # TODO: izpišite izračunano vrednost praga ter prikažite upragovljeno sliko,
    #       POLEG barvne in sivinske slike.

    prag = dolociPrag(obdelanaSlika)
    print(prag)
    _, upragovljenaSlika = cv2.threshold(obdelanaSlika, prag, 255, 0)

    plt.figure()
    plt.title("Upragovljena slika")
    plt.imshow(upragovljenaSlika, cmap="gray")

    plt.show()

    input("Press any key.")
    plt.close("all")
    sys.exit(0)
