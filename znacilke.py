from obrisi import *
from sklearn.feature_selection import SelectKBest


def obdelaj_sliko(slika):
    """Pomožna funkcija za predobdelavo slike."""
    # TODO: spiši funkcijo.
    # Predlagani koraki predobdelave:
    # 1. Prevzorčenje slike na velikost 256 x 256 pikslov
    # 2. Filtriranje prevzorčene slike z uporabo
    #    filtra mediane z okolico 11 pikslov
    # 3. Upragovljanje filtrirane slike z maksimizacijo informacije

    slika = cv2.resize(slika, (256,256))
    slika = cv2.medianBlur(slika, 11)
    prag = dolociPrag(slika)
    _, slika = cv2.threshold(slika, prag, 255, 0)

    #plt.figure()
    #plt.title("slika2")
    #plt.imshow(slika, cmap="gray")
    #plt.show()

    return slika
    #return np.zeros_like(slika)

def pretvori_obris_v_signal(obris):
    """Pretvorba obrisa,
    iz zapisa v obliki zaporedja točk v kompleksni signal."""
    tocke_obrisa = obris["tocke"]
    dolzina = len(tocke_obrisa)
    signal = np.zeros((dolzina,), dtype=np.complex128)
    # TODO: izvedi pretvorbo

    signal.real = [i[0] for i in tocke_obrisa]
    signal.imag = [j[1] for j in tocke_obrisa]

    return signal

def prevzorci_signal(signal, N_nov=64):
    """Prevzorčenje signala poljubne dolžine na fiksno število točk."""
    N_orig = signal.shape[0]
    nov_signal = np.zeros((N_nov,), dtype=signal.dtype)
    for i in range(N_nov):
        i_r = i / (N_nov - 1)
        j_r = i_r * (N_orig - 1)

        if j_r == 0 or j_r == (N_orig - 1):
            nov_signal[i] = signal[int(j_r)]
        else:
            j0 = int(j_r)
            j1 = j0 + 1
            t = j_r - j0

            y0 = signal[j0]
            y1 = signal[j1]
            nov_signal[i] = y0 * (1 - t) + y1 * t

    return nov_signal

def doloci_ffk(slika, kmax, lmax):
    """Določitev vektorja značilk iz najdaljšega obrisa na sliki."""
    #binarnaSlika = obdelaj_sliko(slika)                                         # zakaj bi jo se enkrat obdelali?
    binarnaSlika = slika
    iskalnikObrisov = Iskalnik(binarnaSlika)
    iskalnikObrisov.isci_obrise()
    obris = iskalnikObrisov.podaj_najdaljsi_obris()

    signal = pretvori_obris_v_signal(obris)
    signal = prevzorci_signal(signal)

    signal_fft = np.fft.fft(signal)

    vektor_ffk = np.zeros((2 * kmax * (lmax - 1),), dtype=np.float64)
    ind_vektor = 0

    for i in range(1, kmax + 1):
        for j in range(2, lmax + 1):
            F_k = signal_fft[i + 1] ** j
            F_l = signal[-j + 1] ** i
            F1 = signal_fft[1] ** (i + j)

            # TODO: Izvedi izračun d_ij po formuli, podani v literaturi
            d_ij = (F_k * F_l) / F1;

            d0 = d_ij.real
            d1 = d_ij.imag
            vektor_ffk[ind_vektor] = d0
            vektor_ffk[ind_vektor + 1] = d1
            ind_vektor += 2

    return vektor_ffk

if __name__ == "__main__":
    # TODO: preberi vseh 6 slik v mapi "slike" ter jih shrani v spodaj
    #       inicializiran seznam.
    dn = "slike/"
    fns = "kladivo1 kladivo2 kladivo3 kljuc1 kljuc2 kljuc3".split(" ")
    fns = [dn + fn + ".png" for fn in fns]

    slike = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in fns]

    # obdelava slik s pomožno funkcijo, ki jo morate še dopolniti
    obdelane_slike = []
    for slika in slike:

        obdelana_slika = obdelaj_sliko(slika)
        obdelane_slike.append(obdelana_slika)

    # Izračunaj vektorje značilk vsake izmed slik
    # TODO: dopolni funkcijo za določitev vektorjev značilk
    vektorji = []
    for obdelana_slika in obdelane_slike:

        vektor = doloci_ffk(obdelana_slika, 4, 4)                                    #obdelana_slika namesto slika
        vektorji.append(vektor)


    # izris vektorjev značilk
    for i in range(len(vektorji)):
        if i < 3:
            barva = "b"
        else:
            barva = "g"
        vektor = vektorji[i]

        x = np.arange(len(vektor)) + 0.12 * i

        # TODO: za boljši prikaz vektorje značilk pretvori v logaritemsko
        #       merilo, po izrazu sgn(v) * log_10(|v|)

        log_prikaz = np.sign(vektor) * np.log10(np.abs(vektor))

        plt.bar(x, log_prikaz, width=0.1, color=barva)                            #log_prikaz namesto prikaz

    plt.grid(True)
    plt.show()

    # Izbira najboljših značilk za ločitev razredov objektov
    X = np.array(vektorji)
    y = np.array([0, 0, 0, 1, 1, 1])
    X_new = SelectKBest(k=4).fit_transform(X, y)

    # Prikaz podobnosti vektorjev s kosinusno mero podobnosti
    # ter z evklidsko razdaljo
    razdalje_evk = np.zeros((6, 6), dtype="float64")
    podobnosti_cos = np.zeros_like(razdalje_evk)


    for i in range(len(vektorji)):
        for j in range(len(vektorji)):
            v1 = X_new[i]
            v2 = X_new[j]

            # TODO: izračunaj kosinusno podobnost ter negativno evklidsko
            #       razdaljo med vektorjema v1 in v2

            razdalje_evk[i, j] = - np.linalg.norm(v1 - v2)
            podobnosti_cos[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


    plt.figure()
    plt.title("Negativna evklidska razdalja.")
    plt.imshow(razdalje_evk)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("Kosinusna podobnost.")
    plt.imshow(podobnosti_cos)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
