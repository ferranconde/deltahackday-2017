import matplotlib.pyplot as plt

from sklearn import datasets, svm

digits = datasets.load_digits()

# mostrar imatge d'exemple
plt.imshow(digits.images[155], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

n = len(digits.images)  # 1797 mostres

# actualment hi ha 1797 matrius de 8x8, les convertirem en llistes de 64 elements
dades = digits.images.reshape((n, 64))  # aplanem les matrius
entrenar = dades[:n//2]
labels = digits.target[:n//2]

# model classificador (support vector classifier)
classificador = svm.SVC()

# entrenem, per exemple, amb los primers 1000 elements
classificador.fit(entrenar, labels)

# predim resultats:

while True:
    num = int(raw_input("Num: "))
    plt.imshow(digits.images[num], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    print "La prediccio es: " + str(classificador.predict(dades[num].reshape((1, -1)))[0])
