import pandas as pd
from sklearn import linear_model

dades = pd.read_fwf("renda_residus.txt")
renda = dades[['Renda']]
residus = dades[['Residus']]

model = linear_model.LinearRegression()

model.fit(renda, residus)

while True:
    num = int(raw_input("Renda? "))
    print "Residus estimats:", model.predict(num)[0][0]
