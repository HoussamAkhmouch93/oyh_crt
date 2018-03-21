import numpy as np
import pandas as pd
import time, json
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error


#Les données à entrer.
#La mise initiale en usdt
money = 1000
#Prix du trading.
fees = 0.0005
#Pair à trader (pour l'instant on ne prend que des pairs avec USDT et on considère que USDT = USD).
pair = "USDT_ETH"
#Date début en timestamp
start = "1483228800"
#Date fin en timestamp
end = "1521384257"
#Période par trading en secondes
period = "300"

print('On charge data')
df = pd.read_json("https://poloniex.com/public?command=returnChartData&currencyPair="+pair+"&start="+start+"&end="+end+"&period="+period,convert_dates=['date'])

#Mettre la date en index, pandas comprend que c'est une timeseries.
df=df.set_index('date')

nb_test=50
wa_data = df['weightedAverage']
#size permet de préciser le nombre de valeurs à prédire, remplacer 2 par le nombre de valeurs
size = int(len(wa_data) - nb_test)
#On crée la liste des prix d'ouverture (changer open par close, high, low, pour les différentes prédictions).
prices = np.array(df["weightedAverage"])[size:len(wa_data)]

#Ici on doit mettre la liste des prédictions
wa_data = df['weightedAverage']
size = int(len(wa_data) - nb_test)
train, test = wa_data[0:size], wa_data[size:len(wa_data)]
history = [x for x in train]
predictions = list()

print('Printing Predicted')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f, day=%s, diff=%f' % (yhat, obs,str(test.index[t]),(yhat-obs)*100/yhat))

test=np.array(test)
predictions = np.array(predictions)
error = mean_squared_error(test, predictions)


#On calcul l'argent final de l'investisseur. C'est ça qu'on doit battre.
start_money_crpt=money*(1-fees)/prices[0]
final_money_eur=prices[-1]*(1-fees)*start_money_crpt
gain=final_money_eur
hodl_money = money*(1-fees)*prices[-1]/prices[0]

print("L'investisseur aura %d dollars à la fin de la période." % gain)

#On utilise les prédictions pour acheter chaque fois qu'on pense gagner plus que plus% et vendre quand on pense perdre plus que minus%.
#C'est une méthode assez basique et on pourra faire mieux après.
plus = 0
minus = 0
#Argent du début et est ce qu'on a vendu ou pas.
trade_crypto = money*(1-fees)/prices[0]
trade_money = 0
sold = False
for i in range(len(prices)-1):
	print(i)
	print(predictions[i+1])
	print((1+plus/100)*prices[i])
	print((1-minus/100)*prices[i])
	if (predictions[i+1] > (1+plus/100)*prices[i] and sold == True): #On achète
		sold = False
		trade_crypto = trade_money*(1-fees)/prices[i]
		trade_money = 0
		print('je suis là 1')
	elif (predictions[i+1] < (1-minus/100)*prices[i] and sold == False): #On vend
		sold = True
		trade_money = trade_crypto*prices[i]*(1-fees)
		trade_crypto = 0
		print('je suis là 2')

#argent final du trading.
final_money = trade_money + trade_crypto*(1-fees)*prices[-1]

print("Le bot aura %d dollars à la fin de la période." % final_money)