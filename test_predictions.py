import numpy as np
import pandas as pd


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
end = "1491001200"
#Période par trading en secondes
period = "300"

print('On charge data')
df = pd.read_json("https://poloniex.com/public?command=returnChartData&currencyPair="+pair+"&start="+start+"&end="+end+"&period="+period,convert_dates=['date'])

#Mettre la date en index, pandas comprend que c'est une timeseries.
df=df.set_index('date')

#On crée la liste des prix d'ouverture (changer open par close, high, low, pour les différentes prédictions).
prices = np.array(df["open"])

#Ici on doit mettre la liste des prédictions (là je test avec une liste aléatoire).
noise = np.random.uniform(0.9,1.1,len(prices))
predictions =  prices*noise

print(predictions)

#On calcul l'argent final de l'investisseur. C'est ça qu'on doit battre.
hodl_money = money*(1-fees)*prices[-1]/prices[0]

print("L'investisseur aura %d dollars à la fin de la période." % hodl_money)

#On utilise les prédictions pour acheter chaque fois qu'on pense gagner plus que plus% et vendre quand on pense perdre plus que minus%.
#C'est une méthode assez basique et on pourra faire mieux après.
plus = 5.
minus = 5.
#Argent du début et est ce qu'on a vendu ou pas.
trade_crypto = money*(1-fees)/prices[0]
trade_money = 0
sold = False
for i in range(len(prices)-1):
	if (predictions[i+1] > (1+plus/100)*prices[i] and sold == True):
		sold = False
		trade_crypto = trade_money*(1-fees)/prices[i]
		trade_money = 0
	elif (predictions[i+1] < (1-minus/100)*prices[i] and sold == False):
		sold = True
		trade_money = trade_crypto*prices[i]*(1-fees)
		trade_crypto = 0

#argent final du trading.
final_money = trade_money + trade_crypto*(1-fees)/prices[-1]

print("Le bot aura %d dollars à la fin de la période." % final_money)






