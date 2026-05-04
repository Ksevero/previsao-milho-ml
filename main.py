import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados simulados (dias vs preço do milho)
dias = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
precos = np.array([50,52,53,55,54,56,58,60,62,65])

# Criando o modelo
modelo = LinearRegression()
modelo.fit(dias, precos)

# Previsão para dias futuros
dias_futuros = np.array([11,12,13,14,15]).reshape(-1,1)
previsao = modelo.predict(dias_futuros)

# Plotando
plt.scatter(dias, precos, color='blue', label='Dados reais')
plt.plot(dias, modelo.predict(dias), color='green', label='Modelo')
plt.scatter(dias_futuros, previsao, color='red', label='Previsão')

plt.xlabel('Dias')
plt.ylabel('Preço do milho')
plt.title('Previsão de preço do milho usando Machine Learning')
plt.legend()
plt.show()

# Print previsões
for dia, valor in zip(dias_futuros, previsao):
    print(f"Dia {dia[0]} -> Preço previsto: {valor:.2f}")