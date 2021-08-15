import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

reader = pd.read_csv('worksheet.csv')
data = np.array(pd.DataFrame(reader))
print(data)

M: int = 10 ** 5 + 1
x = data[1:M+1, 1]
y = data[1:M+1, 2]
Iy = data[1:M+1, 3]

plt.plot(x, y, ls='-', lw=0.3, color='black')
plt.show()
plt.plot(x, Iy, ls='-', lw=0.3, color='blue')
plt.show()
