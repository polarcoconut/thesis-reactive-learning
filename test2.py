from scipy.misc import comb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot([1,2,3],[4,5,6])
plt.savefig("jonathantest.png")

plt.show()

def computeThingy(n, p):
    q = 1.0 - p

    k = int(n / 2) + 1

    totalP = 0.0

    for i in range(k, n+1):
        for j in range(k, n+1):
            totalP += (comb(n, i) * (p ** i) * (q ** (n - i)) *
                       comb(n, j) * (p ** j) * (q ** (n - j)))
    for i in range(k, n+1):
        for j in range(k, n+1):
            totalP += (comb(n, i) * (q ** i) * (p ** (n - i)) *
                       comb(n, j) * (q ** j) * (p ** (n - j)))

            
    return totalP

    
for n in range(1, 20):
    print computeThingy(n, 0.55)
