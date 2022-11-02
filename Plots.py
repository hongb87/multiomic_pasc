import matplotlib.pyplot as plt
import numpy as np

barWidth = 0.12
fig = plt.subplots(figsize = (12,8))

up1 = [22,14]
up2 = [26,27]
down1 = [29,26]
down2 = [27,72]
up3 = [13,8]
down3 = [9,13]
common = [22,21]

br1 = [0,0.7]
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

p1 = plt.bar(br1, down1, width = barWidth - 0.02, label = 'downregulated', color = (0.9,0.3,0.4), edgecolor = 'black')
p3 = plt.bar(br1, up1, width = barWidth - 0.02, bottom = down1, label = 'upregulated', color = (0.5,0.8,0.7,1),edgecolor = 'black')
p4 = plt.bar(br2, down2, width = barWidth - 0.02, color = (0.9,0.3,0.4),edgecolor = 'black')
p5 = plt.bar(br2, up2, width = barWidth - 0.02, bottom = down2, color = (0.5,0.8,0.7,1),edgecolor = 'black')
p6 = plt.bar(br3, down3, width = barWidth - 0.02, color = (0.9,0.3,0.4),edgecolor = 'black')
p7 = plt.bar(br3, up3, width = barWidth - 0.02, bottom = down3, color = (0.5,0.8,0.7,1),edgecolor = 'black')

#p2 = plt.bar(br3, common, width = barWidth - 0.02, color = (0.3,0.3,0.9),edgecolor = 'black')

plt.title('Significantly Different Genes vs. Mock (P-value <= 0.05)')

plt.ylabel('Number of genes')
plt.xticks([br1[0],br2[0],br3[0],br1[1],br2[1],br3[1]],['aedesMR','aedesPRV','Ae_InfectionCommon','anophelesMR','anophelesPRV','An_InfectionCommon'], rotation = -25)

plt.text(-0.01,52,'51')
plt.text(-0.01 + 0.12,54,'53')
plt.text(-0.01 + 0.24,23,'22')
plt.text(-0.01 + 0.7,41,'40')
plt.text(-0.01 + 0.12 + 0.7,100,'99')
plt.text(-0.01 + 0.24 + 0.7,22,'21')

plt.legend()
plt.show()




