import random
import matplotlib.pyplot as plt
import numpy as np

players_nr = 5
rounds = 25000
distance_list = [0] * 50

card_list = list(range(1,51))
print(str(card_list))


count_round = 0
for round in range(1,rounds+1):
    random.shuffle(card_list)
    player_cards = card_list[:players_nr] 
    player_cards.sort()

    for i in range(1,players_nr):
        distance = player_cards[i]-player_cards[i-1]
        distance_list[distance] += 1

# getting data of the histogram
#count, bins_count = np.histogram(distance_list, bins=50)

pdf = np.array(distance_list[1:]) / sum(np.array(distance_list[1:]))
cdf = np.cumsum(pdf)

#################################################################
x = range(1,50)
y = np.array(distance_list[1:]) 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, pdf, color='tab:red',label="PDF")
ax.plot(x, cdf, color='tab:blue',label = "CDF")
# Major ticks every 20, minor ticks every 5
major_x_ticks = np.arange(0, 51, 5)
minor_x_ticks = np.arange(0, 51, 1)
major_y_ticks = np.arange(0, 1.05, 0.1)
minor_y_ticks = np.arange(0, 1.05, 0.05)

###########################################################
##No need to change bellow   

ax.set_xticks(major_x_ticks)
ax.set_xticks(minor_x_ticks, minor=True)
ax.set_yticks(major_y_ticks)
ax.set_yticks(minor_y_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
plt.show()

print(str(card_list))