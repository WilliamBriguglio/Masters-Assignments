import pandas as pd
import numpy as np

#initalize totals
total = np.zeros((12, )).astype(int)

max = 0
#for each batch, read in batch stats and aggregate them
for i in range(30):
	cur = pd.read_csv("batch_stats/B_"+str(i), header=None, sep=":").to_numpy()
	total = total + cur[:12,1].astype("int")
	if cur[12,1] > max:
		max = cur[12,1]


#print total stats
fp = open("totalStats.csv", "w")
fp.write(">1024000, " + str(total[0])+"\n")
fp.write(">1000000, " + str(total[1])+"\n")
fp.write(">900000, " + str(total[2])+"\n")
fp.write(">800000, " + str(total[3])+"\n")
fp.write(">700000, " + str(total[4])+"\n")
fp.write(">600000, " + str(total[5])+"\n")
fp.write(">500000, " + str(total[6])+"\n")
fp.write(">400000, " + str(total[7])+"\n")
fp.write(">300000, " + str(total[8])+"\n")
fp.write(">200000, " + str(total[9])+"\n")
fp.write(">100000, " + str(total[10])+"\n")
fp.write("<=100000, " + str(total[11])+"\n")
fp.write("max, "+ str(max))
