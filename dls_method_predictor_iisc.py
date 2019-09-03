import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import math
from scipy.optimize import minimize
import warnings

# Input data using Panda library
data = pd.read_csv('04_cricket_1999to2011.csv')

# converting the input into data frames of columns and rows
sel_data = pd.DataFrame(data, columns = ['Match', 'Innings', 'Runs.Remaining', 'Wickets.in.Hand','Over', 'Innings.Total.Runs'])

# To select the data only for the first innings
inning_first = sel_data['Innings'] == 1
sel_data = sel_data[inning_first]

# We convert the overs compelted into the overs remaining
sel_data['Over'] = 50-sel_data['Over']

# function to find the average of all the runs at a given wicket available
def getMeanRunByWicket(w):
	selWicket = sel_data['Wickets.in.Hand'] == w
	df = sel_data[selWicket]
	return np.mean(df.groupby(['Match'])['Runs.Remaining'].max())

# Function to calculate run with model with given b-value and max run value
def func(Z0, b, x):
	# print(x)
	# print(Z0)
	Z0x = Z0 * (1 - np.exp(-b * x))
	return Z0x

# Function to use the model to predict the run
def func2(z, u, L):
	return z * (1 - np.exp(-L / z * u))

# Function to get the average of all the max runs possible at given wicket and over
def getMeanRun(o,w):
	selWicket = sel_data['Wickets.in.Hand'] == w
	if(o==50): # This is when we have all overs remaining
		selOver = sel_data['Over'] == 49
		sel_data_by_over = sel_data[selOver]
		m = np.mean(sel_data_by_over['Innings.Total.Runs'])
		return m


	selOver = sel_data['Over'] == o
	sel_data_by_over = sel_data[selOver & selWicket]
	m = np.mean(sel_data_by_over['Runs.Remaining'])
	return m

# Function to plot when all wickets in hand
def noWicketRunModelPlot():
	X = np.arange(51)

	Z0 = 238
	b = 0.035

	Y = func(Z0, b, X)
	# print(X)

	plt.plot(X,Y,color='black')

# Function to plot using the mean runs
def noWicketRunDataPlot():
	X = np.arange(51)
	Y = []
	for o in X:
		Y.append(getMeanRun(o,10))
	# print(Y)
	plt.scatter(X,Y,color='blue')

# Function to plot using model with the given wickets
def WicketRunModelPlot(w):
	b = 0.035
	X = np.arange(51)
	Z0 = getMeanRunByWicket(w)
	# print(w)
	# print(Z0)
	Y = func(Z0, b, X)
	# Y = func(Z0, b, X)
	plt.plot(X,Y,color='red')

# Function to get average of maximum runs by overs and wicket
def getAverageMaxRun():
	Z0_list = []
	for w in np.arange(10):
		# WicketRunModelPlot(w)
		Z0_list.append(getMeanRunByWicket(w+1))
	# print(Z0_list)
	return Z0_list
	# AvgRunByOver = []
	# for row in sel_data['Wickets.in.Hand']:
	#     AvgRunByOver.append(Z0_list[row])
	# sel_data['AvgRunByOver'] = AvgRunByOver
	# sel_data['AvgRunByOver'] = sel_data['Wickets.in.Hand'].apply(getMeanRunByWicket)
	# print(sel_data)



# The sum of squared errors loss function, summed across overs and wickets, that we indent to minimize to get the optimum results
def errorFunc(Zopt, train_data):
    squared_errors = []
    Lopt = Zopt[10]
    train_data_run = train_data[0]
    train_data_over = train_data[1]
    train_data_wicket = train_data[2]
    for i in range(len(train_data_run)):
        predicted_run = func2(Zopt[train_data_wicket[i]-1], train_data_over[i], Lopt)
        squared_errors.append(math.pow(predicted_run - train_data_run[i], 2))
        print(predicted_run)
    return np.sum(squared_errors)


# Estimate some Z values manually from data that we will optimize using function
Z0_values = getAverageMaxRun() # 10 Z values for each wicket
Z0_values.append('5') # 1 L value at the end - the constant slope across all data


# Use the Scipy Optimize library's Minimize function ob the target function, with parameters of Z & L, and the data required as arguments, with BFGS algorithm
sol = minimize(errorFunc, Z0_values, args=[sel_data['Runs.Remaining'].values, sel_data['Over'].values, sel_data['Wickets.in.Hand'].values], method='L-BFGS-B')
# print(sol)

# the output of minimize
Z0_final = sol.x
min_error = sol.fun

print("*"*30)

print("\n\nThe minimized total error = "+str(min_error))


# Plot the data of fraction of resources available predicted with optimized values
fig = plt.figure(1)
plt.xlabel('Overs to go')
plt.ylabel('Resource remaining %')

#This is the maximum possible resource avalable prediction of Z
Z50 = func2(Z0_final[9], 50, Z0_final[10])

over_axis = np.arange(51)
modified_over_axis = 50.0 - over_axis

# For each wicket, we plot graphs using the function model to predict with optimized values over wickets
print("\nOptimized values of Z (rounded to integer) are:\n")
for i in range(10):
    y = 100*func2(Z0_final[i], over_axis, Z0_final[10])/Z50
    zf = "{:.0f}".format(Z0_final[i])
    print('Z('+str(i+1)+') = '+str(zf))
    plt.plot(over_axis, y, label='Z('+str(i+1)+') = '+str(zf))
    plt.legend()

print("\n")

print("\nOptimized values of L is:"+str(Z0_final[10]))

print("\n\n")
print("*"*30)

# This is the linear slope over all graphs
slope = -2*over_axis + 100
plt.plot(modified_over_axis, slope, 'black')


fig.suptitle('Assignment 1 (E0 259, Data Analytics, August 2019)', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Total error = '+str(min_error))

plt.show()

# The output also stored as a PDF file
# plt.savefig('Nabarun_Sarkar_Assignment1_output_plot.pdf')



# noWicketRunModelPlot()
# noWicketRunDataPlot()
# plt.xlabel('Overs remaining')
# plt.ylabel('Average runs obtainable')
# plt.show()
# plt.scatter(X,Y)
# plt.plot(X,Y_pred,color='red')
# plt.show()
