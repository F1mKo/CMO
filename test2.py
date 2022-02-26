import pylab
import random

number_of_trials = 1000
number_of_customer_per_hour = 10

## Here we simulate the interarrival time of the customers

list_of_values = []
for i in range(number_of_trials):
    list_of_values.append(random.expovariate(number_of_customer_per_hour))

mean=pylab.mean(list_of_values)
std=pylab.std(list_of_values)
print("Trials =", number_of_trials, "times")
print("Mean =", mean)
print("Standard deviation =", std)

pylab.hist(list_of_values,20)
pylab.xlabel('Value')
pylab.ylabel('Number of times')
pylab.show()
