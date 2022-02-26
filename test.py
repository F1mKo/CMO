import pylab
import random

number_of_trials =100
## Here we simulate the repeated throwing of a single six-sided die
list_of_values = []
for i in range(number_of_trials):
    list_of_values.append(random.normalvariate(7, 2.4))

print("Trials =", number_of_trials, "times.")
print("Mean =", pylab.mean(list_of_values))
print("Standard deviation =", pylab.std(list_of_values))

pylab.hist(list_of_values, pylab.arange(1.5, 13.5, 1.0))
pylab.xlabel('Value')
pylab.ylabel('Number of times')
pylab.show()
