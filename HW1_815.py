#Author: Andres Abreu
#Date: 2/1/2019
#Homework #1 PSHX 815

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np 

#open .dat file
file = open('HW1_input(1).dat')

#placeholder for entries in .dat file
class_data = []

#fill class_data with columns in dat file
for line in file:
   class_data += [line.split()]

#Create list for the grades
grades = [x[3] for x in class_data]
size = len(grades)
 
#Loop over grades and get mean
sum = 0

for grade in grades:
    sum+=float(grade)

mean = sum/size 

#Loop over grades to get standard deviation 
temp_var = 0

for grade in grades:

    temp_diff_sqr = (float(grade)-mean)**2
    temp_var += temp_diff_sqr

variance = temp_var/size
std_dev = sqrt(variance)

#Sort data in descending grade order
data_clone = list(class_data) #clone original list
mod_list = []

for i in range(size):

    temp_grade = 0
    row=[]

    for i in range(len(data_clone)):
        grade = data_clone[i][3]    
        
        if grade > temp_grade:
            temp_grade = grade
            row = data_clone[i]

    data_clone.remove(row)
    mod_list.append(row)

#Display results
for i in range(size):
    print(mod_list[i])

print("Mean Grade: " + str(round(mean,2)))
print("Standard Deviation: " + str(round(std_dev,2)))

#Plot distribution as a histogram and save as .pdf
f = plt.figure()
axes = plt.gca()

axes.set_ylim([0.0,7.0])
plt.ylabel("Students")
plt.xlabel("Grades")
plt.title("Class Grade Distribution")
plt.hist(np.array(grades).astype(np.float), normed=False, bins=10)

f.savefig("grade_dist.pdf", bbox_inches='tight')
