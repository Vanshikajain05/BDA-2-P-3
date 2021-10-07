import warnings

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

print("SECTION-A")

titanic = pd.read_csv(r"C:\titanic dataset.csv")
print(" ")

print("1. Find out the overall chance of survival for a Titanic passenger.")
print(" ")

print("Total number of passengers survived are: ", titanic['survived'].value_counts()[1])
print("Overall chance of survival: ", titanic['survived'].value_counts(normalize=True)[1] * 100)

print(" ")
print("Q2. Find out the chance of survival for a Titanic passenger based on their sex and plot it.")
print(" ")
# Difference in survival between Male and Female.  How many females survived and how many males?

male_female_survival = titanic.groupby('sex').sum()['survived']

# total number of male passengers and total number of female passengers

total_male_female = titanic['sex'].value_counts()

male_female = male_female_survival / total_male_female
print(male_female)
probability = male_female.plot(kind='bar', title='Chance of Survival per Gender')
probability.set_xlabel('Sex')
probability.set_ylabel('Probability')
plt.show()

print(" ")
print("Q3. Find out the chance of survival for a Titanic passenger by traveling class wise and plot it. ")
print(" ")
# Difference in survival based on class

class_wise_survival = titanic.groupby('pclass').sum()['survived']
total_class_wise = titanic['pclass'].value_counts()
survived = class_wise_survival / total_class_wise
print(survived)

# plot the resulting probabilities

class_survival_prob = survived.plot(kind='bar', title='Chance of Survival per Class', color="red")

class_survival_prob.set_ylabel('Probability of Survival')
class_survival_prob.set_xlabel('Class')
plt.show()

print(" ")

print("Q4: Find out the average age for a Titanic passenger who survived by passenger class and sex. ")

print(" ")

fig = plt.figure(figsize=(12, 5))
fig.add_subplot(121)
plt.title('average age for a Titanic passenger who survived by pclass and sex')
sns.barplot(data=titanic, x='age', y='pclass', hue='sex')

meanAgeMale = round(titanic[(titanic['sex'] == "male")]['age'].groupby(titanic['pclass']).mean(), 2)
meanAgeFeMale = round(titanic[(titanic['sex'] == "female")]['age'].groupby(titanic['pclass']).mean(), 2)

print(pd.concat([meanAgeMale, meanAgeFeMale], axis=1, keys=['Male', 'Female']))

print(" ")

print("Q5. Find out the chance of survival for a Titanic passenger based on number of siblings the passenger had on the ship and plot it.")
print(" ")

sns.barplot(x="sibsp", y="survived", data=titanic)
plt.show()
print("Percentage of SibSp 0 who survived is", titanic["survived"][titanic["sibsp"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp 1 who survived is", titanic["survived"][titanic["sibsp"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp 2 who survived is", titanic["survived"][titanic["sibsp"] == 2].value_counts(normalize = True)[1]*100)

print(" ")

print("Q6. Find out the chance of survival for a Titanic passenger based on number of parents/children the passenger "
      "had on the ship and plot it.")

sns.barplot(x="parch", y="survived", data=titanic)
plt.show()
print("Percentage of parch 0 who survived is", titanic["survived"][titanic["parch"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of parch 1 who survived is", titanic["survived"][titanic["parch"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of parch 2 who survived is", titanic["survived"][titanic["parch"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of parch 3 who survived is", titanic["survived"][titanic["parch"] == 3].value_counts(normalize = True)[1]*100)


print(" ")
print("Q7. Plot out the variation of survival and death amongst passengers of different age. ")
print(" ")

# number of survivors in each age group

each_age_group = titanic.groupby('age').sum()['survived']

# number of passengers in each age group

num_in_group = titanic['age'].value_counts(sort=False)

# chance of survival for those in each age group

age_survival = each_age_group / num_in_group
age_survival.sort_values(inplace=True)

# let's take a look at a visual for age by plotting the probabilities

ap = age_survival.plot(kind='bar', title='Chance of Survival per Age Group')

ap.set_ylabel('Age group')
ap.set_xlabel('Probability of Survival')
plt.show()

print(" ")

print("Q8. Plot out the variation of survival and death with age amongst passengers of different passenger classes.")

print(" ")
print("Q9. Find out the survival probability for a Titanic passenger based on title from the name of passenger.")
print(" ")

print(" ")