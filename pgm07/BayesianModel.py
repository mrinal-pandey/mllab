# Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set. You can use Java/Python ML library classes/API.

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data = pd.read_csv('bayesian-model.csv')
data = pd.DataFrame(data)

model = BayesianModel([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('cholestrol', 'heartdisease'),
    ('Family', 'heartdisease'),
    ('Lifestyle', 'diet'),
    ('diet', 'cholestrol')
])

model.fit(data, estimator = MaximumLikelihoodEstimator)
hd_inf = VariableElimination(model)

print()
print('Age : 0 - super senior, 1 - senior, 2 - middle age, 3 - youth, 4 - teen')
print('Gender : 0 - male. 1 - female')
print('Family history : 0 - no, 1 - yes')
print('cholestrol : 0 - high, 1 - borderline, 2 - moderate')
print('Lifestyle : 0 - athelete, 1 - active, 2 - moderate, 3 - sedentary')
print('diet : 0 - high, 1 - medium')
print()

q = hd_inf.query(variables = ['heartdisease'], joint = False, evidence = {
    'age' : int(input('Enter age: ')),
    'Gender' : int(input('Enter gender: ')),
    'Family' : int(input('Enter Family History: ')),
    'cholestrol' : int(input('Enter cholestrol: ')),
    'Lifestyle' : int(input('Enter Lifestyle: ')),
    'diet' : int(input('Enter diet: ')),
})

print(q['heartdisease'])
