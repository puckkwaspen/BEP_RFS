import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ASCA import ASCA

# Load data
df = pd.read_csv("BEP_imputed.csv")

# Select dynamic features
features = ['ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium',
            'Magnesium', 'Weight (kg)', 'BMI', 'Temperature (C)',
            'Systolic', 'Diastolic', 'Leucocytes']
X = df[features].values
X = np.asarray(X)

# Define and encode factors
df['SEX'] = pd.Categorical(df['SEX'])
df['SEQUENCE'] = pd.Categorical(df['SEQUENCE'])

F = df[['SEX', 'SEQUENCE']].values
F = np.asarray(F)

ASCA = ASCA()
ASCA.fit(X,F,interactions = [[0,1]])
ASCA.plot_factors()
ASCA.plot_interactions()