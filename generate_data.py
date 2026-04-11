# generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

df = pd.DataFrame({
    'experience_years': np.random.randint(0, 15, n),
    'test_score':       np.random.randint(50, 100, n),
    'interview_score':  np.random.randint(40, 100, n),
    'skills_count':     np.random.randint(1, 10, n),
    'gender':           np.random.choice([0, 1], n),
})

df['selected'] = (
    (df['test_score'] > 70) &
    (df['experience_years'] > 2) &
    ((df['gender'] == 1) | (df['interview_score'] > 80))
).astype(int)

df.to_csv('data.csv', index=False)
print("✅ data.csv successfully created!")
print(df.head())