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

# Strong bias inject — Males much more likely selected
df['selected'] = 0

for i in range(n):
    score = df.loc[i, 'test_score']
    exp   = df.loc[i, 'experience_years']
    gender = df.loc[i, 'gender']

    if gender == 1:  # Male
        # Male: easy to get selected
        if score > 60 and exp > 1:
            df.loc[i, 'selected'] = 1
    else:  # Female
        # Female: very hard to get selected (strong bias)
        if score > 85 and exp > 8:
            df.loc[i, 'selected'] = 1

df.to_csv('data.csv', index=False)
print("✅ Biased dataset created!")
print(f"Males selected: {df[df['gender']==1]['selected'].sum()}")
print(f"Females selected: {df[df['gender']==0]['selected'].sum()}")
