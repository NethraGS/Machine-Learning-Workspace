import numpy as np
import pandas as pd
np.random.seed(42)
n_customers = 500
first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 
               'Michael', 'Linda', 'William', 'Elizabeth', 'David', 'Susan',
               'Richard', 'Jessica', 'Joseph', 'Sarah', 'Thomas', 'Karen',
               'Charles', 'Nancy', 'Christopher', 'Lisa', 'Daniel', 'Margaret']

last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
              'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez',
              'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore',
              'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White']
data = {
    'first_name': np.random.choice(first_names, n_customers),
    'last_name' : np.random.choice(last_names, n_customers),
    'age': np.random.randint(18, 70, n_customers),
    'annual_income': np.random.random_sample(500)*1000,
    'spending_score': np.random.randint(1, 100, n_customers),
}

df = pd.DataFrame(data)
print(df)
