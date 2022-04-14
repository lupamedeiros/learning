import wandb
import pandas as pd

# collumns used
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hour_per_week', 'native_country',
            'high_income']

# import the data addinf the column's labels
income = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                     header=None,
                     names=columns)

# Convert to CSV
income.to_csv("build/raw_data.csv", index=False)