import numpy as np
import pandas as pd
import os

# Define the names of your numpy files
# numpy_files = ['/isilon/datalake/cialab/scratch/cialab/okoyun/folds/ok_val0.npy', '/isilon/datalake/cialab/scratch/cialab/okoyun/folds/ok_val1.npy', '/isilon/datalake/cialab/scratch/cialab/okoyun/folds/ok_val2.npy']
numpy_files = ['/isilon/datalake/cialab/scratch/cialab/okoyun/folds/ok_test.npy']
# Process each file
for file_name in numpy_files:
    # Load the numpy array
    data = np.load(file_name)
    
    # Create a DataFrame
    # Adjust column names based on your actual data structure
    df = pd.DataFrame(data, columns=['path', 'odx_HL', 'odx_score', 'grade', 'length'])
    
    # Create CSV filename (replace .npy with .csv)
    csv_file = file_name.replace('.npy', '.csv')
    
    # Save as CSV
    df.to_csv(csv_file, index=False)
    
    print(f"Converted {file_name} to {csv_file}")