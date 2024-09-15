import modal
import pandas as pd

# Define the image with system-level and Python dependencies (include pandas)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas")  # Install pandas
)

# Define the Modal app
app = modal.App("csv-reader")

# Define a Modal function to read a CSV file and split it
@app.function(image=image)
def read_and_split_csv_file():
    # Path to the CSV file (adjust as needed)
    csv_file_path = 'customers.csv'  # Update this path
    
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file_path)
    
    # Split the dataframe into two equal partitions
    df1 = df.iloc[:len(df)//2]
    df2 = df.iloc[len(df)//2:]
    
    # Print or return the dataframes
    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    
    return df1, df2

# Define the entrypoint for local execution
@app.local_entrypoint()
def main():
    # Call the function directly and print the result
    df1, df2 = read_and_split_csv_file.local()
    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
