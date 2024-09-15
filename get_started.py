import modal
import pandas as pd

# Define the image with system-level and Python dependencies (include pandas)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas")  # Install pandas
)

# Define the Modal app
app = modal.App("csv-reader")

# Define a Modal function to read a CSV file using pandas
@app.function(image=image)
def read_csv_file():
    # Path to the CSV file (adjust as needed)
    csv_file_path = 'customers.csv'  # Update this path
    
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file_path)
    
    # Print the dataframe (or return it if you want to see the content)
    print(df)
    return df.head()  # Return the first few rows of the dataframe

# Define the entrypoint for local execution
@app.local_entrypoint()
def main():
    # Call the function directly and print the result
    result = read_csv_file.local()
    print(result)
