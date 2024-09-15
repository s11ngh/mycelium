import modal

# Define the image with system-level and Python dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("flask")  # Install Python dependencies
)

# Define the Modal app
app = modal.App("pysyft-server-mycelium")

# Define a Modal function using the custom image
@app.function(image=image)
def run_syft_server():
    import os
    import subprocess
    
    # Set the FLASK_APP environment variable if needed
    os.environ["FLASK_APP"] = "app.py"  # or whichever file contains your Flask app
    
    # Start the PyGrid server (assuming Flask app is in `app.py`)
    subprocess.run(["flask", "run", "--host=0.0.0.0", "--port=5000"])

# Define the entrypoint for local execution
@app.local_entrypoint()
def main():
    # Run the function directly (no .call())
    run_syft_server.local() # Just call the function directly
