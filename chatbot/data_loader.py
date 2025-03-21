import pandas as pd
import os

def load_laptop_data():
    """
    Load laptop data from CSV file
    """
    try:
        # Get the path to the CSV file
        csv_path = os.path.join(os.path.dirname(__file__), 'laptop_data.csv')
        
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        
        # Convert DataFrame to a list of dictionaries for easier processing
        laptops = df.to_dict('records')
        
        print(f"Successfully loaded {len(laptops)} laptops from dataset")
        return laptops
    except Exception as e:
        print(f"Error loading laptop data: {e}")
        # Return empty list on error
        return [] 