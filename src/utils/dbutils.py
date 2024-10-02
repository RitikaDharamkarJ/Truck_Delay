import configparser
import requests

CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'

# Function to read the GitHub API URL from a config file
def read_github_config(CONFIG_FILE_PATH):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)
    
      # Check if 'API' section exists
    if 'API' not in config:
        raise configparser.NoSectionError('API')
    
    github_config = {
        'github_url': config.get('API', 'github_url')
    }
    return github_config

# Function to fetch content from the GitHub API
def fetch_github_contents(CONFIG_FILE_PATH):
    github_config = read_github_config(CONFIG_FILE_PATH)
    api_url = github_config['github_url']
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()  # Return the JSON content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from GitHub: {e}")
        return None

# Example function to display fetched contents
def display_github_contents(CONFIG_FILE_PATH):
    contents = fetch_github_contents(CONFIG_FILE_PATH)
    if contents is not None:
        for item in contents:
            print(f"Name: {item['name']}, Type: {item['type']}, Download URL: {item.get('download_url', 'N/A')}")
    else:
        print("No contents retrieved.")

# Usage Example
if __name__ == "__main__":
    CONFIG_FILE_PATH = 'C:/Truck_Delay_Classification/src/config/config.ini'
    display_github_contents(CONFIG_FILE_PATH)
