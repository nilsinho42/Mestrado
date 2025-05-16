import requests
import json

def check_endpoint(url):
    """Check the API endpoint and print response data"""
    print(f"\nChecking {url}...")
    try:
        response = requests.get(url, timeout=10)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                # Try to parse as JSON
                data = response.json()
                print("Response data:")
                # Pretty print JSON data
                if isinstance(data, dict):
                    for key, value in data.items():
                        print(f"  {key}: {value}")
                else:
                    print(json.dumps(data, indent=2))
            except:
                # If not JSON, print as text
                print("Response (text):")
                print(response.text[:500]) # Limit to first 500 chars
        else:
            print(f"Error response: {response.text[:200]}")
    except Exception as e:
        print(f"Failed to connect: {str(e)}")
    
    

def main():
    import os 

    # Get endpoints:
    aws_fargate_endpoint = "https://t5tj33xnb3.us-east-2.awsapprunner.com/docs"
    azure_fargate_endpoint = "https://deepsort-tracker.icyfield-c44c0ae9.brazilsouth.azurecontainerapps.io/docs"
    gcp_fargate_endpoint = "https://gcp-deepsort-tracker-504277420866.us-central1.run.app/docs"

    endpoints = [aws_fargate_endpoint, azure_fargate_endpoint, gcp_fargate_endpoint]
    for endpoint in endpoints:
        print(endpoint)
        check_endpoint(endpoint)
        print("-" * 50)

if __name__ == "__main__":
    main() 