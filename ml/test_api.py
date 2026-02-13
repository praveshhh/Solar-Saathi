import requests
import json

def test_prediction():
    url = "http://localhost:8000/predict"
    payload = {
        "latitude": 19.076,
        "longitude": 72.878,
        "module_type": "Mono-Si",
        "mounting_type": "fixed_tilt",
        "tilt_angle": 25.0,
        "panel_wattage": 400
    }
    
    print(f"🚀 Testing Solar Saathi API at {url}...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction Successful!")
            print(json.dumps(result["data"], indent=2))
        else:
            print(f"❌ Prediction Failed: {response.text}")
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")

if __name__ == "__main__":
    test_prediction()
