import requests
import os
import time

# --- CONFIGURATION ---
URL = "https://coconut-ai-backend.onrender.com/predict"

def run_test():
    # 1. Server Health Check (Waking up the server)
    health_url = URL.replace("/predict", "/")
    print(f"\n🔍 Checking Server: {health_url}")
    
    try:
        start_time = time.time()
        health_res = requests.get(health_url, timeout=60)
        duration = round(time.time() - start_time, 2)
        print(f"✅ Server is Live! (Response time: {duration}s)")
        print(f"📊 Server Status: {health_res.json()}")
    except Exception as e:
        print(f"⚠️ Server is taking too long to wake up. Try again in 10 seconds.")
        return

    print("-" * 50)

    # 2. Auto-detect Images in the folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    files_in_folder = [f for f in os.listdir('.') if f.lower().endswith(valid_extensions)]

    if not files_in_folder:
        print("❌ Walang nakitang image (.jpg, .png) sa folder na ito.")
        print("Maglagay ka muna ng photo sa D:\\backend3\\")
        return

    print("📷 Mga nakitang photo sa folder mo:")
    for i, filename in enumerate(files_in_folder):
        print(f"   [{i + 1}] {filename}")

    # 3. Choose Photo
    try:
        choice = int(input("\n👉 Pumili ng number ng photo na i-uupload: "))
        selected_file = files_in_folder[choice - 1]
    except (ValueError, IndexError):
        print("❌ Invalid selection. Pls run the script again.")
        return

    # 4. Upload and Predict
    print(f"\n🚀 Processing: {selected_file}...")
    try:
        with open(selected_file, 'rb') as img:
            files = {'file': img}
            data = {'address': 'Abuyog IT Dept - Testing Site'} # Default test location
            
            response = requests.post(URL, files=files, data=data)
            
            if response.status_code == 200:
                res = response.json()
                print("\n" + "="*30)
                print("🎯 PREDICTION SUCCESS")
                print("="*30)
                print(f"Variety    : {res.get('variety_name')}")
                print(f"Confidence : {res.get('confidence')}%")
                print(f"Definition : {res.get('definition')}")
                print(f"Lifespan   : {res.get('lifespan')}")
                print(f"Location   : {res.get('address')}")
                print("="*30 + "\n")
            else:
                print(f"❌ API Error ({response.status_code}): {response.text}")

    except Exception as e:
        print(f"🔥 Error: {e}")

if __name__ == "__main__":
    run_test()