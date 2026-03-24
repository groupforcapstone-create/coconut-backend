import os
import sys
import requests

# --- CONFIGURATION ---
URL = "https://coconut-ai-backend.onrender.com/predict"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def pick_image_from_folder(folder):
    files_in_folder = [
        f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS)
    ]
    if not files_in_folder:
        print("❌ Walang nakitang image (.jpg, .png, .webp) sa folder na ito.")
        return None

    print("📷 Mga nakitang photo sa folder mo:")
    for i, filename in enumerate(files_in_folder):
        print(f"   [{i + 1}] {filename}")

    try:
        choice = int(input("\n👉 Pumili ng number ng photo na i-uupload: "))
        return os.path.join(folder, files_in_folder[choice - 1])
    except (ValueError, IndexError):
        print("❌ Invalid selection.")
        return None


def get_image_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return pick_image_from_folder(".")


def upload_only(image_path):
    if not image_path:
        return

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    if not image_path.lower().endswith(VALID_EXTENSIONS):
        print("❌ Invalid file type. Use .jpg, .jpeg, .png, or .webp.")
        return

    print(f"\n🚀 Uploading: {image_path} ...")
    try:
        with open(image_path, "rb") as img:
            files = {"file": img}
            data = {"address": "Abuyog IT Dept - Testing Site"}
            response = requests.post(URL, files=files, data=data)

        print(f"✅ Upload complete. Status: {response.status_code}")
        print(response.text)
    except Exception as e:
        print(f"🔥 Error: {e}")


if __name__ == "__main__":
    upload_only(get_image_path())
