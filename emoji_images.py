/Users/panagiwths/Desktop/assignments/genemoji/emoji_images

import os
import requests



# Save images to this folder
output_dir = "emoji_images"
os.makedirs(output_dir, exist_ok=True)

with open("emoji_fuzzy_scores.json", "r", encoding="utf-8") as f:
    emoji_data = json.load(f)

# Convert "U+1F3CE FE0F" → "1f3ce-fe0f"
def unicode_to_filename(unicode_str):
    return "-".join(code.replace("U+", "").lower() for code in unicode_str.strip().split())

# Download from Twemoji CDN
def download_emoji(unicode_str, emoji_char):
    filename = unicode_to_filename(unicode_str) + ".png"
    url = f"https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{filename}"
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        print(f"✅ Already exists: {filename}")
        return
    print(f"⬇️ Downloading {emoji_char} → {filename}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved to {path}")
    else:
        print(f"❌ Failed to download {emoji_char} ({unicode_str})")

# Run for each emoji
for entry in emoji_data:
    download_emoji(entry["unicode"], entry["emoji"])