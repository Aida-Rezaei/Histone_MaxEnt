"""
Download ENCODE bigWig files listed in results/encode_metadata.csv
"""

from pathlib import Path
import pandas as pd
import requests
import time

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_ROOT / "results" / "encode_metadata.csv"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = PROJECT_ROOT / "results" / "download_log.csv"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Read metadata
# --------------------------------------------------

df = pd.read_csv(METADATA_FILE)

print("=" * 60)
print(f"Found {len(df)} files to download")
print("=" * 60)

log = []

# --------------------------------------------------
# Download loop
# --------------------------------------------------

for i, row in df.iterrows():

    accession = row["File_Accession"]
    url = row["Download_URL"]

    output_file = DOWNLOAD_DIR / f"{accession}.bigWig"

    print(f"\n[{i+1}/{len(df)}] {accession}")

    # Skip existing files
    if output_file.exists():
        print("Already downloaded.")

        log.append({
            "File_Accession": accession,
            "Status": "Already exists"
        })

        continue

    try:

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_file, "wb") as f:

            for chunk in response.iter_content(chunk_size=1024 * 1024):

                if chunk:

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        print(
                            f"\rDownloading... {percent:5.1f}%",
                            end=""
                        )

        print("\nFinished.")

        log.append({
            "File_Accession": accession,
            "Status": "Downloaded"
        })

        # small pause to be polite to ENCODE
        time.sleep(1)

    except Exception as e:

        print(f"\nFAILED: {e}")

        log.append({
            "File_Accession": accession,
            "Status": f"FAILED: {e}"
        })

# --------------------------------------------------
# Save log
# --------------------------------------------------

pd.DataFrame(log).to_csv(LOG_FILE, index=False)

print("\n" + "=" * 60)
print("Download complete.")
print(f"Files saved to:\n{DOWNLOAD_DIR}")
print(f"Log written to:\n{LOG_FILE}")
print("=" * 60)