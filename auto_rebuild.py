import os
import subprocess
import time

LOG_FOLDER = "data/logs"

last_modified_time = 0

print("Watching log folder for changes...")

while True:

    current_time = max(
        os.path.getmtime(os.path.join(LOG_FOLDER, f))
        for f in os.listdir(LOG_FOLDER)
        if f.endswith(".txt")
    )

    if current_time != last_modified_time:

        print("New logs detected. Rebuilding vector store...")

        subprocess.run(
            ["python", "build_vector_store.py"]
        )

        last_modified_time = current_time

    time.sleep(5)