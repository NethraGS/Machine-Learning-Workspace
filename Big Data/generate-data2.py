import time
import random
from datetime import datetime

while True:
    with open("data/stream/study.csv", "a") as file:
        hours_studied = round(random.uniform(0, 10), 2)  # 0–10 hours
        timestamp = datetime.now()

        file.write(f"{timestamp},{hours_studied}\n")

    time.sleep(1)
