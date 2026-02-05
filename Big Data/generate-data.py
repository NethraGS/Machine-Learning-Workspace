import time
import random
from datetime import datetime

while True:
    with open("data/stream/censor.csv", "a") as file:
        temperature = round(random.uniform(20, 40), 2)
        timestamp = datetime.now()

        file.write(f"{timestamp},{temperature}\n")

    time.sleep(1)
