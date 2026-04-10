import os

def load_logs(folder="data/logs"):

    logs = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        with open(path, "r") as f:

            logs.append(f.read())

    return logs


if __name__ == "__main__":

    logs = load_logs()

    print("Loaded logs:")

    for i, log in enumerate(logs):

        print(f"\nLog {i+1}:")

        print(log)