from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "sensor_database"
COLLECTION_NAME = "environment_data"


def load_sensor_data():
    client = None

    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)

        # Select database and collection
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # Load data from MongoDB
        cursor = collection.find(
            {},
            {
                "_id": 0,
                "light": 1,
                "temperature": 1,
                "humidity": 1,
                "moisture": 1
            }
        )

        data = list(cursor)
        df = pd.DataFrame(data)

        if df.empty:
            print("No data found in the collection.")
            return None

        print("\nSensor Data:")
        print(df)

        return df

    except Exception as e:
        print(f"Error loading data from MongoDB: {e}")
        return None

    finally:
        if client:
            client.close()


def display_graphs(df):
    if df is None or df.empty:
        print("No data available to display graphs.")
        return

    # Make sure all sensor columns are numeric
    sensor_columns = ["light", "temperature", "humidity", "moisture"]

    for column in sensor_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows with missing values
    df = df.dropna(subset=sensor_columns)

    if df.empty:
        print("No valid numeric data available to plot.")
        return

    # Create an index to represent reading number
    df["reading_number"] = range(1, len(df) + 1)

    # Plot all sensor values in one graph
    plt.figure(figsize=(10, 6))

    plt.plot(df["reading_number"], df["light"], marker="o", label="Light")
    plt.plot(df["reading_number"], df["temperature"], marker="o", label="Temperature")
    plt.plot(df["reading_number"], df["humidity"], marker="o", label="Humidity")
    plt.plot(df["reading_number"], df["moisture"], marker="o", label="Moisture")

    plt.title("Sensor Data from MongoDB")
    plt.xlabel("Reading Number")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Display each sensor in a separate graph
    for column in sensor_columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["reading_number"], df[column], marker="o")
        plt.title(f"{column.capitalize()} Readings")
        plt.xlabel("Reading Number")
        plt.ylabel(column.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = load_sensor_data()
    display_graphs(df)