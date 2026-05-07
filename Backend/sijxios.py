import json
import random
import threading
from datetime import datetime, timezone
from statistics import median
from statsmodels.tsa.arima.model import ARIMA
import paho.mqtt.client as mqtt
from pymongo import MongoClient, ASCENDING, DESCENDING
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
from prophet import Prophet


# -----------------------------
# MongoDB configuration
# -----------------------------

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "iot_project"
MONGO_COLLECTION_NAME = "sensor_data"
PREDICTION_MONGO_URI = "mongodb://localhost:27017/"
PREDICTION_DATABASE_NAME = "sensor_database"
PREDICTION_COLLECTION_NAME = "environment_data"


mongo_client = MongoClient(MONGO_URI)
prediction_mongo_client = MongoClient(PREDICTION_MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
sensor_collection = db[MONGO_COLLECTION_NAME]
prediction_db = prediction_mongo_client[PREDICTION_DATABASE_NAME]
prediction_collection = prediction_db[PREDICTION_COLLECTION_NAME]


# -----------------------------
# MQTT configuration
# -----------------------------

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/gateway-001/telemetry"
MQTT_TOPIC2 = "iot/gateway-001/updateparams"

mqtt_client = mqtt.Client(client_id="iot-server")
params_lock = threading.Lock()
PLANT_PARAMS = {
    "fieldCapacity": 80.0,
    "sampleFrequency": 10,
    "soilVolumeLiters": 5.0,
    "targetSoilMoisture": 55.0,
    "wiltingPoint": 30.0,
}


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # okay for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


METRIC_FIELD_NAMES = {
    "temperature": "temperature",
    "light": "light",
    "moisture": "moisture",
    "humidity": "humidity"
}

prediction_model_lock = threading.Lock()
prediction_models = {}
prediction_cache = {}
prediction_training_error = None


def newest_documents(limit):
    return (
        sensor_collection
        .find()
        .sort([
            ("timestamp", DESCENDING),
            ("_id", DESCENDING),
        ])
        .limit(limit)
    )


def serialize_timestamp(value):
    if isinstance(value, datetime):
        return value.isoformat()

    if value is None:
        return None

    return str(value)



def read_numeric_field(document, field_names):
    for field_name in field_names:
        value = document.get(field_name)

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue

    return None



def store_sensor_data(data):
    """
        Stores one MQTT packet in MongoDB.
    """
    
    
    decoded = data["decoded"]
    document = {
        "gateway_id": data.get("gateway_id", "unknown"),
        "timestamp": data.get("timestamp"),
        "received_at": datetime.now(timezone.utc),
        "light": decoded["light"],
        "temperature": decoded["temperature"],
        "humidity": decoded["humidity"],
        "moisture": decoded["moisture"],
    }

    # print(document)

    result = sensor_collection.insert_one(document)

    # print(f"Data stored in MongoDB with id: {result.inserted_id}")


def serialize_document(document):
    """
    Converts MongoDB document to JSON-friendly format.
    MongoDB ObjectId and datetime are not directly JSON serializable.
    """

    return {
        "id": str(document["_id"]),
        "gateway_id": document.get("gateway_id"),
        "temperature": document.get("temperature"),
        "humidity": document.get("humidity"),
        "light": document.get("light"),
        "moisture": document.get("moisture"),
        "received_at": document.get("timestamp")
    }


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
        client.subscribe(MQTT_TOPIC2)
        print(f"Subscribed to topics: {MQTT_TOPIC}, {MQTT_TOPIC2}")
    else:
        print(f"Failed to connect to MQTT broker. Return code: {rc}")


def parse_update_params_payload(payload: str):
    pairs = [part.strip() for part in payload.split(",") if part.strip()]
    parsed = {}

    for pair in pairs:
        if ":" not in pair:
            continue

        key, value = [item.strip() for item in pair.split(":", 1)]
        if key not in PLANT_PARAMS:
            continue

        try:
            parsed[key] = float(value)
        except ValueError:
            return None

    if not parsed:
        return None

    if "sampleFrequency" in parsed:
        parsed["sampleFrequency"] = int(parsed["sampleFrequency"])

    return parsed


def update_local_params(params):
    with params_lock:
        for key, value in params.items():
            if key in PLANT_PARAMS:
                PLANT_PARAMS[key] = value

        return json.dumps(PLANT_PARAMS, indent=2)


def on_message(client, userdata, message):
    try:
        payload = message.payload.decode().strip()

        if message.topic == MQTT_TOPIC2:
            parsed = parse_update_params_payload(payload)
            if parsed is None:
                print(f"Unable to parse update params payload: {payload}")
                return

            update_local_params(parsed)
            return

        data = json.loads(payload)
        store_sensor_data(data)

    except KeyError as e:
        print(f"Missing field in MQTT packet: {e}")

    except ValueError as e:
        print(f"Invalid numeric value in MQTT packet: {e}")

    except Exception as e:
        print(f"Error processing MQTT message: {e}")


def start_mqtt_client():
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    mqtt_client.loop_forever()



@app.get("/api/sensor-data")
def get_sensor_data(limit: int = 50):
    limit = min(limit, 500)

    documents = newest_documents(limit)

    return [serialize_document(document) for document in documents]


@app.get("/api/latest")
def get_latest_sensor_data():
    document = sensor_collection.find_one(
        sort=[("received_at", DESCENDING),("timestamp", DESCENDING),("_id", DESCENDING)])

    if document is None:
        return {"message": "No data available"}
    
    # print(serialize_document(document))
    return serialize_document(document)



def read_timestamp(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)

            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    return None


def load_prediction_training_series():
    series = {
        key: {
            "timestamps": [],
            "values": [],
        }
        for key in METRIC_FIELD_NAMES
    }

    documents = prediction_collection.find().sort([
        ("timestamp", ASCENDING),
        ("_id", ASCENDING),
    ])

    for document in documents:
        timestamp = read_timestamp(document.get("timestamp"))

        for key, field_name in METRIC_FIELD_NAMES.items():
            value = read_numeric_field(document, [field_name])

            if value is None:
                continue

            series[key]["timestamps"].append(timestamp)
            series[key]["values"].append(value)

    return series


def collect_training_series(documents):
    series = {
        key: {
            "timestamps": [],
            "values": [],
        }
        for key in METRIC_FIELD_NAMES
    }

    latest_marker = "empty"

    for document in documents:
        latest_marker = str(document.get("_id", latest_marker))
        timestamp = read_timestamp(document.get("timestamp"))

        for key, field_name in METRIC_FIELD_NAMES.items():
            value = read_numeric_field(document, [field_name])

            if value is None:
                continue

            series[key]["timestamps"].append(timestamp)
            series[key]["values"].append(value)

    return series, latest_marker


def has_training_values(series):
    return any(
        len(metric_series["values"]) >= 2
        for metric_series in series.values()
    )


def recent_documents_from(collection, sort_fields, limit):
    documents = list(
        collection
        .find()
        .sort(sort_fields)
        .limit(limit)
    )
    documents.reverse()

    return documents


def load_recent_prediction_training_series(limit):
    live_documents = recent_documents_from(
        sensor_collection,
        [
            ("received_at", DESCENDING),
            ("timestamp", DESCENDING),
            ("_id", DESCENDING),
        ],
        limit,
    )
    live_series, live_marker = collect_training_series(live_documents)

    if has_training_values(live_series):
        return live_series, {
            "source": f"{MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}",
            "marker": live_marker,
        }

    historical_documents = recent_documents_from(
        prediction_collection,
        [
            ("timestamp", DESCENDING),
            ("_id", DESCENDING),
        ],
        limit,
    )
    historical_series, historical_marker = collect_training_series(
        historical_documents
    )

    return historical_series, {
        "source": f"{PREDICTION_DATABASE_NAME}.{PREDICTION_COLLECTION_NAME}",
        "marker": historical_marker,
    }


def normalize_prophet_dates(timestamps, value_count):
    valid_timestamps = [
        timestamp
        for timestamp in timestamps
        if timestamp is not None
    ]

    if len(valid_timestamps) == value_count:
        dates = pd.to_datetime(valid_timestamps, utc=True)
        return pd.DatetimeIndex(dates).tz_convert(None)

    return pd.date_range(
        start="2024-01-01",
        periods=value_count,
        freq=infer_prophet_frequency(timestamps),
    )


def infer_prophet_frequency(timestamps):
    valid_timestamps = sorted(
        timestamp
        for timestamp in timestamps
        if timestamp is not None
    )

    if len(valid_timestamps) < 2:
        return "10s"

    deltas = [
        (valid_timestamps[index] - valid_timestamps[index - 1]).total_seconds()
        for index in range(1, len(valid_timestamps))
    ]
    seconds = max(1, int(round(median(deltas))))

    return f"{seconds}s"


def train_arima_model(values, order=(1, 1, 1)):
    latest = values[-1] if values else None

    if len(values) < 8:
        return {
            "kind": "repeat_latest",
            "latest": latest,
        }

    try:
        model = ARIMA(values, order=order, trend="t")
        return {
            "kind": "arima",
            "model": model.fit(),
            "latest": latest,
        }
    except Exception as error:
        print(f"ARIMA training failed: {error}")
        return {
            "kind": "repeat_latest",
            "latest": latest,
        }


def forecast_arima_model(model_data, steps):
    if steps <= 0 or model_data.get("latest") is None:
        return []

    if model_data.get("kind") != "arima":
        return [round(model_data["latest"], 2) for _step in range(steps)]

    try:
        forecast = model_data["model"].forecast(steps=steps)
        return [round(float(value), 2) for value in forecast]
    except Exception as error:
        print(f"ARIMA prediction failed: {error}")
        return [round(model_data["latest"], 2) for _step in range(steps)]


def train_prophet_model(values, timestamps):
    latest = values[-1] if values else None

    if len(values) < 2:
        return {
            "kind": "repeat_latest",
            "latest": latest,
        }

    try:
        dates = normalize_prophet_dates(timestamps, len(values))
        df = pd.DataFrame({
            "ds": dates,
            "y": values,
        })
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
        )
        model.fit(df)

        return {
            "kind": "prophet",
            "model": model,
            "frequency": infer_prophet_frequency(timestamps),
            "last_date": df["ds"].iloc[-1],
            "latest": latest,
        }
    except Exception as error:
        print(f"Prophet training failed: {error}")
        return {
            "kind": "repeat_latest",
            "latest": latest,
        }


def forecast_prophet_model(model_data, steps):
    if steps <= 0 or model_data.get("latest") is None:
        return []

    if model_data.get("kind") != "prophet":
        return [round(model_data["latest"], 2) for _step in range(steps)]

    try:
        future_dates = pd.date_range(
            start=model_data["last_date"],
            periods=steps + 1,
            freq=model_data["frequency"],
        )[1:]
        forecast = model_data["model"].predict(pd.DataFrame({"ds": future_dates}))

        return [
            round(float(value), 2)
            for value in forecast["yhat"]
        ]
    except Exception as error:
        print(f"Prophet prediction failed: {error}")
        return [round(model_data["latest"], 2) for _step in range(steps)]


def train_least_squares_model(values):
    latest = values[-1] if values else None

    if len(values) < 2:
        return {
            "intercept": latest if latest is not None else 0,
            "latest": latest,
            "slope": 0,
            "value_count": len(values),
        }

    mean_x = (len(values) - 1) / 2
    mean_y = sum(values) / len(values)
    numerator = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(values)
    )
    denominator = sum(
        (index - mean_x) ** 2
        for index, _value in enumerate(values)
    )

    slope = numerator / denominator if denominator else 0

    return {
        "intercept": mean_y - slope * mean_x,
        "latest": latest,
        "slope": slope,
        "value_count": len(values),
    }


def forecast_least_squares_model(model_data, steps):
    latest = model_data.get("latest")

    if steps <= 0 or latest is None:
        return []

    return [
        round(latest + model_data["slope"] * (step + 1), 2)
        for step in range(steps)
    ]


def least_squares_value_at(model_data, index):
    if model_data.get("latest") is None:
        return 0

    if "intercept" in model_data:
        return model_data["intercept"] + model_data["slope"] * index

    value_count = model_data.get("value_count", 1)
    return model_data["latest"] + model_data["slope"] * (
        index - value_count + 1
    )


def mean_value(values):
    return sum(values) / len(values) if values else 0


def build_regression_tree(samples, depth=0, max_depth=3, min_leaf_size=2):
    samples = sorted(samples, key=lambda sample: sample[0])
    values = [sample[1] for sample in samples]

    if depth >= max_depth or len(samples) <= min_leaf_size * 2:
        return {"value": mean_value(values)}

    prefix_sum = [0]
    prefix_square_sum = [0]

    for value in values:
        prefix_sum.append(prefix_sum[-1] + value)
        prefix_square_sum.append(prefix_square_sum[-1] + value ** 2)

    def range_squared_error(start, end):
        count = end - start

        if count <= 0:
            return 0

        total = prefix_sum[end] - prefix_sum[start]
        square_total = prefix_square_sum[end] - prefix_square_sum[start]

        return square_total - (total ** 2 / count)

    best_index = None
    best_loss = None

    for index in range(min_leaf_size, len(samples) - min_leaf_size + 1):
        if samples[index - 1][0] == samples[index][0]:
            continue

        loss = (
            range_squared_error(0, index)
            + range_squared_error(index, len(samples))
        )

        if best_loss is None or loss < best_loss:
            best_index = index
            best_loss = loss

    if best_index is None:
        return {"value": mean_value(values)}

    threshold = (samples[best_index - 1][0] + samples[best_index][0]) / 2
    left_samples = samples[:best_index]
    right_samples = samples[best_index:]

    if not left_samples or not right_samples:
        return {"value": mean_value(values)}

    return {
        "threshold": threshold,
        "left": build_regression_tree(
            left_samples,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
        ),
        "right": build_regression_tree(
            right_samples,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
        ),
    }


def predict_regression_tree(tree, index):
    if "value" in tree:
        return tree["value"]

    if index <= tree["threshold"]:
        return predict_regression_tree(tree["left"], index)

    return predict_regression_tree(tree["right"], index)


def train_decision_tree_model(values):
    baseline = train_least_squares_model(values)

    if len(values) < 4:
        return {
            "kind": "least_square",
            "model": baseline,
        }

    samples = [
        (
            index,
            float(value) - least_squares_value_at(baseline, index),
        )
        for index, value in enumerate(values)
    ]

    return {
        "kind": "tree",
        "baseline": baseline,
        "tree": build_regression_tree(samples),
        "value_count": len(values),
    }


def forecast_decision_tree_model(model_data, steps):
    if steps <= 0:
        return []

    if model_data.get("kind") != "tree":
        return forecast_least_squares_model(model_data["model"], steps)

    value_count = model_data["value_count"]

    return [
        round(
            least_squares_value_at(model_data["baseline"], value_count + step)
            + float(
                predict_regression_tree(
                    model_data["tree"],
                    value_count + step,
                )
            ),
            2,
        )
        for step in range(steps)
    ]


def train_random_forest_model(values, tree_count=25):
    baseline = train_least_squares_model(values)

    if len(values) < 4:
        return {
            "kind": "least_square",
            "model": baseline,
        }

    samples = [
        (
            index,
            float(value) - least_squares_value_at(baseline, index),
        )
        for index, value in enumerate(values)
    ]
    min_leaf_size = max(1, min(4, len(samples) // 6))
    random_source = random.Random(len(values) * 1009)
    trees = []

    for tree_index in range(tree_count):
        bootstrap_samples = [
            random_source.choice(samples)
            for _sample in samples
        ]
        trees.append(build_regression_tree(
            bootstrap_samples,
            max_depth=2 + (tree_index % 3),
            min_leaf_size=min_leaf_size,
        ))

    return {
        "kind": "forest",
        "baseline": baseline,
        "trees": trees,
        "value_count": len(values),
    }


def forecast_random_forest_model(model_data, steps):
    if steps <= 0:
        return []

    if model_data.get("kind") != "forest":
        return forecast_least_squares_model(model_data["model"], steps)

    trees = model_data["trees"]
    value_count = model_data["value_count"]

    return [
        round(
            least_squares_value_at(model_data["baseline"], value_count + step)
            +
            sum(
                predict_regression_tree(tree, value_count + step)
                for tree in trees
            ) / len(trees),
            2,
        )
        for step in range(steps)
    ]


def normalize_prediction_algorithm(algorithm):
    normalized = algorithm.strip().lower().replace("_", " ").replace("-", " ")

    aliases = {
        "least square": "least_square",
        "least squares": "least_square",
        "linear least squares": "least_square",
        "linear least square": "least_square",
        "arima": "arima",
        "prophet": "prophet",
        "decision tree": "decision_tree",
        "random forest": "random_forest",
    }

    return aliases.get(normalized, "least_square")


def train_metric_models(values, timestamps):
    return {
        "value_count": len(values),
        "models": {
            "least_square": train_least_squares_model(values),
            "arima": train_arima_model(values),
            "prophet": train_prophet_model(values, timestamps),
            "decision_tree": train_decision_tree_model(values),
            "random_forest": train_random_forest_model(values),
        },
    }


def train_prediction_model(values, timestamps, algorithm):
    if algorithm == "arima":
        return train_arima_model(values)

    if algorithm == "prophet":
        return train_prophet_model(values, timestamps)

    if algorithm == "decision_tree":
        return train_decision_tree_model(values)

    if algorithm == "random_forest":
        return train_random_forest_model(values)

    return train_least_squares_model(values)


def train_metric_model(values, timestamps, algorithm):
    models = {
        algorithm: train_prediction_model(values, timestamps, algorithm),
    }

    if algorithm != "least_square":
        models["least_square"] = train_least_squares_model(values)

    return {
        "value_count": len(values),
        "models": models,
    }


def train_prediction_models_for_series(training_series, algorithm):
    trained_models = {}

    for metric_key, metric_series in training_series.items():
        trained_models[metric_key] = train_metric_model(
            metric_series["values"],
            metric_series["timestamps"],
            algorithm,
        )

    return trained_models


def train_prediction_models():
    global prediction_models, prediction_training_error

    try:
        training_series = load_prediction_training_series()
        trained_models = {}

        for metric_key, metric_series in training_series.items():
            values = metric_series["values"]
            timestamps = metric_series["timestamps"]
            trained_models[metric_key] = train_metric_models(values, timestamps)

        with prediction_model_lock:
            prediction_models = trained_models
            prediction_training_error = None

        counts = {
            metric_key: metric_models["value_count"]
            for metric_key, metric_models in trained_models.items()
        }
        print(
            "Prediction models trained from "
            f"{PREDICTION_DATABASE_NAME}.{PREDICTION_COLLECTION_NAME}: {counts}"
        )
    except Exception as error:
        with prediction_model_lock:
            prediction_models = {}
            prediction_training_error = str(error)

        print(f"Prediction model training failed: {error}")


def forecast_trained_values(metric_models, steps, algorithm):
    if metric_models is None or metric_models.get("value_count", 0) == 0:
        return []

    model_data = metric_models["models"].get(algorithm)

    if model_data is None:
        model_data = metric_models["models"]["least_square"]
        algorithm = "least_square"

    if algorithm == "arima":
        return forecast_arima_model(model_data, steps)

    if algorithm == "prophet":
        return forecast_prophet_model(model_data, steps)

    if algorithm == "decision_tree":
        return forecast_decision_tree_model(model_data, steps)

    if algorithm == "random_forest":
        return forecast_random_forest_model(model_data, steps)

    return forecast_least_squares_model(model_data, steps)


@app.on_event("startup")
def train_prediction_models_on_startup():
    global prediction_training_error

    try:
        training_series, training_metadata = load_recent_prediction_training_series(28)
        default_algorithm = "least_square"
        trained_models = train_prediction_models_for_series(
            training_series,
            default_algorithm,
        )
        cache_key = (
            training_metadata["source"],
            training_metadata["marker"],
            28,
            default_algorithm,
        )

        with prediction_model_lock:
            prediction_cache[cache_key] = trained_models
            prediction_training_error = None

        counts = {
            metric_key: metric_models["value_count"]
            for metric_key, metric_models in trained_models.items()
        }
        print(
            "Prediction cache warmed from "
            f"{training_metadata['source']}: {counts}"
        )
    except Exception as error:
        with prediction_model_lock:
            prediction_training_error = str(error)

        print(f"Prediction cache warmup failed: {error}")


@app.get("/api/predictions")
def get_predictions(steps: int = 10, limit: int = 28, algorithm: str = "least square"):
    steps = max(0, min(steps, 100))
    limit = max(4, min(limit, 500))
    prediction_algorithm = normalize_prediction_algorithm(algorithm)
    training_series, training_metadata = load_recent_prediction_training_series(limit)
    cache_key = (
        training_metadata["source"],
        training_metadata["marker"],
        limit,
        prediction_algorithm,
    )

    with prediction_model_lock:
        cached_models = prediction_cache.get(cache_key)

    if cached_models is None:
        try:
            cached_models = train_prediction_models_for_series(
                training_series,
                prediction_algorithm,
            )
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=f"Prediction models are not available: {error}",
            ) from error

        with prediction_model_lock:
            prediction_cache.clear()
            prediction_cache[cache_key] = cached_models

    predictions = {
        key: forecast_trained_values(
            cached_models.get(key),
            steps,
            prediction_algorithm,
        )
        for key in METRIC_FIELD_NAMES
    }

    return {
        "algorithm": prediction_algorithm,
        "source": training_metadata["source"],
        "training_counts": {
            key: len(metric_series["values"])
            for key, metric_series in training_series.items()
        },
        "predictions": predictions,
    }


class UpdateParams(BaseModel):
    fieldCapacity: float
    sampleFrequency: int
    soilVolumeLiters: float
    targetSoilMoisture: float
    wiltingPoint: float


@app.post("/api/update-params")
def update_params(params: UpdateParams):
    payload = (
        f"fieldCapacity : {params.fieldCapacity}, "
        f"sampleFrequency : {params.sampleFrequency}, "
        f"soilVolumeLiters : {params.soilVolumeLiters} , "
        f"targetSoilMoisture : {params.targetSoilMoisture}, "
        f"wiltingPoint : {params.wiltingPoint}"
    )

    dic_payload =  update_local_params(params.model_dump())

    result = mqtt_client.publish(MQTT_TOPIC2, dic_payload)

    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise HTTPException(
            status_code=500,
            detail=f"MQTT publish failed with code {result.rc}",
        )

    return {"status": "ok", "published_payload": payload}


@app.delete("/api/sensor-data")
def delete_all_sensor_data():
    result = sensor_collection.delete_many({})

    return {"deleted_count": result.deleted_count}


if __name__ == "__main__":
    mqtt_thread = threading.Thread(target=start_mqtt_client)
    mqtt_thread.daemon = True
    mqtt_thread.start()

    uvicorn.run(app,host="0.0.0.0",port=8000)
