from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# ================= SPARK SESSION =================
spark = (
    SparkSession.builder
    .appName("Live Spark ML Example - Train Once")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ================= FEATURE ASSEMBLER =================
assembler = VectorAssembler(
    inputCols=["temperature"],
    outputCol="features"
)

# ================= TRAIN MODEL (RUNS ONCE) =================
def train_model():
    print("Training model ONLY ONCE")

    train_data = spark.createDataFrame(
        [
            (25.0, 25.1),
            (30.0, 30.1),
            (35.0, 35.0)
        ],
        ["temperature", "label"]
    )

    train_features = assembler.transform(train_data)

    lr = LinearRegression(
        featuresCol="features",
        labelCol="label"
    )

    return lr.fit(train_features)


model = train_model()

# ================= STREAMING DATA =================
schema = "timestamp STRING, temperature DOUBLE"

stream_df = (
    spark.readStream
    .schema(schema)
    .csv("data/stream")
)

feature_df = assembler.transform(stream_df)

# ================= APPLY MODEL =================
predictions = model.transform(feature_df)

query = (
    predictions
    .select(
        col("timestamp"),
        col("temperature"),
        round(col("prediction"), 2).alias("prediction")
    )
    .writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", "false")
    .option("numRows", 100)
    .start()
)

query.awaitTermination()
