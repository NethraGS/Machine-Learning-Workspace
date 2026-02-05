from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# ================= SPARK SESSION =================
spark = (
    SparkSession.builder
    .appName("Live Spark ML - Pass Fail Prediction")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ================= FEATURE ASSEMBLER =================
assembler = VectorAssembler(
    inputCols=["hours_studied"],
    outputCol="features"
)

# ================= TRAIN MODEL (RUNS ONCE) =================
def train_model():
    print("Training Pass/Fail model ONLY ONCE")

    # hours_studied, label (0 = Fail, 1 = Pass)
    train_data = spark.createDataFrame(
        [
            (1.0, 0),
            (2.0, 0),
            (3.0, 0),
            (4.0, 0),
            (5.0, 1),
            (6.0, 1),
            (7.0, 1),
            (8.0, 1),
            (9.0, 1)
        ],
        ["hours_studied", "label"]
    )

    train_features = assembler.transform(train_data)

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label"
    )

    return lr.fit(train_features)


model = train_model()

# ================= STREAMING DATA =================
schema = "timestamp STRING, hours_studied DOUBLE"

stream_df = (
    spark.readStream
    .schema(schema)
    .csv("data/stream")
)

feature_df = assembler.transform(stream_df)

# ================= APPLY MODEL =================
predictions = model.transform(feature_df)

result = predictions.select(
    col("timestamp"),
    col("hours_studied"),
    col("probability"),
    when(col("prediction") == 1, "PASS")
    .otherwise("FAIL")
    .alias("result")
)

query = (
    result.writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", "false")
    .option("numRows", 100)
    .start()
)

query.awaitTermination()
