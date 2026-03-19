from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, lower, regexp_replace, trim, length

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("MusicGenreClassifier_XGBoost") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 1. Load Datasets
mendeley_df = spark.read.csv("data/Mendeley_dataset.csv", header=True, inferSchema=True, multiLine=True, quote='"', escape='"')
student_df = spark.read.csv("data/Student_dataset.csv", header=True, inferSchema=True, multiLine=True, quote='"', escape='"')

# 2. Merge Datasets
cols = ["artist_name", "track_name", "release_date", "genre", "lyrics"]
merged_df = mendeley_df.select(cols).union(student_df.select(cols)).dropna()

# 3. Clean Dataset
cleaned_df = merged_df.withColumn("genre", lower(trim(col("genre"))))
valid_genres = ["pop", "country", "blues", "jazz", "reggae", "rock", "hip hop", "retro"]
cleaned_df = cleaned_df.filter(
    (col("genre").isin(valid_genres)) & 
    (length(col("genre")) < 15)
)

cleaned_df = cleaned_df.withColumn(
    "lyrics", 
    regexp_replace(col("lyrics"), r"[\n\r\t]", " ")
).withColumn(
    "lyrics", 
    regexp_replace(col("lyrics"), r"[^a-zA-Z\s]", "")
).withColumn(
    "lyrics", 
    lower(trim(col("lyrics")))
).filter(
    (col("lyrics").isNotNull()) & (col("lyrics") != "")
)

cleaned_df.coalesce(1).write.format("csv") \
    .mode("overwrite") \
    .option("header", "true") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .option("quoteAll", "true") \
    .save("data/Merged_dataset")

# 4. Feature Engineering Pipeline
# Convert text labels to numerical indices
indexer = StringIndexer(inputCol="genre", outputCol="label")

# NLP Stages: Tokenize -> Remove Stopwords -> TF-IDF
tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=5000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# 5. XGBoost Model
xgb = SparkXGBClassifier(
    features_col="features",
    label_col="label",
    max_depth=8,
    n_estimators=100,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    use_gpu=False
)

# Build and Train Pipeline
pipeline = Pipeline(stages=[indexer, tokenizer, remover, hashingTF, idf, xgb])
train_data, test_data = cleaned_df.randomSplit([0.8, 0.2], seed=42)

print("Training XGBoost Pipeline...")
model = pipeline.fit(train_data)

# Evaluation
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("-" * 30)
print(f"✅ Training Complete!")
print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="f1"
)
f1_score = f1_evaluator.evaluate(predictions)
print(f"Test F1 Score: {f1_score:.4f}")

# 6. Save the trained PipelineModel
model.write().overwrite().save("models/genre_xgb_model")
print("Model saved successfully to 'models/genre_xgb_model'")

spark.stop()