import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder().appName("MusicGenreClassifier").getOrCreate()
import spark.implicits._

// 1. Load Mendeley Dataset
val rawData = spark.read.option("header", "true").option("inferSchema", "true").csv("data/Merged_dataset.csv")

// 2. Pre-processing: Focus on lyrics and genre
val data = rawData.select("lyrics", "genre").filter($"lyrics".isNotNull && $"genre".isNotNull)

// 3. Index the labels (Genre names to numeric IDs)
val labelIndexer = new StringIndexer().setInputCol("genre").setOutputCol("label").fit(data)

// 4. Feature Engineering Pipeline
val tokenizer = new Tokenizer().setInputCol("lyrics").setOutputCol("words")
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10000)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

// 5. Model: Naive Bayes supports multi-class classification
val nb = new NaiveBayes().setLabelCol("label").setFeaturesCol("features")

val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, hashingTF, idf, nb))

// 6. Train/Test Split (80/20)
val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 1234L)

// 7. Train the model
val model = pipeline.fit(trainingData)

// 8. Evaluation
val predictions = model.transform(testData)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)

println(s"-------------------------------------------")
println(s"Training Complete. Model Accuracy: ${accuracy * 100}%")
println(s"-------------------------------------------")

// 9. Save the model (Overwrite if exists)
model.write.overwrite().save("model/genre_model")

sys.exit(0)