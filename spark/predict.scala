import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder().appName("MusicPredictor").getOrCreate()
import spark.implicits._

// Load the trained model
val model = PipelineModel.load("model/genre_model")

// Read input from a temporary file or argument (Flask will pass this)
val inputLyrics = sc.textFile("temp_input.txt").collect().mkString(" ")

val testDf = Seq(inputLyrics).toDF("lyrics")
val predictions = model.transform(testDf)

// Extract probabilities and labels
// Note: We need the labels from the StringIndexer stage to map index back to Genre Name
val labels = model.stages(0).asInstanceOf[org.apache.spark.ml.feature.StringIndexerModel].labels

val result = predictions.select("probability").collect()(0)
val probabilities = result.getAs[org.apache.spark.ml.linalg.Vector](0).toArray

// Format output as JSON-like string for Python to read
val output = labels.zip(probabilities).map { case (label, prob) => s"$label:$prob" }.mkString(",")
println(s"RESULT_START|$output|RESULT_END")

sys.exit(0)