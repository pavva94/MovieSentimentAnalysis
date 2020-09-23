import java.nio.file.{Files, Paths}

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.sql.SparkSession

class MovieSentimentAnalysisEstimator {

  def estimateReview(review: String, localMode:Boolean): Int = {

    val path = "Documents/Projects/UniBo/LanguagesAndAlgorithmsForArtificialIntelligence/SentimentAnalysis/src/main/"   // FILL WITH PATH

    val model_path =
      if (localMode) {path + "resources/MLPModel2/"}
      else {"s3n://sentiment-analysis-data-2020/Models/MLPModel2/"}

    val model: MultilayerPerceptronClassificationModel =
      if (Files.exists(Paths.get(model_path))) { // THEN
        val model = MultilayerPerceptronClassificationModel.load(model_path)
        print("Model loaded.\n")
        model
      }
      else { // ELSE
        print("Model not found, new training!\n")
        val model = new MovieSentimentAnalysisTrainer().createEstimator(localMode)
        model
      }

    val transform_path =
      if (localMode) {path + "resources/TransformDataModel/"}
      else {"s3n://sentiment-analysis-data-2020/Models/TransformDataModel/"}

    println("Transform data..")
    val transformPipeline = if (Files.exists(Paths.get(transform_path))) { // THEN
        val pipeline = PipelineModel.load(transform_path)
        print("Transform Model loaded.\n")
        pipeline
      } else { // ELSE
        print("Transform Model not found, new training!\n")
        val pipeline = new TransformData().createTransformPipeline(localMode = localMode)
        pipeline
      }

    val spark = if (localMode) {
      println("Local Mode selected")
      // session for local distributed cluster
      SparkSession.builder
        .appName("Sentiment Analysis Classifier")
        .master("spark://MBPdiAlessandro.homenet.telecomitalia.it:7077")
        .getOrCreate()
    } else {
      println("AWS Mode selected")
      // session for AWS
      SparkSession.builder
        .appName("Sentiment Analysis Classifier")
        .getOrCreate()
    }
    import spark.implicits._

    val testDF = List(review).toDF("review")

    val finalData = transformPipeline.transform(testDF)
    val dataset = finalData.select("features")
    val result = model.transform(dataset)
    val predictedReview: Int = result.select("PredictedLabel").first().getDouble(0).toInt
    if (predictedReview == 0)
      println("Your review has a NEGATIVE sentiment")
    else
      println("Your review has a POSITIVE sentiment")

    spark.stop()
    predictedReview
  }
}
