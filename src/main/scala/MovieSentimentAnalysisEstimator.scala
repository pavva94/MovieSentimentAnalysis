import java.nio.file.{Files, Paths}

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.sql.SparkSession

class MovieSentimentAnalysisEstimator {

  def estimateReview(review: String, localMode:Boolean): Int = {

    val path = "Documents/Projects/UniBo/LanguagesAndAlgorithmsForArtificialIntelligence/SentimentAnalysis/src/main/"   // FILL WITH PATH
    val model_path = if (localMode) {path + "resources/MLPModel2/"} else {"s3n://sentiment-analysis-data-2020/MLPModel2/"}

    val model: MultilayerPerceptronClassificationModel =
      if (Files.exists(Paths.get( model_path))) { // THEN
        val model = MultilayerPerceptronClassificationModel.load(model_path)
        print("Model loaded.\n")
        model
      }
      else {
        print("Model not found, ERROR!\n")
        return -1
      }

    println("Transform data..") // TODO Check existing of files
    val transformPipeline = PipelineModel.load(
      if (localMode) {path + "resources/TransformDataModel/"}
      else {"s3n://sentiment-analysis-data-2020/TransformDataModel/"}
    )

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
