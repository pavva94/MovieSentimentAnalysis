import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.sql.SparkSession

class MovieSentimentAnalysisEstimator {

  def estimateReview(review: String, localMode:Boolean): Int = {

    val spark = if (localMode) {
//      println("Local Mode selected")
      // session for local distributed cluster
      SparkSession.builder
        .appName("Sentiment Analysis Classifier")
        .master("spark://MBPdiAlessandro.homenet.telecomitalia.it:7077")
        .getOrCreate()
    } else {
//      println("AWS Mode selected")
      // session for AWS
      SparkSession.builder
        .appName("Sentiment Analysis Classifier")
        .getOrCreate()
    }
    import spark.implicits._

    val path = ""   // FILL WITH PATH

    val transform_path =
      if (localMode) {path + "resources/TransformDataModel/"}
      else {"s3n://sentiment-analysis-data-2020/Models/TransformDataModel/"}

    println("Transform data..")
    val transformPipeline =
      try {
        val pipeline = PipelineModel.load(transform_path)
        print("Transform Model loaded.\n")
        pipeline
      } catch {
        case ex: Exception => {
          print("Transform Model not found, new training!\n")
          val pipeline = new TransformData().createTransformPipeline(localMode = localMode)
          pipeline
        }
      }

    val model_path =
      if (localMode) {path + "resources/MLPModel2/"}
      else {"s3n://sentiment-analysis-data-2020/Models/MLPModel2/"}

    val model: MultilayerPerceptronClassificationModel =
      try {
        val model = MultilayerPerceptronClassificationModel.load(model_path)
        print("MLP Model loaded.\n")
        model
      } catch {
        case ex: Exception =>
          print("MLP Model not found, new training!\n")
          val model = new MovieSentimentAnalysisTrainer().createEstimator(localMode = localMode)
          model
      }

    val testDF = List(review).toDF("review")

    val finalData = transformPipeline.transform(testDF)
    val dataset = finalData.select("features")
    val result = model.transform(dataset)

    val predictedReview: Int = result.select("PredictedLabel").first().getDouble(0).toInt
    if (predictedReview == 0)
      println("Your review has a NEGATIVE sentiment")
    else
      println("Your review has a POSITIVE sentiment")

//    spark.stop()
    predictedReview
  }
}
