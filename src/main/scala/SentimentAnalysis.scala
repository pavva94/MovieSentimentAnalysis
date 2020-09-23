import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession


object SentimentAnalysis {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    // check args: Bad use of scala but i had to do in this way because args.toList on my cluster doesn't worked
    val review: String = {
      try {
        if (args(0) == "--review")
          args(1).toString
        else
          "" // default is empty
      } catch {
        case ex: ArrayIndexOutOfBoundsException =>
          println("Arguments missing: review. Using test review...")
          "I loved this film."
      }
    }

    val localMode: Boolean = {
      try {
        if (args(2) == "--localMode")
          args(3).toBoolean
        else
          true // default is local mode
      } catch {
        case ex: ArrayIndexOutOfBoundsException =>
          println("Arguments missing")
          false
      }
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

    if (!localMode) {
      // AWS configuration
      spark.sparkContext
        .hadoopConfiguration.set("fs.s3.access.key", "ASIA3XR2YR5TUZTF7BWQ")
      spark.sparkContext
        .hadoopConfiguration.set("fs.s3.secret.key", "vK1KeTmmr41jY/ulGrg/4wZcLvTU1/kGnwJhmlku")
      spark.sparkContext
        .hadoopConfiguration.set("fs.s3.endpoint", "s3.amazonaws.com")
    }

    println("Local Mode: " + localMode)

    val estimator = new MovieSentimentAnalysisEstimator()

    print(estimator.estimateReview(review = review, localMode = true))

  }
}




