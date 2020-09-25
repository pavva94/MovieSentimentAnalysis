import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession


object SentimentAnalysis {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    // check args: Bad use of scala but i had to do in this way because args.toList on my cluster doesn't worked
    val review: String = {
      try {
        if (args(0) == "--review") {
          println(args(1).toString)
          args(1).toString
        } else {
          println("Argument error: review. Using test review...")
          "I loved this film. But there is an error."
        }
      } catch {
        case ex: ArrayIndexOutOfBoundsException => {
          println("Arguments missing: review. Using test review...")
          "I loved this film."
        }
      }
    }

    println("Review to be analysed: \"" + review + "\"")

    val localMode: Boolean = {
      try {
        if (args(2) == "--localMode")
          args(3).toBoolean
        else {
          print("Default mode il Local Mode")
          true // default is local mode
        }
      } catch {
        case ex: ArrayIndexOutOfBoundsException =>
          println("Arguments missing: local mode. Using default true...")
          true
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
        .hadoopConfiguration.set("fs.s3.access.key", "")
      spark.sparkContext
        .hadoopConfiguration.set("fs.s3.secret.key", "")
      spark.sparkContext
        .hadoopConfiguration.set("fs.s3.endpoint", "s3.amazonaws.com")
    }

    println("Local Mode: " + localMode)

    val estimator = new MovieSentimentAnalysisEstimator()

    print(estimator.estimateReview(review = review, localMode = localMode))

  }
}




