import org.apache.log4j.{Level, Logger}


object SentimentAnalysis {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    // check args: Bad use of scala but i had to do in this way because args.toList on my cluster doesn't worked
    val loadModel: Boolean = {
      try {
        if (args(0) == "--loadModel")
          args(1).toBoolean
        else
          false // default is training
      } catch {
        case ex: ArrayIndexOutOfBoundsException =>
          println("Arguments missing")
          false
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

    println("Local Mode: " + localMode)
    println("Load Model: " + loadModel)

    val estimator = new MovieSentimentAnalysisEstimator()

    print(estimator.estimateReview("It seems like i loved it", localMode=true))

  }
}




