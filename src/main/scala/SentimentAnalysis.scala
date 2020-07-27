import java.nio.file.{Paths, Files}

import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.log4j._
import org.apache.spark.ml.linalg.SparseVector


object SentimentAnalysis {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val path = "" // FILL WITH PATH


    val spark = SparkSession.builder
      .appName("Sentiment Analysis Classifier")
      .master("local[2]")
      .config("spark.executor.memory", "8gb")
      .config("spark.driver.memory", "8gb")
      .config("spark.memory.offHeap.enabled", "true")
      .config("spark.memory.offHeap.size","16gb ")
      .getOrCreate()

    // Replace Key with your AWS account key (You can find this on IAM
//    spark.sparkContext
//      .hadoopConfiguration.set("fs.s3n.access.key", "AKIARTV2LE4BZ26JARQG")
//    // Replace Key with your AWS secret key (You can find this on IAM
//    spark.sparkContext
//      .hadoopConfiguration.set("fs.s3n.secret.key", "Rk+53PbE9KvVVigCe5F484zT/4OVDoy6cS4ALQLW")
//    spark.sparkContext
//      .hadoopConfiguration.set("fs.s3n.endpoint", "s3.amazonaws.com")

    val data = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .format("csv")
      .load(path + "/resources/MovieReviewDataset.csv")
//      .load("s3n://exam-lang-2020.s3.eu-west-3.amazonaws.com/AirQualityRoadsideFinal.csv")
      .cache()
      .repartition(500)
      .na.drop()
      .toDF()

    data.printSchema()

    val tokenizer = new RegexTokenizer()
      .setInputCol("review")
      .setOutputCol("rawWords")
      .setPattern("\\w+")
      .setGaps(false)

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("rawWords")
      .setOutputCol("words")

    val countVectorizer = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(1000)
      .setMinDF(7.0)
      .setMinTF(1.0)

    val transformPipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        stopWordsRemover,
        countVectorizer))

    val finalData = transformPipeline.fit(data).transform(data)

    val dataset = finalData.select("features", "sentiment")
    dataset.show()

    // Split the data into train and test
    val splits = dataset.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

//    val lr = new LogisticRegression()
//      .setMaxIter(15000)
//      .setLabelCol("sentiment")
//      .setFeaturesCol("features")
//
//    // Fit the model
//    val lrModel = lr.fit(train)
//
//    // Print the coefficients and intercept for logistic regression
//    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
//    val layers = Array[Int](3, 5, 4, 3)
    //network architecture, better to keep tuning it until metrics converge
    val numFeatures = train.first().getAs[SparseVector]("features").toArray.length
    val layers = Array[Int](
      numFeatures,
      numFeatures / 2,
      numFeatures / 4,
      2
    )

    // check if there is an instance of MLP saved
    def getModel(): MultilayerPerceptronClassificationModel =
      if (Files.exists(Paths.get(path + "resources/MLPModel/"))) {
        val model = MultilayerPerceptronClassificationModel.load(path + "resources/MLPModel/")
        print("Model loaded.")
        model
      } else {
        print("Model not found, training...")
        // create the trainer and set its parameters
        val trainer = new MultilayerPerceptronClassifier()
          .setFeaturesCol("features")
          .setLabelCol("sentiment")
          .setPredictionCol("PredictedLabel")
          .setLayers(layers)
          .setBlockSize(56)
          .setSeed(1234L)
          .setMaxIter(100)

        // train the model
        val model = trainer.fit(train)
        // ID 922 has label 3 not 2
        // save model for later use
        model.save(path + "resources/MLPModel/")
        model
      }


    val model = getModel()


    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("PredictedLabel", "sentiment")
    predictionAndLabels.show()
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("sentiment")
      .setPredictionCol("PredictedLabel")
    //evaluator.evaluate(result.select("PredictedLabel"))
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

    spark.stop()
  }
}




