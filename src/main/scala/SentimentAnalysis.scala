import java.lang.ArrayIndexOutOfBoundsException
import java.nio.file.{Files, Paths}

import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession


object SentimentAnalysis {
  def main(args: Array[String]) {
//    Logger.getLogger("org").setLevel(Level.ERROR)

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

    println(!localMode)
    println(loadModel)

    val path = ""   // FILL WITH PATH

    val spark = if (localMode) {
      println("Local Mode selected")
      // session for local distributed cluster
      SparkSession.builder
        .appName("Sentiment Analysis Classifier")
        .master("spark://MBPdiAlessandro.homenet.telecomitalia.it:7077")
        .getOrCreate()
        //      .config("spark.executor.memory", "8gb")
        //      .config("spark.driver.memory", "8gb")
        //      .config("spark.memory.offHeap.enabled", "true")
        //      .config("spark.memory.offHeap.size","16gb ")
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

    println("Loading data..")
    val data = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .format("csv")
      .load(
        if (localMode) {path + "/resources/MovieReviewDataset.csv"}
        else {"s3://sentiment-analysis-data-2020/MovieReviewDataset.csv"}
      )
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
      .setVocabSize(5000)
      .setMinDF(5.0)
      .setMinTF(1.0)

    val transformPipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        stopWordsRemover,
        countVectorizer))

    println("Transform data..")
    val finalData = transformPipeline.fit(data).transform(data)

    val dataset = finalData.select("features", "sentiment")
    dataset.show()

    // Split the data into train and test
    val splits = dataset.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // check if there is an instance of MLP saved
    val model: MultilayerPerceptronClassificationModel =
//      if (Files.exists(Paths.get(path + "resources/MLPModel2/"))) {
      if (Files.exists(
        Paths.get(
          if (localMode) {path + "resources/MLPModel2/"}
          else {"s3n://sentiment-analysis-data-2020/MLPModel2/"}
        )) && loadModel)
      {  // THEN
//        val model = MultilayerPerceptronClassificationModel.load()
        val model = MultilayerPerceptronClassificationModel.load(
          if (localMode) {path + "resources/MLPModel2/"}
          else {"s3n://sentiment-analysis-data-2020/MLPModel2/"}
        )
        print("Model loaded.\n")
        model
      }
      else {
        print("Model not found, training...\n")

        // network architecture
        val numFeatures = train.first().getAs[SparseVector]("features").toArray.length
        val layers = Array[Int](
          numFeatures,
          numFeatures / 2,
          2
        )
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
        // save model for later use
        model.save(
          if (localMode) {path + "resources/MLPModel2/"}
          else {"s3://sentiment-analysis-data-2020/MLPModel2/"}
        )
        model
      }

    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("PredictedLabel", "sentiment")
    predictionAndLabels.show()
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("sentiment")
      .setPredictionCol("PredictedLabel")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

    spark.stop()
  }
}




