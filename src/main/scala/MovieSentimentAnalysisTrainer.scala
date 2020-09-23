import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession

class MovieSentimentAnalysisTrainer {

  def createEstimator(localMode: Boolean): MultilayerPerceptronClassificationModel = {

    val path = "Documents/Projects/UniBo/LanguagesAndAlgorithmsForArtificialIntelligence/SentimentAnalysis/src/main/"   // FILL WITH PATH

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
        .hadoopConfiguration.set("fs.s3.access.key", "ASIA3XR2YR5TUZTF7BWQ")
      spark.sparkContext
        .hadoopConfiguration.set("fs.s3.secret.key", "vK1KeTmmr41jY/ulGrg/4wZcLvTU1/kGnwJhmlku")
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

    val transformPipeline = new TransformData().createTransformPipeline(localMode)

    println("Transform data..")
    val finalData = transformPipeline.transform(data)

    val dataset = finalData.select("features", "sentiment")
    dataset.show()

    // Split the data into train and test
    val splits = dataset.randomSplit(Array(0.9, 0.1), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    val model_path = if (localMode) {"resources/MLPModel2/"} else {"s3n://sentiment-analysis-data-2020/Models/MLPModel2/"}

    print("Training...\n")

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
    model.save(model_path)

    println()
    println()
    println("The model is created and saved in:" + model_path)

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

    model

  }
}
