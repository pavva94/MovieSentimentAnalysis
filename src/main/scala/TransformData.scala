import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.SparkSession

class TransformData {

  def createTransformPipeline(localMode: Boolean): PipelineModel = {
    val path = ""   // FILL WITH PATH

    val spark = if (localMode) {
//      println("Local Mode selected")
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
//      println("AWS Mode selected")
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
      ).cache()
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
      .setMinDF(5.0)
      .setMinTF(2.0)

    val transformPipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        stopWordsRemover,
        countVectorizer))

    println("Fit pipeline..")
    val tp = transformPipeline.fit(data)

    val model_path =
      if (localMode) {path + "resources/TransformDataModel/"}
      else {"s3n://sentiment-analysis-data-2020/Models/TransformDataModel/"}

    tp.save(model_path)

//    spark.stop()

    tp
  }
}
