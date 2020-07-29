import org.apache.spark.sql.SparkSessionimport org.apache.spark.sql.functions._val spark = SparkSession.builder.appName("PreProcessData").master("local[*]").getOrCreate()// big problem parsing the csv after the download. I had to change the delimiter in somethingval dataset1 = spark.read.option("header","true").option("escape","\"").option("delimiter",",").format("csv")  .load("").cache()    // FILL WITH PATHval dataset2 = spark.read.option("header","true").option("escape","\"").option("delimiter",",").format("csv")  .load("").cache()    // FILL WITH PATHval dataset3 = spark.read.option("header","true").option("escape","\"").option("delimiter",",").format("csv")  .load("").cache()    // FILL WITH PATH// Dataset 3 and 2  have a boolean value for the label and the text review only, instead dataset 1 contain other columns that aren't usefull for this project// Dataset 1 contains also a "unsup" label for unsupervised data, I'll remove that// Dataset 1 step: remove "type" and "file" columns, remove "unsup" label rows, change "pos" in 1 and "neg" in 0val ds11 = dataset1.filter(dataset1("label") isin("neg", "pos", "unsup"))  // removing errors derived by parsingval ds1 = ds11.filter(ds11("label") isin("pos", "neg"))  // removing "unsup"  .drop("type", "file", "_c0")   // removing unused columns  .withColumn("label", when(col("label").equalTo("neg"), 0)    .when(col("label").equalTo("pos"), 1)    .otherwise(col("label")))  .withColumnRenamed("label", "sentiment")val ds2 = dataset2.filter(dataset2("sentiment") isin("negative", "positive"))  // removing errors derived by parsing  .withColumn("sentiment", when(col("sentiment").equalTo("negative"), 0)    .when(col("sentiment").equalTo("positive"), 1)    .otherwise(col("sentiment")))val ds3 = dataset3.filter(dataset3("sentiment") isin("1", "0"))  // removing errors derived by parsingds1.show()ds1.count()ds2.show()ds2.count()ds3.show()ds3.count()val df = ds1.union(ds2).union(ds3)df.show()df.count()df.coalesce(1).write.format("csv")  .mode("overwrite")  .option("header", "true")  .save("")    // FILL WITH PATHspark.stop()