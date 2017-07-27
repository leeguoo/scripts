import findspark
findspark.init()

import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

fullpath = '/work/baby2014review.csv'

#read data
print("read dataframe")
df = spark.read.csv(fullpath,
                    sep=',',
                    inferSchema=True,
                    header=True)

#select columns
print("select review & ratings")
df = df.select("reviewText","overall")

#drop null
print("drop null")
df = df.na.drop()

df.printSchema()
df.show()

###
from pyspark.ml.feature import RegexTokenizer

print("tokenize review")
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words",pattern="\\W")
df = tokenizer.transform(df).select("words","overall")
df.show()

## COUNT
from pyspark.ml.feature import CountVectorizer

print("countvectorize")
df = CountVectorizer(inputCol="words", 
                     outputCol="countVector", 
                     vocabSize=2000, 
                     minDF=8.0)\
                     .fit(df)\
                     .transform(df)\
                     .select("countVector","overall")

df.show(truncate=False)

### 

from pyspark.ml.feature import IDF
print("tfidf")
df = IDF(inputCol="countVector", outputCol="tfidf").fit(df).transform(df).select("tfidf","overall")
df.show()

###

from pyspark.ml.feature import PCA
print("pca")
df = PCA(k=300, inputCol="tfidf", outputCol="pca").fit(df).transform(df).select("pca","overall")
df.show()
#df.show(truncate=False)

###

from pyspark.ml.regression import RandomForestRegressor


rf = RandomForestRegressor(numTrees=50, 
                           maxDepth=5, 
                           seed=42,
                           labelCol='overall', 
                           featuresCol='pca',
                           predictionCol='prediction')
model = rf.fit(df)
pred = model.transform(df)

pred.show()

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="overall",
                                predictionCol="prediction")
print("r2",evaluator.evaluate(pred,{evaluator.metricName: "r2"}))
print("mse",evaluator.evaluate(pred,{evaluator.metricName: "mse"}))
