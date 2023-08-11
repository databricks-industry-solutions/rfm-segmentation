# Databricks notebook source
# MAGIC %md The purpose of this notebook is to demonstrate RFM Segmentation.  This notebook can be found at https://github.com/databricks-industry-solutions/rfm-segmentation.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC Not every customer has the same revenue potential for a given retail organization.  By acknowledging this fact, retailers can better tailer their engagement to ensure the profitability of their relationship with a customer.  But how do we distinguish between higher and lower value customers? And now might we identify specific behaviors to address in order to turn good customers into great ones?
# MAGIC
# MAGIC Today, we often address this concern with an estimation of customer lifetime value (CLV).  But while CLV estimations can be incredibly helpful, we often don't need precise revenue estimates when deciding which customers to engage with which offers.  Instead, a lightweight approach that examines the recency, frequency and (per-interaction) monetary value of a given customer can go a long way to divide customers into groups of higher, lower and in-between value, and this is exactly what the practice of [RFM segmentation](https://link.springer.com/content/pdf/10.1057/palgrave.jdm.3240019.pdf) provides.
# MAGIC
# MAGIC Pre-dating the formal CLV techniques frequently used today, RFM segmentation remains surprisingly popular with marketing teams looking to quickly organize customers into value-aligned groupings.  In this notebook, we want to demonstrate how an RFM segmentation can be performed and operationalized to enable personalized workflows.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install openpyxl==3.1.2

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import mlflow

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md ##Step 1: Access the Data
# MAGIC
# MAGIC The dataset we will use for this exercise is the [Online Retail Data Set](http://archive.ics.uci.edu/ml/datasets/Online+Retail) available from the UCI Machine Learning Repository.  It represents a little over 1-year of customer purchase history through an ecommerce site.  We can download the dataset as follows:

# COMMAND ----------

# DBTITLE 1,Download Dataset
# MAGIC %sh 
# MAGIC
# MAGIC rm -rf /dbfs/tmp/clv/online_retail  # drop any old copies of data
# MAGIC mkdir -p /dbfs/tmp/clv/online_retail # ensure destination folder exists
# MAGIC
# MAGIC # download data to destination folder
# MAGIC wget -N http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx -P /dbfs/tmp/clv/online_retail

# COMMAND ----------

# MAGIC %md The dataset is made available as an Excel spreadsheet which we can read as follows:

# COMMAND ----------

# DBTITLE 1,Access the Dataset
xlsx_filename = "/dbfs/tmp/clv/online_retail/Online Retail.xlsx"

# schema of the excel spreadsheet data range
orders_schema = {
  'InvoiceNo':str,
  'StockCode':str,
  'Description':str,
  'Quantity':np.int64,
  'InvoiceDate':np.datetime64,
  'UnitPrice':np.float64,
  'CustomerID':str,
  'Country':str  
  }

# read spreadsheet to pandas dataframe
# the openpyxl library must be installed for this step to work 
orders_pd = pd.read_excel(
  xlsx_filename, 
  sheet_name='Online Retail',
  header=0, # first row is header
  dtype=orders_schema
  )

# calculate sales amount as quantity * unit price
orders_pd['SalesAmount'] = orders_pd['Quantity'] * orders_pd['UnitPrice']

# display dataset
display(orders_pd)

# COMMAND ----------

# MAGIC %md The dataset is pretty small and reading it from Excel to a pandas dataframe makes a lot of sense given its size and format.  But more typically, sales data will be made available within the lakehouse as a database table that we will read to a Spark dataframe.  In order emulate this experience, we will convert our pandas dataframe to a Spark dataframe now: 

# COMMAND ----------

# DBTITLE 1,Convert to Spark Dataframe
# convert to Spark dataframe
orders = spark.createDataFrame(orders_pd)
orders.cache()

# COMMAND ----------

# MAGIC %md There are numerous fields in this dataset with each record representing a line item in a sales transaction. The fields included are:
# MAGIC
# MAGIC | Field | Description |
# MAGIC |-------------:|-----:|
# MAGIC |InvoiceNo|A 6-digit integral number uniquely assigned to each transaction|
# MAGIC |StockCode|A 5-digit integral number uniquely assigned to each distinct product|
# MAGIC |Description|The product (item) name|
# MAGIC |Quantity|The quantities of each product (item) per transaction|
# MAGIC |InvoiceDate|The invoice date and a time in mm/dd/yy hh:mm format|
# MAGIC |UnitPrice|The per-unit product price in pound sterling (£)|
# MAGIC |CustomerID| A 5-digit integral number uniquely assigned to each customer|
# MAGIC |Country|The name of the country where each customer resides|
# MAGIC |SalesAmount| Derived as Quantity * UnitPrice |
# MAGIC
# MAGIC Of these fields, the ones of particular interest for our work are *InvoiceNo* which identifies the transaction, *InvoiceDate* which identifies the date of that transaction, and *CustomerID* which uniquely identifies the customer across multiple transactions. The *SalesAmount* field is derived from the *Quantity* and *UnitPrice* fields in order to provide us a monetary amount around which we can estimate value.  For RFM segmentation, these values, *i.e.* a reliable customer identifier, a transaction date and a transaction amount, are all we need to perform our work.
# MAGIC
# MAGIC If you scroll through the dataset a bit, you will see several instances where the *SalesAmount* field is negative.  While the documentation associated with this dataset doesn't state this, it appears as if these lines represent returns.  In a larger, more complete dataset, we might reconcile these with the original transactions or line items in order to exclude sales amounts for items later returned.  However, there doesn't appear to be an easy way to consistently do that work here so that we will simply exclude any records with sales amounts less than zero.
# MAGIC
# MAGIC **NOTE** In the real world, you would absolutely want to reconcile returns with prior purchases.  Otherwise, you may overestimate the monetary value associated with a customer.  Our approach here is simply based on a limitation with this dataset.
# MAGIC
# MAGIC Doing this, we might generate per-order summary values as follows.  Please note that we are consolidating transactions for a given customer on purchase date so that if a customer made multiple purchases on a given date, they will be viewed as a single transaction.  This is a common practice in both CLV estimation and RFM segmentation but you should use your judgement about the appropriateness of this techniques to your business when implementing it with your data:

# COMMAND ----------

# DBTITLE 1,Consolidate User-Transactions on Date
orders_consolidated = (
  orders
    .filter('salesamount > 0.0') # remove returns from dataset
    .withColumn('invoicedate', fn.to_date('invoicedate')) # convert date to just datepart
    .groupBy('customerid', 'invoicedate') # group on customer and date
      .agg(fn.sum('salesamount').alias('salesamount')) # sum sales amount
  )

display(orders_consolidated)

# COMMAND ----------

# MAGIC %md ##Step 2: Calculate the RFM Metrics
# MAGIC
# MAGIC To perform RFM segmentation, we need to calculate recency, frequency and monetary value metrics for each customer.  Unlike in formal CLV estimation approaches where these metrics often require slightly more complex logic, in an RFM segmentation these metrics can be calculated simplistically as:<p>
# MAGIC
# MAGIC * **R**ecency - Days since last date in the overall data set that the customer last made a purchase. Higher values indicate worse customer performance.
# MAGIC * **F**requency - Number of unique dates on which a purchase is made by the customer. Higher values indicate better customer performance.
# MAGIC * **M**onetary Value - Average per-date spend by the customer. Higher values indicate better customer performance.
# MAGIC     
# MAGIC With all this in mind, here is how we might calculate RFM metrics as follows:

# COMMAND ----------

# DBTITLE 1,Calculate RFM Metrics
# get last date in dataset
last_date = (
  orders_consolidated
    .groupBy()
      .agg(fn.max('invoicedate').alias('lastdate'))
  )

# calculate metrics
rfm_metrics = (
  orders_consolidated
    .crossJoin(last_date)
    .groupBy('customerid') # for each customer
      .agg(
        fn.min(fn.datediff('lastdate','invoicedate')).alias('recency'), # days since last date in dataset (get lowest value as recency)
        fn.countDistinct('invoicedate').alias('frequency'), # unique dates on which purchases occur
        fn.avg('salesamount').alias('monetary_value') # avg spend per purchase (works because we've already summed sales per customer-date)
        )
  )

# display metrics
display(
  rfm_metrics
)

# COMMAND ----------

# MAGIC %md Before proceeding, it's important that we examine the distribution of each of our RFM metrics:

# COMMAND ----------

# DBTITLE 1,Examine Value Distributions
# send metrics to pandas for visualizations
df_pd = rfm_metrics.toPandas()

# configure plot as three charts in a single row
f, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))

# generate one chart per metric
for i, metric in enumerate(['recency', 'frequency', 'monetary_value']):
   
  # use metric name as chart title
  axes[i].set_title(metric)
  
  # define histogram chart
  axes[i].hist(df_pd[metric], bins=10)

# COMMAND ----------

# MAGIC %md From these visualizations, we can see that outliers are affecting the distributions of our frequency and monetary value metrics.  If we were to dig into the dataset a bit, we'd see that a few high-frequency purchasers and a few (incredibly) high spend customers are creating this.  With more context, we might determine how best to handle these outliers, but without this, we might simply put a cap on values for these metric as follows:

# COMMAND ----------

# DBTITLE 1,Cap the Frequency & Monetary Value Metrics
# calculate metrics
rfm_metrics_cleansed = (
  rfm_metrics
    .withColumn('frequency', fn.expr("case when frequency > 30 then 30 else frequency end"))
    .withColumn('monetary_value', fn.expr("case when monetary_value > 2500 then 2500 else monetary_value end"))
  )

# COMMAND ----------

# DBTITLE 1,Examine Value Distributions
# extract metrics to pandas for visualization
df_pd = rfm_metrics_cleansed.toPandas()

# configure plot as three charts in a single row
f, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))

# generate one chart per metric
for i, metric in enumerate(['recency', 'frequency', 'monetary_value']):
   
  # use metric name as chart title
  axes[i].set_title(metric)
  
  # define chart
  axes[i].hist(df_pd[metric], bins=10)

# COMMAND ----------

# MAGIC %md Having finalized our metrics, we'll extract these data to a pandas dataframe as this will better align with the remaining work we are to perform.  If you have too much data for a pandas dataframe, everything we are doing in the remainder of this notebook can be recreated using capabilities in Spark MLLib.  However, we believe it is easier to perform this work using sklearn and that you will find more examples online implemented using that library.  Having consolidated your data to a few metrics per customer identifier, a pandas dataframe should be more than sufficient for many 10s of millions of customers on a good sized cluster.  Should you experience memory pressures with a pandas dataframe, consider taking a random sample to support model training:

# COMMAND ----------

# DBTITLE 1,Extract Metrics to Pandas Dataframe
inputs_pd = rfm_metrics_cleansed.toPandas()

# COMMAND ----------

# MAGIC %md ##Step 3: Organize Customers into Quantiles
# MAGIC
# MAGIC With our metrics properly cleansed, we can assign our different recency, frequency and monetary values to bins.  In doing this, we need to consider both the number of bins to employ and the number of values to assign to each.
# MAGIC
# MAGIC In RFM segmentation, we typically select either 5 or 10 bins per metric.  Our goal with this exercise is not to optimize our cluster design from a statistical point view but instead to find a workable number of segments that our marketing team can effectively employ.  With 5 bins, we have 125 potential groupings and with 10 bins, we have 1,000 potential groupings. In practice, we will often find a smaller number of meaningful clusters from these possible combinations, but already the smaller of these values is excessive for our needs.  So, we'll start with 5 bins per metric.  If we don't find good clustering results later, we could return to this step and bump the quantiles up to 10.
# MAGIC
# MAGIC To divide our metrics into bins, we can use a number of strategies.  With a *uniform* binning strategy, each bin has a consistent width, much like within the histograms above. With a *quantile* strategy, bins have a variable width so that values are distributed relatively evenly between each bin.  As our goal is to differentiate between users based on relative value, the *quantile* approach seems to be more appropriate:
# MAGIC

# COMMAND ----------

# DBTITLE 1,Define Binning Logic
# defining binning transformation
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

# apply binner to each column
col_trans = ColumnTransformer(
  [
    ('r_bin', binner, ['recency']),
    ('f_bin', binner, ['frequency']),
    ('m_bin', binner, ['monetary_value'])
    ],
  remainder='drop'
  )

# COMMAND ----------

# MAGIC %md One quick note before implementing the binning is that our frequency and monetary value metrics reflect better customer performance as their values increase.  Separately, our recency values indicate lower performance as its values increase.  To help our marketing team make sense of these values, we'll reverse the bin ordinals we calculate on recency so that as the recency bin increases it too reflects better customer performance:

# COMMAND ----------

# DBTITLE 1,Apply Binning Logic
# invert the recency values so that higher is better
inputs_pd['recency'] = inputs_pd['recency'] * -1

# bin the data
bins = col_trans.fit_transform(inputs_pd)

# add bins to input data
inputs_pd['r_bin'] = bins[:,0]
inputs_pd['f_bin'] = bins[:,1]
inputs_pd['m_bin'] = bins[:,2]

# display dataset
display(inputs_pd)

# COMMAND ----------

# DBTITLE 1,Examine Bin Distributions
# configure plot as three charts in a single row
f, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))

for i, metric in enumerate(['r_bin','f_bin','m_bin']):
   
  # use metric name as chart title
  axes[i].set_title(metric)
  
  # define chart
  axes[i].hist(inputs_pd[metric], bins=5)

# COMMAND ----------

# MAGIC %md From the visualizations, we can see that our bins are not perfectly even and, in the case of frequency, we even have a gap in values. This is to be expected with metrics such as RFM that demonstrate high degrees of skew.

# COMMAND ----------

# MAGIC %md ##Step 4: Explore Potential Clustering
# MAGIC
# MAGIC Before jumping into cluster assignments, let's get ourselves oriented to the scored data and visually inspect whether clustering may be appropriate.  One commonly employed way of doing this is through [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://lvdmaaten.github.io/tsne/).
# MAGIC
# MAGIC The goal of the t-SNE algorithm is to describe the structural complexity of our data along 2 or 3 component axis.  It is a dimension reduction technique but one we cannot use an input into other machine learning algorithms.  It simply helps us visualize some of the inherent structure within the data:

# COMMAND ----------

# DBTITLE 1,Assign t-SNE Scores to Data
# train the tsne model and compute x and y axes for our values 
# (adjust the perplexity value higher or lower until you receive a visual result that's meaningful)
tsne = TSNE(n_components=2, perplexity=80, n_iter=1000, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(inputs_pd[['r_bin','f_bin','m_bin']])

# return the axes assignments to our metrics dataset
inputs_pd['tsne_one'] = tsne_results[:,0]
inputs_pd['tsne_two'] = tsne_results[:,1]

# display results
display(inputs_pd)

# COMMAND ----------

# MAGIC %md Now, we can visualize this data, highlighting individual points based on their RFM scores in order to get a clearer since of the structure of our scored customer data. Notice that warmer colors align with higher scores for each metric:

# COMMAND ----------

# DBTITLE 1,Visualize the Values
# configure plot as three charts in a single row
f, axes = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))

for i, metric in enumerate(['r_bin', 'f_bin', 'm_bin']):
  
  # unique values for this metric
  n = inputs_pd[['{0}'.format(metric)]].nunique()[0]
  
  # use metric name as chart title
  axes[i].set_title(metric)
  
  # define chart
  sns.scatterplot(
    x='tsne_one',
    y='tsne_two',
    hue='{0}'.format(metric),
    palette=sns.color_palette('coolwarm', n),
    data=inputs_pd,
    legend=False,
    alpha=0.4,
    ax = axes[i]
    )

# COMMAND ----------

# MAGIC %md From the visualizations, we can see some clear divisions between users along these three metrics.  We can also see how different regions align with one another with regards to the strength of their recency, frequency and monetary value metrics.  We will return to these visualizations after cluster assignment to help us understand what each cluster tells us about our customers.

# COMMAND ----------

# MAGIC %md ##Step 5: Cluster Customers Based on RFM Scores
# MAGIC
# MAGIC From a quick visual inspection of the t-SNE charts, we can see sufficient overlap between RFM metrics that we wouldn't expect 5 x 5 x 5 (125) clusters to emerge from our data.  In fact, 125 clusters wouldn't be useful for our marketing team.  So we can explore defining between 2 and 20 clusters (though from the visuals, it appears 20 clusters won't be needed either).
# MAGIC
# MAGIC **NOTE** The processing times on this step will drop as you add virtual cores across the worker nodes of your cluster up to the number of k values you are exploring.

# COMMAND ----------

# DBTITLE 1,Evaluate Cluster Size
# define max value of k to explore
max_k = 20

# copy binned_pd to each worker node to facilitate parallel evaluation
inputs_pd_broadcast = sc.broadcast(inputs_pd[['r_bin','f_bin','m_bin']])


# function to train and score clusters based on k cluster count
@fn.udf('float')
def get_silhouette(k):

  # train a model on k
  km = KMeans(
    n_clusters=k, 
    init='random',
    n_init=10000
    )
  kmeans = km.fit( inputs_pd_broadcast.value )

  # get silhouette score for model 
  silhouette = silhouette_score( 
      inputs_pd_broadcast.value,  # x values
      kmeans.predict(inputs_pd_broadcast.value) # cluster assignments 
      )
  
  # return score
  return float(silhouette)


# assemble an dataframe containing each k value
iterations = (
  spark
    .range(2, max_k + 1, step=1, numPartitions=sc.defaultParallelism) # get values for k
    .withColumnRenamed('id','k') # rename to k
    .repartition( max_k-1, 'k' ) # ensure data are well distributed
    .withColumn('silhouette', get_silhouette('k'))
  )
  
# release the distributed dataset
inputs_pd_broadcast.unpersist()

# display the results of our analysis
display( 
  iterations
      )

# COMMAND ----------

# MAGIC %md From our chart, it appears 8 clusters might be a good target number of clusters.  Yes, there are higher silhouette scores we could achieve, but it appears from the curve that at 8 clusters, the incremental gains with added clusters begin to decline.
# MAGIC
# MAGIC With that in mind, we can define our model to support 8 clusters and finalize our pipeline.  

# COMMAND ----------

# DBTITLE 1,Assemble & Train Pipeline
# define model
model = KMeans(
  n_clusters=8, 
  init='random',
  n_init=10000
  )

# couple model with transformations
pipe = Pipeline(steps=[
  ('binnerize', col_trans),
  ('cluster', model)
  ])

# train pipeline
fitted_pipe = pipe.fit( inputs_pd )

# assign clusters
inputs_pd['cluster'] = pipe.predict( inputs_pd )

# display cluster assignments
display(inputs_pd)

# COMMAND ----------

# DBTITLE 1,Visualize Cluster Assignments
f, axes = plt.subplots(nrows=1, ncols=4, squeeze=True, figsize=(42, 10))

axes[0].set_title('cluster')
sns.scatterplot(
  x='tsne_one',
  y='tsne_two',
  hue='cluster',
  palette=sns.color_palette('husl', inputs_pd[['cluster']].nunique()[0]),
  data=inputs_pd,
  alpha=0.4,
  ax = axes[0]
  )
axes[0].legend(loc='lower left', ncol=2, fancybox=True)

# chart the RFM scores
for i, metric in enumerate(['r_bin', 'f_bin', 'm_bin']):
  
  # unique values for this metric
  n = inputs_pd[['{0}'.format(metric)]].nunique()[0]
  
  # use metric name as chart title
  axes[i+1].set_title(metric)
  
  # define chart
  sns.scatterplot(
    x='tsne_one',
    y='tsne_two',
    hue='{0}'.format(metric),
    palette=sns.color_palette('coolwarm', n),
    data=inputs_pd,
    legend=False,
    alpha=0.4,
    ax = axes[i+1]
    )

# COMMAND ----------

# MAGIC %md The visualization of clusters relative to the RFM metrics helps us understand how clusters related to the metric values and the range of values associated with each cluster. We could perform more detailed analysis of the clusters to understand the distance between members and between the various clusters but a quick visual inspection is often sufficient for this kind of work.
# MAGIC
# MAGIC In addition, we can extract the centroids of each cluster to more precisely understand how each relates to the RFM metrics. Please note that these centroids have exact positions that are captured in fractional values. But to help simplify the comparison of the clusters, we've rounded these up to the nearest integer value:

# COMMAND ----------

# DBTITLE 1,Identify Cluster Centers
clusters = []

# for each cluster
for c in range(0, pipe[-1].n_clusters):
  # get integer values for metrics assocaited with each centroid
  centroids = np.abs(pipe[-1].cluster_centers_[c].round(0).astype('int')).tolist()
  # captuer cluster and centroid values
  clusters += [ [c] + centroids]

# convert details to dataframe
clusters_pd = pd.DataFrame(clusters, columns=['cluster','r_bin','f_bin','m_bin'])

display(clusters_pd)

# COMMAND ----------

# MAGIC %md ##Step 6: Persist Cluster Assignments
# MAGIC
# MAGIC The previous steps in this notebook are intended to demonstrate how an RFM segmentation might be performed, but how might we operationalize the model for on-going work?  Every time we retrain our model, the centroids attached to a cluster id will vary.  If we wish to re-score customers periodically but keep cluster centroids the same between those runs, we need to persist and re-use our model.  This is made easy using the MLFlow model registry: 

# COMMAND ----------

# DBTITLE 1,Assign Model Name
model_name = 'rfm_segmentation'

# COMMAND ----------

# DBTITLE 1,Persist the Model
with mlflow.start_run(run_name='deployment ready'):

  mlflow.sklearn.log_model(
    fitted_pipe,
    'model',
    registered_model_name=model_name
    )

# COMMAND ----------

# MAGIC %md We can then elevate our model to production status to indicate it is ready for use in an on-going ETL pipeline:

# COMMAND ----------

# DBTITLE 1,Elevate Model to Production Status
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
latest_model_info = client.search_model_versions(f"name='{model_name}'")[0]
model_version = latest_model_info.version
model_status = latest_model_info.current_stage

# move model to production status
if model_status.lower() != 'production':
  client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage='production',
    archive_existing_versions=True
    ) 

# COMMAND ----------

# MAGIC %md With our model persisted and elevated to production status, applying it to data is relatively easy:

# COMMAND ----------

# DBTITLE 1,Apply Model to Customer Data
# define user-defined function for rfm segment assignment
rfm_segment = mlflow.pyfunc.spark_udf(spark, model_uri=f'models:/{model_name}/production')

# apply function to summary customer metrics
display(
  rfm_metrics_cleansed
    .withColumn('cluster', 
        rfm_segment(fn.struct('recency','frequency','monetary_value'))
          )
  )

# COMMAND ----------

# MAGIC %md Of course, to make use of these clusters, we'll want access to descriptive information about what each cluster represents.  Typically, the marketing team will assign friendly labels to each cluster that explain what they represent in easy to understand terms.  For our purposes, we'll just persist the centroid information extracted in the last step.  These data could then be joined with the output of the previous cell to provide friendly labels for each cluster assignment:

# COMMAND ----------

# DBTITLE 1,Persist Descriptive Info for Each Cluster
_ = (
  spark
    .createDataFrame(clusters_pd)
    .withColumn( # more typically, a friendly name would be assigned by marketing
      'label', 
      fn.expr("concat('Cluster ', cluster, ': r=', r_bin, ', f=', f_bin, ', m=', m_bin )")
      )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('rfm_clusters')
  )

display(spark.table('rfm_clusters'))

# COMMAND ----------

# MAGIC %md ##Step 7: Activating Customers
# MAGIC
# MAGIC At this stage, the Data Science/Data Engineering work for RFM segmentation has been done.  The question then arises, *how do we make use of these data?*  
# MAGIC
# MAGIC What we need is a tool that will link our cluster assignments with marketing content and deliver that content via appropriate channels.  Reverse ETL tools such as [Census](https://www.getcensus.com/) provide specific functionality for this.
# MAGIC
# MAGIC The first step in doing this for many organizations will be accessing the Census software service.  This can be done through a free trial that Census provides [here](https://app.getcensus.com/?utm_campaign=Audience%20Hub&utm_source=Databricks%20RFM%20blog).  Then, a connection to Databricks is made using the steps documented [here](https://docs.getcensus.com/sources/databricks).
# MAGIC
# MAGIC Once Census is connected to Databricks, the RFM segment data can be made visible via the Census [Audience Hub](https://www.getcensus.com/?utm_campaign=Audience%20Hub&utm_source=Databricks%20RFM%20blog) functionality.  Within Audience Hub, the marketing team can use a combination of customer demographics (housed in other tables or environments) and RFM segment assignments to identify an audience receptive to a particular message:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/rfm_figure2.png'>
# MAGIC </p>
# MAGIC
# MAGIC With the audience defined, Census then allows the marketing team to connect to downstream activation channels, such as those supported through Braze:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/rfm_figure3.png'>
# MAGIC </p>
# MAGIC
# MAGIC The audience can then be used with a specific unit of content to fire off an email message or perform some other form of content delivery, connecting a particular group of customers with targeted messages and offers that help the marketing team drive an intended outcome:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/rfm_figure4.png'>

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | openpyxl | Python library to read/write Excel 2010 xlsx/xlsm/xltx/xltm files| MIT | https://pypi.org/project/openpyxl/ |
