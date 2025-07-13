# Databricks notebook source
# MAGIC %md
# MAGIC # Data Transformation Project: Analyzing Fictional Sales Data
# MAGIC
# MAGIC This project demonstrates fundamental data transformation techniques using Apache Spark DataFrames within Databricks. We will simulate raw sales data and refine it through several steps to prepare it for analysis.
# MAGIC
# MAGIC **Goals:**
# MAGIC - Create a Spark DataFrame from raw Python data.
# MAGIC - Perform common data cleaning and transformation operations (filtering, adding calculated columns, concatenating strings, type conversion).
# MAGIC - Showcase the power of PySpark for data manipulation.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, lit, concat, when, round as spark_round # Import round and alias it to avoid conflict with Python's built-in round()

# Define the schema for our raw sales data
sales_schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("customer_name", StringType(), True),
    StructField("item", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("price_per_unit", DoubleType(), True),
    StructField("region", StringType(), True),
    StructField("status", StringType(), True) # e.g., "Pending", "Completed", "Cancelled"
])

# Our raw, messy sales data
raw_sales_data = [
    ("ORD001", "Alice Smith", "Laptop", 1, 1200.50, "North", "Completed"),
    ("ORD002", "Bob Johnson", "Mouse", 2, 25.00, "South", "Pending"),
    ("ORD003", "Charlie Brown", "Keyboard", 1, 75.25, "East", "Completed"),
    ("ORD004", "Alice Smith", "Monitor", 1, 300.00, "North", "Pending"),
    ("ORD005", "Eve Davis", "Laptop Bag", 1, 45.00, "West", "Cancelled"),
    ("ORD006", "Bob Johnson", "Headphones", 3, 50.00, "South", "Completed"),
    ("ORD007", "Grace Lee", "Webcam", 1, 60.00, "East", "Completed"),
    ("ORD008", "Frank White", "Mouse", None, 25.00, "North", "Completed"), # Missing quantity
    ("ORD009", "Charlie Brown", "External SSD", 1, 99.99, "East", "Pending"),
    ("ORD010", "Alice Smith", "Webcam", 1, 60.00, "North", "Completed")
]

# Create the initial DataFrame
df_raw_sales = spark.createDataFrame(raw_sales_data, schema=sales_schema)

print("--- Raw Sales Data ---")
df_raw_sales.show()
df_raw_sales.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Data Cleaning - Handling Missing Values and Filtering Invalid Records
# MAGIC
# MAGIC Before analysis, we need to ensure data quality. In this step, we will:
# MAGIC 1. **Fill missing 'quantity' values:** Replace `None` (null) values in the `quantity` column with `0`.
# MAGIC 2. **Filter out cancelled orders:** Remove records where the `status` is 'Cancelled'. This focuses our analysis on active or completed sales.

# COMMAND ----------

# Fill missing 'quantity' with 0
df_cleaned_quantity = df_raw_sales.na.fill(0, subset=["quantity"])

# Filter out orders with 'Cancelled' status
df_filtered_orders = df_cleaned_quantity.filter(col("status") != "Cancelled")

print("--- Data After Cleaning & Filtering ---")
df_filtered_orders.show()
df_filtered_orders.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Feature Engineering - Calculating Total Price and Customer Full Name
# MAGIC
# MAGIC Now, let's derive new, useful information from our existing data:
# MAGIC 1. **Calculate `total_price`:** This is `quantity * price_per_unit`. We'll also round it to 2 decimal places.
# MAGIC 2. **Create `customer_full_name`:** This combines `customer_name` for consistency, though in this dataset, it's already full. This step is more illustrative for concatenation.

# COMMAND ----------

from pyspark.sql.functions import col, lit, concat, when, round as spark_round, sum # Make sure 'sum' is also imported!
from pyspark.sql.types import DoubleType, IntegerType # Import these for casting

# Calculate total_price = quantity * price_per_unit
# Explicitly cast quantity and price_per_unit to DoubleType before multiplication
# This helps ensure the calculation results in a numeric type
df_with_total_price = df_filtered_orders.withColumn(
    "total_price",
    spark_round(
        (col("quantity").cast(DoubleType()) * col("price_per_unit").cast(DoubleType())),
        2
    )
)

# Create customer_full_name (simple concatenation as customer_name is already full)
df_with_customer_full_name = df_with_total_price.withColumn(
    "customer_full_name",
    concat(col("customer_name"), lit("")) # Lit("") is just to ensure it's treated as a string concatenation
)

print("--- Data After Calculating Total Price & Full Name ---")
df_with_customer_full_name.show()
df_with_customer_full_name.printSchema() # Check the schema here to ensure total_price is DoubleType!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Data Aggregation - Summarizing Sales by Region and Status
# MAGIC
# MAGIC Finally, let's aggregate our data to get insights. We will:
# MAGIC 1. **Calculate total sales per region:** Sum the `total_price` for each `region`.
# MAGIC 2. **Count orders per status and region:** See how many orders are 'Completed' vs 'Pending' in each region.

# COMMAND ----------

# Aggregation 1: Total sales per region
df_sales_by_region = df_with_customer_full_name.groupBy("region") \
                                                 .agg(spark_round(sum("total_price"), 2).alias("total_regional_sales")) \
                                                 .orderBy("region")

print("--- Total Sales by Region ---")
df_sales_by_region.show()

# Aggregation 2: Count orders per status and region
from pyspark.sql.functions import count

df_orders_by_status_region = df_with_customer_full_name.groupBy("region", "status") \
                                                      .agg(count("order_id").alias("number_of_orders")) \
                                                      .orderBy("region", "status")

print("\n--- Number of Orders by Region and Status ---")
df_orders_by_status_region.show()

print("\n--- Project Complete! ---")

# COMMAND ----------

