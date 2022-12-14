
def create_and_extract(working_dir):
    
    # Save dataset from Kaggle API only if it hasn't been saved yet
    if not exists(f'{working_dir}\\BankChurners.csv'):
        
        !kaggle datasets download -d whenamancodes/credit-card-customers-prediction --quiet
        
        # Extract zipfile
        with zipfile.ZipFile('C:\\Users\\kalle\\My python stuff\\PySpark\\credit-card-customers-prediction.zip', 'r') as zip_ref:
            zip_ref.extractall('C:\\Users\\kalle\\My python stuff\\PySpark')
    
    # Load csv into PySpark
    df = spark.read.csv(f'{working_dir}\\BankChurners.csv', inferSchema=True, header=True)
    
    return df

def bucketer(df):
    
    df.fillna(value=0)
    
    # Bucketize continuous values
    bucketizer = Bucketizer()
    bucketizer = Bucketizer(splitsArray=[
                                  [-float("inf"), 20, 40, 60, float("inf")], 
                                  [-float("inf"), 20, 30, 40, 50, float("inf")],
                                  [-float("inf"), 5000, 10000, 15000, 30000, float("inf")],
                                  [-float("inf"), 500, 1000, 1500, 2000, float("inf")],
                                  [-float("inf"), 5000, 10000, 15000, 20000, 25000, 30000, float("inf")],
                                  [-float("inf"), 1, 1.5, 2.0, float("inf")],
                                  [-float("inf"), 500, 1000, 1500, float("inf")],
                                  [-float("inf"), 20, 25, 30, 35, float("inf")],
                                  [-float("inf"), 0.5, 1, 1.5, 2, 2.5, 3, float("inf")],
                                  [-float("inf"), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, float("inf")]],
                        
                        inputCols=['Customer_Age', 'Months_on_book', 
                                   'Credit_Limit', 'Total_Revolving_Bal',
                                   'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                                   'Total_Trans_Amt', 'Total_Trans_Ct',
                                   'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'], 
                        outputCols=['Customer_Age_Buckets', 'Months_on_book_Buckets', 
                                    'Credit_Limit_Buckets', 'Total_Revolving_Bal_Buckets',
                                    'Avg_Open_To_Buy_Buckets', 'Total_Amt_Chng_Q4_Q1_Buckets',
                                    'Total_Trans_Amt_Buckets', 'Total_Trans_Ct_Buckets', 
                                    'Total_Ct_Chng_Q4_Q1_Buckets', 'Avg_Utilization_Ratio_Buckets'])

    bucketed = bucketizer.setHandleInvalid('keep').transform(df)
    
    return bucketed

def encoder(df):
    
    # Encode categorical values
    to_encode = ['Attrition_Flag', 'Gender', 'Education_Level', 
                 'Marital_Status', 'Income_Category', 'Card_Category']

    encode_to = ['Attrition_Flag_encoded', 'Gender_encoded', 'Education_Level_encoded', 
                 'Marital_Status_encoded', 'Income_Category_encoded', 'Card_Category_encoded']

    indexer = StringIndexer(inputCols=to_encode, outputCols=encode_to)
    indexer_fit = indexer.fit(df)
    df_indexed = indexer_fit.transform(df)
    
    return df_indexed

def clean_up(df):
    
    # Drop all columns that have been replaced by buckets or encoded columns
    drop_cols = ['Customer_Age', 'Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1','Total_Trans_Amt', 
                 'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Attrition_Flag', 'Gender', 'Education_Level',
                 'Marital_Status', 'Income_Category', 'Card_Category',
                 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']

    dropped = df.drop(*drop_cols)
    
    return dropped

def execute():
    
    print('Extracting Data')
    spark_df = create_and_extract('C:\\Users\\kalle\\My python stuff\\PySpark')
    print('\n ===================== \n')

    print('Transforming Data')
    bucketed = bucketer(spark_df)
    encoded = encoder(bucketed)
    cleaned = clean_up(encoded)
    print('\n ===================== \n')

    return f'Process Finished'
