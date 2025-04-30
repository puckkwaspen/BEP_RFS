"""
Preprocessing Pipeline for DAISY Clinical Data
==============================================

This script handles the preprocessing, merging, and imputation of lab results, vital signs,
and demographic data from the DAISY study.

Key Functions:
--------------
- `lab(df)`: Cleans and reshapes lab test data (e.g., ALT, AST, Glucose).
- `vitals(df)`: Processes vitals and calculates BMI.
- `age(df)`: Extracts demographics (AGE, SEX) for Anorexia Nervosa patients.
- `merge_lab_age(df_lab, df_age)`: Merges lab data with demographics.
- `merge_final(df_lab_age, df_vitals)`: Combines all data into one dataset.
- `mice_imputation(df)`: Imputes missing values using MICE with PMM.
- `final_imputations_and_export(df)`: Final simple imputation and BMI recalculation.
- `add_time_features(df)`: Creates delta and percent change features.

Outputs:
--------
- `BEP_imputed.csv`: Cleaned and imputed dataset.
- `BEP_imputed_delta.csv`: Dataset with delta features.
- `BEP_imputed_percentage_change.csv`: Dataset with percent-change features.
"""
import pandas as pd
import numpy as np
import miceforest as mf
from sklearn.impute import SimpleImputer

#### specify the output paths of the created dataframes to be saved to CSVs here ####
output_imputed = "BEP_imputed.csv"
output_imputed_time_feat = "BEP_imputed_time_feat.csv"
output_delta = "BEP_imputed_delta.csv"
output_percentage_change = "BEP_imputed_percentage_change.csv"


def lab(df):
    """
    Preprocesses laboratory data from the DAISY dataset.

    This function filters relevant lab measurements (e.g., ALT, AST, Glucose),
    converts and formats date information, reshapes the data into a wide format,
    renames columns to standardized names, and filters out patients with fewer
    than 3 lab records. It also handles special characters in test results
    (like "<8") and ensures numerical columns are correctly typed.

    :param df: pandas DataFrame containing raw lab data with columns such as
               'O_ITEM', 'UITSLAG_WAARDE', 'p_DATE_BEPALING', 'pid', 'intid', etc.
    :return: pandas DataFrame in wide format with cleaned and converted lab values,
             indexed by ROW and including only patients with ≥3 measurements.
    """

    # Filter relevant lab items
    df = df[df['O_ITEM'].isin([
        'Kalium', 'Leucocyten', 'ALAT (GPT)', 'ASAT (GOT)',
        'Fosfaat anorganisch', 'Magnesium', 'Glucose (n.n.)'
    ])].copy()

    # Convert date and extract UNIX timestamp
    df['p_DATE_BEPALING'] = pd.to_datetime(df['p_DATE_BEPALING'])
    df['DATE'] = df['p_DATE_BEPALING'].dt.normalize()
    df['DATE'] = df['DATE'].astype('int64') // 10 ** 9

    # Drop unneeded columns
    df.drop(columns=['STATUS_AANVRAAG', 'O_STATUS_UITSLAG', 'p_DATE_BEPALING'], inplace=True)

    # Rename columns
    df.rename(columns={
        'pid': 'PATIENT_ID',
        'intid': 'INTAKE_ID',
        'O_ITEM': 'CHEMICAL_VALUE',
        'UITSLAG_WAARDE': 'VALUE_RESULT',
        'NORMAALWAARDE': 'NORMAL_RANGE',
        'seq_num-lab': 'SEQUENCE'
    }, inplace=True)

    # Sort and pivot
    df.sort_values(by='PATIENT_ID', inplace=True)
    df = df.pivot_table(
        index=['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE'],
        columns='CHEMICAL_VALUE',
        values='VALUE_RESULT',
        aggfunc='first'
    ).rename_axis(None, axis=1).reset_index()

    # Add row index
    df['ROW'] = range(1, len(df) + 1)
    df.set_index('ROW', inplace=True)

    # Keep only patients with 3 or more measurements
    patient_counts = df['PATIENT_ID'].value_counts()
    df = df[df['PATIENT_ID'].isin(patient_counts[patient_counts >= 3].index)]

    # Rename columns to clean variable names
    df.rename(columns={
        'ASAT (GOT)': 'AST',
        'ALAT (GPT)': 'ALT',
        'Fosfaat anorganisch': 'Phosphate',
        'Kalium': 'Potassium',
        'Leucocyten': 'Leucocytes',
        'Glucose (n.n.)': 'Glucose'
    }, inplace=True)

    # Clean and convert columns
    for col in ['ALT', 'AST']:
        df[col] = df[col].apply(clean_column)

    to_convert = ['Magnesium', 'ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium', 'Leucocytes']
    df[to_convert] = df[to_convert].apply(pd.to_numeric, errors='coerce')

    return df

def clean_column(value):
    """
    Cleans individual lab result values by handling special cases.

    This function is designed to clean lab result values that may include:
    - Values like "<8", which are converted to one less than the number (e.g., "<8" → 7)
    - Purely alphabetical values (e.g., "NEG", "POS"), which are converted to NaN
    - Any numeric values are returned as-is

    :param value: A single lab result value (typically string or numeric)
    :return: A cleaned numeric value or NaN if the original was non-numeric text
    """
    if isinstance(value, str):
        if value.startswith("<"):
            return int(value[1:]) - 1  # Convert "<8" to 7
        elif value.isalpha():  # Check if the value is only letters
            return np.nan  # Replace letters with NaN
    return value  # Keep numeric values as they are


def vitals(df):
    """
    Processes and reshapes vital signs data from the DAISY dataset.

    This function handles transformation of vital signs data (e.g., blood pressure, BMI, temperature)
    by converting timestamps, melting long-format measurements into wide format, cleaning and renaming
    columns, computing BMI from height and weight, and aggregating measurements per patient-intake-date.

    :param df: pandas DataFrame containing raw vital signs data, with columns such as:
               'pid', 'intid', 'O_METING', 'WAARDE1', 'WAARDE2', 'p_DT_METING', etc.
    :return: pandas DataFrame with one row per patient per intake per date, containing vital signs like
             'Weight (kg)', 'Height (m)', 'BMI', 'Systolic', 'Diastolic', 'Temperature (C)', and sequence numbers.
    """
    # Convert to datetime and extract date/time
    df['p_DT_METING'] = pd.to_datetime(df['p_DT_METING'])
    df['DATE'] = df['p_DT_METING'].dt.normalize()
    df['TIME'] = df['p_DT_METING'].dt.time
    df['DATE'] = df['DATE'].astype('int64') // 10**9  # Unix timestamp

    # Drop unnecessary columns
    df.drop(columns=['Split', 'p_DT_METING', 'TIME'], inplace=True)

    # Rename columns
    df.rename(columns={
        'pid': 'PATIENT_ID',
        'intid': 'INTAKE_ID',
        'O_METING': 'MEASUREMENT ITEM',
        'WAARDE1': 'VALUE 1',
        'WAARDE2': 'VALUE 2',
        'seq_num-vitals': 'SEQUENCE'
    }, inplace=True)

    # Filter relevant measurement items
    df = df[df['MEASUREMENT ITEM'].isin([
        'Body Mass Index', 'Tensie / Pols', 'Temperatuur (c)'
    ])].copy()

    # Sort and reset index
    df.sort_values(by=['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Melt value columns
    df_melted = df.melt(
        id_vars=['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE', 'MEASUREMENT ITEM'],
        value_vars=['VALUE 1', 'VALUE 2'],
        var_name='VALUE_TYPE',
        value_name='VALUE'
    )

    # Make measurement item labels unique (e.g., "Tensie / Pols 1", "Tensie / Pols 2")
    df_melted['MEASUREMENT ITEM'] = df_melted['MEASUREMENT ITEM'] + " " + df_melted.groupby(
        ['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE', 'MEASUREMENT ITEM']
    ).cumcount().add(1).astype(str)

    # Pivot to wide format
    df_pivot = df_melted.pivot_table(
        index=['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE'],
        columns='MEASUREMENT ITEM',
        values='VALUE',
        aggfunc='first'
    ).reset_index()

    # Calculate Height and BMI
    df_pivot['Height (m)'] = df_pivot['Body Mass Index 1'] / 100
    df_pivot['BMI'] = df_pivot['Body Mass Index 2'] / (df_pivot['Height (m)'] ** 2)

    # Drop Body Mass Index 1 (height in cm)
    df_pivot.drop(columns=['Body Mass Index 1'], inplace=True)

    # Rename measurement columns
    df_pivot.rename(columns={
        'Tensie / Pols 1': 'Systolic',
        'Tensie / Pols 2': 'Diastolic',
        'Temperatuur (c) 1': 'Temperature (C)',
        'Body Mass Index 2': 'Weight (kg)'
    }, inplace=True)

    # Reorder columns
    df_pivot = df_pivot[[
        'PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE',
        'Weight (kg)', 'Height (m)', 'BMI',
        'Systolic', 'Diastolic', 'Temperature (C)'
    ]]

    # Recalculate SEQUENCE
    df_pivot.sort_values(by=['PATIENT_ID', 'INTAKE_ID', 'DATE'], inplace=True)
    df_pivot['SEQUENCE'] = df_pivot.groupby(['PATIENT_ID', 'INTAKE_ID']).cumcount() + 1

    # Aggregate: keep only first values per patient-intake-date
    df_final = df_pivot.groupby(['PATIENT_ID', 'INTAKE_ID', 'DATE'], as_index=False).agg({
        'Weight (kg)': 'first',
        'Height (m)': 'first',
        'BMI': 'first',
        'Temperature (C)': 'first',
        'Systolic': 'first',
        'Diastolic': 'first'
    })

    # Add SEQUENCE again after aggregation
    df_final['SEQUENCE'] = df_final.groupby(['PATIENT_ID', 'INTAKE_ID']).cumcount() + 1
    df_final = df_final.sort_values(by=['PATIENT_ID', 'INTAKE_ID', 'DATE'])

    return df_final


def age(df):
    """
    Processes patient demographic data, filtering for relevant cases and formatting key fields.

    This function filters the input dataset to include only patients diagnosed with Anorexia Nervosa,
    converts date fields to UNIX timestamps, standardizes column names, maps gender to binary values,
    and selects key demographic variables (AGE and SEX) along with identifiers.

    :param df: pandas DataFrame containing demographic data, including fields like
               'pid', 'intid', 'EDtype', 'p_startdate', 'Main-Age', and 'Main-Bsex'.
    :return: pandas DataFrame with columns ['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE',
             'DATE', 'SEX', 'AGE'] for patients with Anorexia Nervosa only.
    """
    # Convert date column and extract UNIX timestamp
    df['p_startdate'] = pd.to_datetime(df['p_startdate'])
    df['DATE'] = df['p_startdate'].dt.normalize()
    df['DATE'] = df['DATE'].astype('int64') // 10**9  # UNIX timestamp

    # Filter only patients with Anorexia nervosa
    df = df[df['EDtype'] == 'Anorexia nervosa'].copy()

    # Convert INTAKE_ID to int
    df['intid'] = df['intid'].astype(int)

    # Select relevant columns
    df = df[[
        'intid', 'seq_num-edeq', 'pid', 'DATE', 'Main-Age', 'Main-Bsex'
    ]]

    # Rename columns
    df.rename(columns={
        'pid': 'PATIENT_ID',
        'intid': 'INTAKE_ID',
        'seq_num-edeq': 'SEQUENCE',
        'Main-Age': 'AGE',
        'Main-Bsex': 'SEX'
    }, inplace=True)

    # Map gender to binary
    df['SEX'] = df['SEX'].map({'Vrouw': 1, 'Man': 0})

    # Reorder columns
    df = df[['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE', 'SEX', 'AGE']]

    return df

def merge_lab_age(df_lab, df_age):
    """
    Merges preprocessed lab test data with patient demographic data (age and sex).

    This function:
    - Filters lab records to include only patients also present in the demographics dataset.
    - Merges lab and age data on 'PATIENT_ID' and 'INTAKE_ID'.
    - Resolves overlapping column names (e.g., DATE, SEQUENCE).
    - Aggregates lab and demographic values by patient and date, keeping the first
      available value for each measurement.

    :param df_lab: pandas DataFrame with cleaned and reshaped lab test data, including fields like
                   'PATIENT_ID', 'INTAKE_ID', 'ALT', 'AST', 'Phosphate', etc.
    :param df_age: pandas DataFrame with demographic info, including 'AGE' and 'SEX'.
    :return: pandas DataFrame merged on patient-date level, containing lab results and demographics.
    """
    # Ensure only patients in both datasets are kept
    df_lab = df_lab[df_lab['PATIENT_ID'].isin(df_age['PATIENT_ID'])].copy()

    # Merge lab and age data
    df_merged = df_lab.merge(df_age, on=['PATIENT_ID', 'INTAKE_ID'], how='left')

    # Rename overlapping columns to keep only relevant ones
    df_merged.rename(columns={
        'SEQUENCE_x': 'SEQUENCE',
        'DATE_x': 'DATE'
    }, inplace=True)

    # Drop unnecessary duplicates
    df_merged.drop(columns=['DATE_y', 'SEQUENCE_y'], inplace=True)

    # Reorder columns
    df_merged = df_merged[[
        'PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE', 'AGE', 'SEX',
        'ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium', 'Leucocytes', 'Magnesium'
    ]]

    # Aggregate by patient and date, taking the first valid entry per group
    df_final = df_merged.groupby(['PATIENT_ID', 'DATE'], as_index=False).agg({
        'INTAKE_ID': 'first',
        'SEQUENCE': 'first',
        'AGE': 'first',
        'SEX': 'first',
        'ALT': 'first',
        'AST': 'first',
        'Phosphate': 'first',
        'Glucose': 'first',
        'Potassium': 'first',
        'Leucocytes': 'first',
        'Magnesium': 'first'
    })

    return df_final


def merge_final(df_lab_age, df_vitals):
    """
        Merges lab + demographic data with vitals measurements into a unified clinical dataset.

        This function:
        - Joins lab/age and vitals datasets using ['DATE', 'PATIENT_ID', 'INTAKE_ID'].
        - Resolves column name conflicts (e.g., SEQUENCE) from merging.
        - Reorders columns to a consistent and interpretable layout.
        - Converts lab results to numeric types.
        - Fills missing height values using forward/backward fill per patient.
        - Cleans known outlier values in temperature and replaces zeroes in blood pressure with NaN.

        :param df_lab_age: pandas DataFrame containing merged lab results and demographic information.
        :param df_vitals: pandas DataFrame with vital sign measurements (e.g., blood pressure, temperature).
        :return: pandas DataFrame combining lab, demographic, and vital data for each patient-intake-date.
    """
    # Merge lab+age data with vitals on DATE, PATIENT_ID, INTAKE_ID
    df_combined = pd.merge(
        df_lab_age, df_vitals,
        on=['DATE', 'PATIENT_ID', 'INTAKE_ID'],
        how='left',
        suffixes=('_final', '_merged')
    )

    # Rename and drop duplicate SEQUENCE columns
    df_combined.rename(columns={'SEQUENCE_final': 'SEQUENCE'}, inplace=True)
    df_combined.drop(columns=['SEQUENCE_merged'], inplace=True)

    # Reorder columns
    new_column_order = [
        'PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE',
        'AGE', 'SEX', 'Weight (kg)', 'Height (m)', 'BMI',
        'Temperature (C)', 'Systolic', 'Diastolic',
        'ALT', 'AST', 'Phosphate', 'Glucose',
        'Potassium', 'Leucocytes', 'Magnesium'
    ]
    df_combined = df_combined[new_column_order]

    # Convert lab result columns to numeric
    columns_to_convert = [
        'ALT', 'AST', 'Phosphate', 'Glucose',
        'Potassium', 'Leucocytes', 'Magnesium'
    ]
    df_combined[columns_to_convert] = df_combined[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Fill in missing heights per patient
    df_combined['Height (m)'] = df_combined.groupby('PATIENT_ID')['Height (m)'].transform(lambda x: x.ffill().bfill())

    # Clean specific outlier or invalid values
    df_combined['Temperature (C)'] = df_combined['Temperature (C)'].replace([43, 33.7], float('nan'))
    df_combined['Systolic'] = df_combined['Systolic'].replace(0, float('nan'))
    df_combined['Diastolic'] = df_combined['Diastolic'].replace(0, float('nan'))

    return df_combined


def mice_imputation(df):
    """
        Performs multivariate imputation on clinical features using MICE with Predictive Mean Matching.

        This function:
        - Selects key clinical variables with missing values (e.g., lab results, vitals).
        - Applies MICE (Multiple Imputation by Chained Equations) via the `miceforest` package.
        - Uses Predictive Mean Matching (PMM) with 3 imputed datasets, then averages the results.
        - Replaces the original columns with the averaged imputed values.
        - Recalculates missing BMI values using imputed weight and height.

        :param df: pandas DataFrame containing clinical features, including missing values.
        :return: pandas DataFrame with the same structure but with imputed values for key features.
    """
    # Columns to impute
    cols_to_impute = [
        'Weight (kg)', 'Temperature (C)', 'Systolic',
        'Diastolic', 'ALT', 'AST', 'Phosphate',
        'Glucose', 'Potassium', 'Leucocytes', 'Magnesium'
    ]

    # Subset the dataframe
    df_subset = df[cols_to_impute].copy()

    # Initialize the MICE imputation kernel
    kernel = mf.ImputationKernel(
        df_subset,
        num_datasets=3,
        random_state=123
    )

    # Perform imputation with Predictive Mean Matching
    kernel.mice(
        iterations=20,
        mean_match_candidates=5
    )

    # Average the imputed datasets
    imputed_datasets = [kernel.complete_data(dataset=i) for i in range(3)]
    imputed_avg = sum(imputed_datasets) / len(imputed_datasets)

    # Merge averaged imputed values back into the original dataframe
    df[cols_to_impute] = imputed_avg

    # Recalculate BMI where missing
    df['BMI'] = df['BMI'].fillna(df['Weight (kg)'] / (df['Height (m)'] ** 2))

    return df

def final_imputations_and_export(df, output_path='BEP_imputed.csv'):
    """
        Applies final simple imputations to demographic features and recalculates BMI.

        This function:
        - Performs mean imputation for numeric fields (AGE and Height).
        - Performs mode imputation for categorical field (SEX).
        - Recalculates BMI using imputed weight and height values.
        - Returns the updated DataFrame. (Saving to CSV is optional and not included in this version.)

        :param df: pandas DataFrame containing clinical and demographic data with potential missing values.
        :param output_path: str, optional path to save the imputed DataFrame as a CSV file (not used here directly).
        :return: pandas DataFrame with completed AGE, HEIGHT, and SEX values, and updated BMI.
    """
    # Mean imputation for numerical columns
    num_imputer = SimpleImputer(strategy='mean')
    df[['AGE', 'Height (m)']] = num_imputer.fit_transform(df[['AGE', 'Height (m)']])

    # Mode imputation for categorical column
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df[['SEX']] = mode_imputer.fit_transform(df[['SEX']])

    # Recalculate BMI
    df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)

    return df

def add_time_features(df):
    """
    Generates time-based features by calculating the absolute and percentage change
    in key clinical values over time for each patient.

    This function:
    - Computes the absolute difference (delta) between consecutive time points for each patient.
    - Computes the percentage change relative to the previous value.
    - Returns two separate DataFrames:
        1. One with delta features (e.g., ALT_delta)
        2. One with percent change features (e.g., ALT_percent_change)
    - Drops the original clinical measurement columns from both outputs.

    :param df: pandas DataFrame containing time-series clinical data, including
               lab values and vitals per patient over time.
    :return: Tuple of two pandas DataFrames:
             - df_delta: with only the delta (absolute change) features
             - df_pct: with only the percent change features
    """
    # Time-based feature columns
    columns_time_feat = [
        'ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium',
        'Magnesium', 'Weight (kg)', 'BMI', 'Temperature (C)', 'Systolic', 'Diastolic', 'Leucocytes'
    ]

    # Create copies of the DataFrame for each output
    df_delta = df.copy()
    df_pct = df.copy()

    for col in columns_time_feat:
        # Delta (absolute change)
        df_delta[f'{col}_delta'] = df_delta.groupby('PATIENT_ID')[col].diff()

        # Percent change
        df_pct[f'{col}_percent_change'] = df_pct.groupby('PATIENT_ID')[col].pct_change() * 100

    # Drop the original columns from both
    df_delta.drop(columns=columns_time_feat, inplace=True)
    df_pct.drop(columns=columns_time_feat, inplace=True)

    return df_delta, df_pct


print("🔄 Loading and preprocessing lab data...")
lab_clean = lab(pd.read_csv("../../annonymizedDatasets/maskedDAIsy_LabCombinedNew.csv", sep="\t"))
print("✅ Lab data preprocessed.\n")

print("🔄 Loading and preprocessing vitals data...")
vitals_clean = vitals(pd.read_csv("../../annonymizedDatasets/maskedDAIsy_Vitals.csv", sep="\t"))
print("✅ Vitals data preprocessed.\n")

print("🔄 Loading and preprocessing demographics data...")
age_clean = age(pd.read_csv("../../annonymizedDatasets/maskedDAIsy_AllDatasetsCombinedWoRepIntakes_v1.tsv", sep="\t"))
print("✅ Demographics data preprocessed.\n")

print("🔗 Merging lab data with demographics data...")
df_merge_lab_age = merge_lab_age(lab_clean, age_clean)
print("✅ Lab + demographics data merged.\n")

print("🔗 Merging with vitals data...")
df_merged = merge_final(df_merge_lab_age, vitals_clean)
print("✅ Final merged dataset ready.\n")

print("📊 Here's a quick look at the merged dataset (first 10 rows):\n")
print(df_merged.head(10))
print()

print("⚙️ Starting MICE imputation on selected clinical features...")
df_imputed = mice_imputation(df_merged)
print("✅ MICE imputation complete.\n")

print("🛠️ Performing final simple imputations for AGE, HEIGHT, and SEX...")
print("🛠️ Saving the imputed dataset to a CSV file.")
df_final = final_imputations_and_export(df_imputed)
df_final.to_csv(output_imputed, index=False)
print(f"✅ Final dataset saved to: {output_imputed}\n")

print("⏱️ Generating time-based features (delta and percent change)...")
df_deltas, df_pct_changes = add_time_features(df_final)
df_deltas.to_csv(output_delta, index=False)
df_pct_changes.to_csv(output_percentage_change, index=False)
print("✅ Time-based delta and percent change features created.\n")
print(f"✅ Delta dataset saved to: {output_delta}\n")
print(f"✅ Percentage change dataset saved to: {output_percentage_change}\n")