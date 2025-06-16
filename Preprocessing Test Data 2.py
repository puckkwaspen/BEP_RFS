import pandas as pd
import numpy as np
import miceforest as mf
from sklearn.impute import SimpleImputer
import random
import os
from datetime import timedelta

random.seed(42)

#### specify the output paths of the created dataframes to be saved to CSVs here ####
output_imputed_test = "Data/BEP_imputed_TEST2.csv"
output_imputed_time_feat_test = "Data/BEP_imputed_time_feat_TEST2.csv"
output_delta_test = "Data/BEP_imputed_delta_TEST2.csv"
output_percentage_change_test = "Data/BEP_imputed_percentage_change_TEST2.csv"
# IDs for patients who are confirmed not to have had RFS (control group)
ids = [
    1234462616, 38369030, 1349893812, 1473253698, 555297858, 1816358694,
    1299788868, 1297135868, 1725195538, 802367291, 665623621, 436817496,
    1600610614, 1254377835, 1262630547, 205737925
]

def lab(df, ids):
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
    df = df[df['CLIENT_CODE_X'].isin(ids)]

    # Filter relevant lab items
    df = df[df['ITEM_OMS'].isin([
        'Kalium (mmol/l)', 'Leucocyten (10^9/l)', 'ALAT (GPT) (U/l)', 'ASAT (GOT) (U/l)',
        'Fosfaat anorganisch (mmol/l)', 'Magnesium (mmol/l)', 'Glucose (n.n.) (mmol/l)'
    ])].copy()

    # Convert date and extract UNIX timestamp
    df['DATUM_BEPALING'] = pd.to_datetime(df['DATUM_BEPALING'], errors='coerce')
    df['DATE'] = df['DATUM_BEPALING'].dt.normalize()
    df['DATE'] = df['DATE'].view('int64') // 10 ** 9

    # Drop unneeded columns
    df.drop(columns=['DATUM_BEPALING', 'ITEM_CODE'], inplace=True)

    # Rename columns
    df.rename(columns={
        'CLIENT_CODE_X': 'PATIENT_ID',
        'SEQ_MED_AANMELDING': 'INTAKE_ID',
        'ITEM_OMS': 'CHEMICAL_VALUE',
        'UITSLAG_WAARDE': 'VALUE_RESULT'
    }, inplace=True)

    df.sort_values(by=['PATIENT_ID', 'INTAKE_ID', 'DATE'], inplace=True)
    df['SEQUENCE'] = df.groupby(['PATIENT_ID', 'INTAKE_ID']).cumcount() + 1

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
        'ASAT (GOT) (U/l)': 'AST',
        'ALAT (GPT) (U/l)': 'ALT',
        'Fosfaat anorganisch (mmol/l)': 'Phosphate',
        'Kalium (mmol/l)': 'Potassium',
        'Leucocyten (10^9/l)': 'Leucocytes',
        'Glucose (n.n.) (mmol/l)': 'Glucose',
        'Magnesium (mmol/l)': 'Magnesium',
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

def vitals(df, ids):
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

    df = df[df['CLIENT_CODE_X'].isin(ids)]

    df = df.copy()

    # Convert to datetime and extract date/time
    df['DT_METING'] = pd.to_datetime(df['DT_METING'], errors='coerce')
    df = df.dropna(subset=['DT_METING'])  # Drop invalid datetimes
    df['DATE'] = df['DT_METING'].dt.normalize()
    df['DATE'] = df['DATE'].view('int64') // 10**9  # Unix timestamp

    # Drop unnecessary columns
    df.drop(columns=['OPMERKING', 'DT_METING'], inplace=True)

    # Rename columns
    df.rename(columns={
        'CLIENT_CODE_X': 'PATIENT_ID',
        'SEQ_MED_AANMELDING': 'INTAKE_ID',
        'O_METING': 'MEASUREMENT ITEM',
        'WAARDE1': 'VALUE 1',
        'WAARDE2': 'VALUE 2'
    }, inplace=True)

    # Filter relevant measurement items
    df = df[df['MEASUREMENT ITEM'].isin([
        'Body Mass Index', 'Tensie / Pols', 'Temperatuur (c)'
    ])].copy()

    # Sort and reset index
    df.sort_values(by=['PATIENT_ID', 'INTAKE_ID', 'DATE'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Melt value columns
    df_melted = df.melt(
        id_vars=['PATIENT_ID', 'INTAKE_ID', 'DATE', 'MEASUREMENT ITEM'],
        value_vars=['VALUE 1', 'VALUE 2'],
        var_name='VALUE_TYPE',
        value_name='VALUE'
    )

    # Make measurement item labels unique (e.g., "Tensie / Pols 1", "Tensie / Pols 2")
    df_melted['MEASUREMENT ITEM'] = df_melted['MEASUREMENT ITEM'] + " " + df_melted.groupby(
        ['PATIENT_ID', 'INTAKE_ID', 'DATE', 'MEASUREMENT ITEM']
    ).cumcount().add(1).astype(str)

    # Pivot to wide format
    df_pivot = df_melted.pivot_table(
        index=['PATIENT_ID', 'INTAKE_ID', 'DATE'],
        columns='MEASUREMENT ITEM',
        values='VALUE',
        aggfunc='first'
    ).reset_index()

    # Calculate Height and BMI
    df_pivot['Body Mass Index 1'] = pd.to_numeric(df_pivot['Body Mass Index 1'], errors='coerce')
    df_pivot['Body Mass Index 2'] = pd.to_numeric(df_pivot['Body Mass Index 2'], errors='coerce')
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
        'PATIENT_ID', 'INTAKE_ID', 'DATE',
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

def age(df, ids):
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
    df = df[df['Client_code_x'].isin(ids)]

    df = df.copy()

    # Convert date column and extract UNIX timestamp
    df['datum_baseline'] = pd.to_datetime(df['datum_baseline'])
    df['DATE'] = df['datum_baseline'].dt.normalize()
    df['DATE'] = df['DATE'].view('int64') // 10**9  # UNIX timestamp

    df = df[df['Geslacht'] != 'Man']

    # Convert INTAKE_ID to int
    df['aanmeldnummer'] = df['aanmeldnummer'].astype(int)

    # Select relevant columns
    df = df[[
        'aanmeldnummer', 'Client_code_x', 'DATE', 'leeftijd_baseline'
    ]]

    # Rename columns
    df.rename(columns={
        'Client_code_x': 'PATIENT_ID',
        'aanmeldnummer': 'INTAKE_ID',
        'leeftijd_baseline': 'AGE'
    }, inplace=True)

    df['SEQUENCE'] = df.groupby(['PATIENT_ID', 'INTAKE_ID']).cumcount() + 1

    # Reorder columns
    df = df[['PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE', 'AGE']]

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

    df_lab['PATIENT_ID'] = pd.to_numeric(df_lab['PATIENT_ID'], errors='coerce').astype('Int64')
    df_age['PATIENT_ID'] = pd.to_numeric(df_age['PATIENT_ID'], errors='coerce').astype('Int64')

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
        'PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE', 'AGE',
        'ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium', 'Leucocytes', 'Magnesium'
    ]]

    # Aggregate by patient and date, taking the first valid entry per group
    df_final = df_merged.groupby(['PATIENT_ID', 'DATE'], as_index=False).agg({
        'INTAKE_ID': 'first',
        'SEQUENCE': 'first',
        'AGE': 'first',
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
    Adds specific patients back in post-cleaning.
    Applies:
    - Intake duration limit (12 weeks)
    - Removal of isolated early timestamps
    - Recalculation of SEQUENCE
    """

    df_lab_age['PATIENT_ID'] = pd.to_numeric(df_lab_age['PATIENT_ID'], errors='coerce').astype('Int64')
    df_vitals['PATIENT_ID'] = pd.to_numeric(df_vitals['PATIENT_ID'], errors='coerce').astype('Int64')
    df_lab_age['INTAKE_ID'] = pd.to_numeric(df_lab_age['INTAKE_ID'], errors='coerce').astype('Int64')
    df_vitals['INTAKE_ID'] = pd.to_numeric(df_vitals['INTAKE_ID'], errors='coerce').astype('Int64')

    # Merge lab+age data with vitals
    df_combined = pd.merge(
        df_lab_age, df_vitals,
        on=['DATE', 'PATIENT_ID', 'INTAKE_ID'],
        how='left',
        suffixes=('_final', '_merged')
    )

    # Handle duplicate SEQUENCE columns
    df_combined.rename(columns={'SEQUENCE_final': 'SEQUENCE'}, inplace=True)
    df_combined.drop(columns=['SEQUENCE_merged'], inplace=True)

    # Convert DATE to datetime for temporal logic
    df_combined['DATE'] = pd.to_datetime(df_combined['DATE'], unit='s')

    # Step 1: Limit each intake to max 12 weeks
    df_combined = df_combined[
        df_combined.groupby(['PATIENT_ID', 'INTAKE_ID'])['DATE']
        .transform(lambda x: x - x.min()) <= timedelta(weeks=12)
    ]

    # Reorder columns
    new_column_order = [
        'PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'DATE',
        'AGE', 'Weight (kg)', 'Height (m)', 'BMI',
        'Temperature (C)', 'Systolic', 'Diastolic',
        'ALT', 'AST', 'Phosphate', 'Glucose',
        'Potassium', 'Leucocytes', 'Magnesium'
    ]
    df_combined = df_combined[new_column_order]

    # Convert lab values to numeric
    columns_to_convert = [
        'ALT', 'AST', 'Phosphate', 'Glucose',
        'Potassium', 'Leucocytes', 'Magnesium'
    ]
    df_combined[columns_to_convert] = df_combined[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Fill in missing heights per patient
    df_combined['Height (m)'] = df_combined.groupby('PATIENT_ID')['Height (m)'].transform(lambda x: x.ffill().bfill())

    # Clean known outliers
    df_combined['Temperature (C)'] = df_combined['Temperature (C)'].replace([43, 33.7], float('nan'))
    df_combined['Systolic'] = df_combined['Systolic'].replace(0, float('nan'))
    df_combined['Diastolic'] = df_combined['Diastolic'].replace(0, float('nan'))

    # Compute DAYS_SINCE_ADMISSION
    df_combined['DAYS_SINCE_ADMISSION'] = (
        df_combined.groupby(['PATIENT_ID', 'INTAKE_ID'])['DATE']
        .transform(lambda x: (x - x.min()).dt.days)
    )

    # Convert DATE back to UNIX
    df_combined['DATE'] = df_combined['DATE'].view('int64') // 10**9

    # Load BEP_imputed.csv and append specific patients (929, 1363)
    df_extra = pd.read_csv("Data/BEP.csv")
    df_selected = df_extra[df_extra['PATIENT_ID'].isin([929, 1363])]
    df = pd.concat([df_combined, df_selected], ignore_index=True)

    added_patients = df[df['PATIENT_ID'].isin([929, 1363])]
    print(f"✅ Added {len(added_patients)} rows for PATIENT_IDs 929 and 1363")

    # Set default flags
    df['RFS'] = 0
    df['CONTROL'] = 1

    # only keep patients with three or more measurements
    patient_counts = df['PATIENT_ID'].value_counts()
    df = df[df['PATIENT_ID'].isin(patient_counts[patient_counts >= 3].index)]

    return df


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
    df['Temperature (C)'] = pd.to_numeric(df['Temperature (C)'], errors='coerce').astype('float64')
    df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce').astype('float64')
    df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce').astype('float64')

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
        datasets=3,
        random_state=123,
        mean_match_candidates=5
    )

    # Perform imputation with Predictive Mean Matching
    # removed mean_match_candidates = 5
    kernel.mice(
        iterations=20
    )

    # Randomly select one of the imputed datasets
    random_index = random.randint(0, kernel.dataset_count() - 1)

    # Extract the randomly selected imputed dataset
    imputed_df = kernel.complete_data(dataset=random_index)

    # Merge imputed values back into the original dataframe
    df[cols_to_impute] = imputed_df[cols_to_impute]

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

    # Recalculate BMI
    df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)

    df.sort_values(by=['PATIENT_ID', 'INTAKE_ID', 'DATE'], inplace=True)
    df['SEQUENCE'] = df.groupby(['PATIENT_ID', 'INTAKE_ID']).cumcount() + 1

    return df

def add_time_features(df):
    """
    Generates time-based features by calculating the per-day absolute and percentage change
    in key clinical values for each patient, accounting for variable time intervals.

    :param df: pandas DataFrame with a 'days_since_admission' column and clinical features.
    :return: Tuple of two pandas DataFrames:
             - df_delta_per_day: with delta per day features (e.g., ALT_delta_per_day)
             - df_pct_per_day: with percent change per day features (e.g., ALT_percent_change_per_day)
    """
    columns_time_feat = [
        'ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium',
        'Magnesium', 'Weight (kg)', 'BMI', 'Temperature (C)', 'Systolic', 'Diastolic', 'Leucocytes'
    ]

    df = df.sort_values(by=['PATIENT_ID', 'DAYS_SINCE_ADMISSION']).copy()

    # Compute time differences
    df['day_diff'] = df.groupby('PATIENT_ID')['DAYS_SINCE_ADMISSION'].diff().fillna(1)

    # To avoid division by zero
    df['day_diff'] = df['day_diff'].replace(0, 1)

    # Create new DataFrames for delta and percentage change per day
    df_delta = df.copy()
    df_pct = df.copy()

    for col in columns_time_feat:
        # Delta per day
        raw_diff = df.groupby('PATIENT_ID')[col].diff()
        df_delta[f'{col}_delta_per_day'] = (raw_diff / df['day_diff']).fillna(0)

        # Percent change per day
        pct_change = df.groupby('PATIENT_ID')[col].pct_change()
        df_pct[f'{col}_percent_change_per_day'] = (pct_change / df['day_diff']).fillna(0) * 100

    # Drop original columns and temporary ones
    df_delta.drop(columns=columns_time_feat + ['day_diff'], inplace=True)
    df_pct.drop(columns=columns_time_feat + ['day_diff'], inplace=True)

    return df_delta, df_pct

print("🔄 Loading and preprocessing lab data...")
# Load the lab files
df_lab1 = pd.read_csv("../../Daisy_lab_part1.csv")
df_lab2 = pd.read_csv("../../Daisy_lab_part2.csv")
# Concatenate them
df_labs = pd.concat([df_lab1, df_lab2], ignore_index=True)
lab_clean = lab(df_labs, ids)
print("✅ Lab data preprocessed.\n")

print("🔄 Loading and preprocessing vitals data...")
df_vital1 = pd.read_csv("../../Daisy_metingen_part1.csv", encoding='latin1')
df_vital2 = pd.read_csv("../../Daisy_metingen_part2.csv", encoding='latin1')
df_vitals = pd.concat([df_vital1, df_vital2], ignore_index=True)
vitals_clean = vitals(df_vitals, ids)
print("✅ Vitals data preprocessed.\n")

print("🔄 Loading and preprocessing demographics data...")
age_clean = age(pd.read_csv("../../Daisy_main.csv", sep = ';'), ids)
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

print("🛠️ Performing final simple imputations for AGE, and HEIGHT...")
print("🛠️ Standardising all the data...")
print("🛠️ Saving the imputed dataset to a CSV file.")
df_final = final_imputations_and_export(df_imputed)
df_final.to_csv(output_imputed_test, index=False)
print(f"✅ Final dataset saved to: {output_imputed_test}\n")

print("⏱️ Generating time-based features (delta and percent change)...")
df_deltas, df_pct_changes = add_time_features(df_final)
df_deltas.to_csv(output_delta_test, index=False)
df_pct_changes.to_csv(output_percentage_change_test, index=False)
print("✅ Time-based delta and percent change features created.\n")
print(f"✅ Delta dataset saved to: {output_delta_test}\n")
print(f"✅ Percentage change dataset saved to: {output_percentage_change_test}\n")


print("🔗 Merging test set data...")
# Input file paths (v1 and v2)
files_v1 = {
    "imputed_test": "Data/BEP_imputed_TEST1.csv",
    "delta_test": "Data/BEP_imputed_delta_TEST1.csv",
    "pct_change_test": "Data/BEP_imputed_percentage_change_TEST1.csv"
}

files_v2 = {
    "imputed_test": "Data/BEP_imputed_TEST2.csv",
    "delta_test": "Data/BEP_imputed_delta_TEST2.csv",
    "pct_change_test": "Data/BEP_imputed_percentage_change_TEST2.csv"
}

# Output paths
output_paths = {
    "imputed_test": "Data/BEP_imputed_TEST.csv",
    "delta_test": "Data/BEP_imputed_delta_TEST.csv",
    "pct_change_test": "Data/BEP_imputed_percentage_change_TEST.csv"
}

# Concatenate and save
for key in files_v1.keys():
    df1 = pd.read_csv(files_v1[key])
    df2 = pd.read_csv(files_v2[key])

    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined.to_csv(output_paths[key], index=False)
    print(f"✅ Saved combined file to: {output_paths[key]}")

print("✅ Final merged test dataset ready.\n")

all_versioned_files = list(files_v1.values()) + list(files_v2.values())

# Delete each file
for file_path in all_versioned_files:
    try:
        os.remove(file_path)
        print(f"🗑️ Deleted: {file_path}")
    except FileNotFoundError:
        print(f"⚠️ File not found (skipped): {file_path}")
    except Exception as e:
        print(f"❌ Error deleting {file_path}: {e}")






