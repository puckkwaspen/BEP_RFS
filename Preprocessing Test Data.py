"""
Preprocessing Pipeline for DAISY Clinical Test Data
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
import random

#### specify the output paths of the created dataframes to be saved to CSVs here ####
output_imputed_test = "BEP_imputed_TEST.csv"
output_imputed_time_feat_test = "BEP_imputed_time_feat_TEST.csv"
output_delta_test = "BEP_imputed_delta_TEST.csv"
output_percentage_change_test = "BEP_imputed_percentage_change_TEST.csv"

def demo_test(df_demo, df_main):
    """
    This function generates masked IDs where needed. It checks which masked IDs already exist,
    and generates new one with similar length. The function then converts the data to the needed
    type and renames and reorders the column.
    :param df_demo: dataframe that still needs to be (partially) masked
    :param df_main: dataframe that is already masked
    :return: the masked dataframe
    """
    main_int = df_main['INTAKE_ID'].dropna().astype(int).unique().tolist()
    new_int = df_demo['intid'].dropna().astype(int).unique().tolist()
    main_pid = df_main['PATIENT_ID'].dropna().astype(int).unique().tolist()
    new_pid = df_demo['pid'].dropna().astype(int).unique().tolist()

    # Combine sets for uniqueness checks
    existing_int_ids = set(main_int + new_int)
    existing_pids = set(main_pid + new_pid)

    # Determine typical lengths
    typical_int_len = int(pd.Series(main_int).astype(str).str.len().median())
    typical_pid_len = int(pd.Series(main_pid).astype(str).str.len().median())

    # Apply masking for missing values
    for idx, row in df_demo.iterrows():
        if pd.isna(row['intid']):
            df_demo.at[idx, 'intid'] = generate_unique_id(typical_int_len, existing_int_ids)
        if pd.isna(row['pid']):
            df_demo.at[idx, 'pid'] = generate_unique_id(typical_pid_len, existing_pids)

    df = df_demo.drop_duplicates(subset='pid', keep='first').copy()

    # Safe datetime conversion with dayfirst
    df['datum_baseline'] = pd.to_datetime(df['datum_baseline'], dayfirst=True)

    # Convert to UNIX timestamp
    df['DATE'] = df['datum_baseline'].astype('int64') // 10 ** 9

    # Convert to integers
    df['intid'] = df['intid'].astype(int)
    df['pid'] = df['pid'].astype(int)

    # Rename columns
    df = df.rename(columns={
        'pid': 'PATIENT_ID',
        'intid': 'INTAKE_ID',
        'leeftijd_baseline': 'AGE',
        'Geslacht': 'SEX'
    })

    # Map gender
    df['SEX'] = df['SEX'].map({'Vrouw': 1, 'Man': 0})

    # Reorder columns
    df = df[['PATIENT_ID', 'INTAKE_ID', 'DATE', 'SEX', 'AGE', 'cid', 'ggzob_id']]

    return df

def generate_unique_id(length, existing_ids):
    while True:
        new_id = random.randint(10**(length - 1), 10**length - 1)
        if new_id not in existing_ids:
            existing_ids.add(new_id)  # Add to set to prevent reuse
            return new_id


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

def lab_test(df_lab, df_demo):

    # Filter relevant lab items
    df = df_lab[df_lab['O_AANVR_UITSLAG_ITEM_LANG'].isin([
        'Kalium', 'Leucocyten', 'ALAT (GPT)', 'ASAT (GOT)',
        'Fosfaat anorganisch', 'Magnesium', 'Glucose (n.n.)'
    ])].copy()

    # Convert date and extract UNIX timestamp
    df['DT_BEPALING'] = pd.to_datetime(df['DT_BEPALING'], dayfirst=True, errors='coerce')
    df['DATE'] = df['DT_BEPALING'].dt.normalize()
    df['DATE'] = df['DATE'].astype('int64') // 10 ** 9

    df['ggzob_id'] = df['ggzob_id'].astype(int)

    # Drop unneeded columns
    df.drop(columns=['STATUS_AANVRAAG', 'O_STATUS_UITSLAG', 'DT_BEPALING', 'SEQ_ZPAT_PATIENT', 'UITSLAG_CONCLUSIE',
                         'UITSLAG_TEKST_LAB', 'NORMAALWAARDE', 'AANVRAAG_NUMMER', 'UITSLAGREGEL', 'intid'],
                inplace=True)

    df = df[df['ggzob_id'].isin(df_demo['ggzob_id'])]

    # get the pseudonimized patient_id and intake_id from the df_demo
    df = df.merge(df_demo[['PATIENT_ID', 'ggzob_id', 'INTAKE_ID']], on='ggzob_id', how='left')

    # Rename columns
    df.rename(columns={
        'O_AANVR_UITSLAG_ITEM_LANG': 'CHEMICAL_VALUE',
        'UITSLAG_WAARDE': 'VALUE_RESULT'
    }, inplace=True)

    # Sort and pivot
    df.sort_values(by='PATIENT_ID', inplace=True)

    df_lab = df.pivot_table(
        index=['PATIENT_ID', 'INTAKE_ID', 'DATE', 'ggzob_id'],
        columns='CHEMICAL_VALUE',
        values='VALUE_RESULT',
        aggfunc='first'
    ).rename_axis(None, axis=1).reset_index()

    # Add row index
    df_lab['ROW'] = range(1, len(df_lab) + 1)
    df_lab.set_index('ROW', inplace=True)

    # Keep only patients with 3 or more measurements
    patient_counts = df_lab['PATIENT_ID'].value_counts()
    df_lab = df_lab[df_lab['PATIENT_ID'].isin(patient_counts[patient_counts >= 3].index)]

    # Rename columns to clean variable names
    df_lab.rename(columns={
        'ASAT (GOT)': 'AST',
        'ALAT (GPT)': 'ALT',
        'Fosfaat anorganisch': 'Phosphate',
        'Kalium': 'Potassium',
        'Leucocyten': 'Leucocytes',
        'Glucose (n.n.)': 'Glucose'
    }, inplace=True)

    # Clean and convert columns
    for col in ['ALT', 'AST']:
        df_lab[col] = df_lab[col].apply(clean_column)

    to_convert = ['Magnesium', 'ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium', 'Leucocytes']
    df_lab[to_convert] = df_lab[to_convert].apply(pd.to_numeric, errors='coerce')

    df_lab['SEQUENCE'] = df_lab.groupby(['PATIENT_ID', 'INTAKE_ID'])['DATE'].rank(method='first').astype(int)

    return df_lab


def vitals_test(df_vitals, df_demo):
    # Convert to datetime and extract date/time
    df_vitals['DT_METING'] = pd.to_datetime(df_vitals['DT_METING'], dayfirst=True, errors='coerce')
    df_vitals['DATE'] = df_vitals['DT_METING'].dt.normalize()
    df_vitals['TIME'] = df_vitals['DT_METING'].dt.time
    df_vitals['DATE'] = df_vitals['DATE'].astype('int64') // 10 ** 9  # Unix timestamp

    # Filter relevant measurement items
    df_vitals = df_vitals[df_vitals['O_METING'].isin([
        'Body Mass Index', 'Tensie / Pols', 'Temperatuur (c)'
    ])].copy()

    df_vitals[['ggzob_id', 'cid']] = df_vitals[['ggzob_id', 'cid']].astype(int)

    df_vitals = df_vitals[df_vitals['ggzob_id'].isin(df_demo['ggzob_id'])]

    # get the pseudonimized patient_id and intake_id from the df_demo
    df_vitals = df_vitals.merge(
        df_demo[['ggzob_id', 'PATIENT_ID', 'INTAKE_ID']],
        on='ggzob_id',
        how='left'
    )

    # Drop unnecessary columns
    df_vitals.drop(columns=['DT_METING', 'TIME', 'OPMERKING', 'intid', 'pid'], inplace=True)

    # Rename columns
    df_vitals.rename(columns={
        'O_METING': 'MEASUREMENT ITEM',
        'WAARDE1': 'VALUE 1',
        'WAARDE2': 'VALUE 2'
    }, inplace=True)

    # Sort and reset index
    df_vitals.sort_values(by=['PATIENT_ID', 'INTAKE_ID', 'DATE'], inplace=True)
    df_vitals.reset_index(drop=True, inplace=True)

    # Melt value columns
    df_melted = df_vitals.melt(
        id_vars=['PATIENT_ID', 'INTAKE_ID', 'ggzob_id', 'cid', 'DATE', 'MEASUREMENT ITEM'],
        value_vars=['VALUE 1', 'VALUE 2'],
        var_name='VALUE_TYPE',
        value_name='VALUE'
    )

    # Append ' 1' or ' 2' based on VALUE_TYPE to MEASUREMENT ITEM
    df_melted['MEASUREMENT ITEM'] = (
                df_melted['MEASUREMENT ITEM'] + ' ' + df_melted['VALUE_TYPE'].str.extract(r'(\d)')[0].fillna(''))

    # Pivot to wide format
    df_pivot = df_melted.pivot_table(
        index=['PATIENT_ID', 'INTAKE_ID', 'DATE'],
        columns='MEASUREMENT ITEM',
        values='VALUE',
        aggfunc='first'
    ).reset_index()

    # Handle integer-like columns with commas
    cols_to_int = ['Body Mass Index 1', 'Tensie / Pols 1', 'Tensie / Pols 2']
    for col in cols_to_int:
        df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce').astype('Int64')  # Convert to nullable int

    # Handle float columns (only need one conversion per column)
    cols_to_float = ['Body Mass Index 2', 'Temperatuur (c) 1']
    for col in cols_to_float:
        df_pivot[col] = df_pivot[col].astype(str).str.replace(',', '.', regex=False)
        df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce').astype(float)

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

    df_final.columns.name = None

    desired_order = [
        'PATIENT_ID', 'INTAKE_ID', 'DATE', 'SEQUENCE',
        'Height (m)', 'Weight (kg)', 'BMI',
        'Systolic', 'Diastolic', 'Temperature (C)'
    ]

    df_final = df_final[desired_order]

    return df_final


def merge_test(df_demo, df_lab, df_vitals):

    # Ensure only patients in both datasets are kept
    df_lab = df_lab[df_lab['PATIENT_ID'].isin(df_demo['PATIENT_ID'])].copy()

    # Merge lab and age data
    df_merged = df_lab.merge(df_demo, on=['PATIENT_ID', 'INTAKE_ID'], how='left')

    # Rename overlapping columns to keep only relevant ones
    df_merged.rename(columns={
        'DATE_x': 'DATE',
        'ggzob_id_x': 'ggzob_id',
    }, inplace=True)

    # Drop unnecessary duplicates
    df_merged.drop(columns=['DATE_y', 'ggzob_id_y'], inplace=True)

    # Reorder columns
    df_merged = df_merged[[
        'PATIENT_ID', 'INTAKE_ID', 'SEQUENCE', 'cid', 'ggzob_id', 'DATE', 'AGE', 'SEX',
        'ALT', 'AST', 'Phosphate', 'Glucose', 'Potassium', 'Leucocytes', 'Magnesium'
    ]]

    # Aggregate by patient and date, taking the first valid entry per group
    df_tog = df_merged.groupby(['PATIENT_ID', 'DATE'], as_index=False).agg({
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

    # Merge lab+age data with vitals on DATE, PATIENT_ID, INTAKE_ID
    df_combined = pd.merge(
        df_tog, df_vitals,
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
    df_combined['Temperature (C)'] = df_combined['Temperature (C)'].replace([45.5], float('nan'))
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

    # Randomly select one of the imputed datasets
    random_index = random.randint(0, kernel.num_datasets - 1)

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

    # Mode imputation for categorical column
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df[['SEX']] = mode_imputer.fit_transform(df[['SEX']])

    # Recalculate BMI
    df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)

    return df


def RFS_labels(df):
    # Define electrolytes to monitor
    electrolytes = ['Phosphate', 'Potassium', 'Magnesium']

    # Create placeholder columns for drop %
    for col in electrolytes:
        df[f'{col}_DROP_%'] = None

    # Group by patient-intake
    rfs_labeled_groups = []

    for _, group in df.groupby(['PATIENT_ID', 'INTAKE_ID']):
        group = group.sort_values('SEQUENCE')
        baseline = group.iloc[0]

        # Calculate % drop from baseline
        for col in electrolytes:
            base_value = baseline[col]
            group[f'{col}_DROP_%'] = (base_value - group[col]) / base_value * 100

        # Create RFS label if any electrolyte drop ≥ 10%
        group['RFS'] = ((group[[f'{col}_DROP_%' for col in electrolytes]] >= 20).any(axis=1)).astype(int)

        rfs_labeled_groups.append(group)

    # Combine groups
    df_final = pd.concat(rfs_labeled_groups).reset_index(drop=True)

    # Drop the temporary drop % columns
    drop_cols = [f'{col}_DROP_%' for col in electrolytes]
    df_final.drop(columns=drop_cols, inplace=True)

    # Move RFS column just after 'SEX'
    cols = df_final.columns.tolist()
    if 'RFS' in cols:
        cols.remove('RFS')
        insert_at = cols.index('SEX') + 1 if 'SEX' in cols else len(cols)
        cols.insert(insert_at, 'RFS')
        df_final = df_final[cols]

    return df_final

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
        df_delta[f'{col}_delta'] = df_delta.groupby('PATIENT_ID')[col].diff().fillna(0)

        # Percent change
        df_pct[f'{col}_percent_change'] = df_pct.groupby('PATIENT_ID')[col].pct_change().fillna(0) * 100

    # Drop the original columns from both
    df_delta.drop(columns=columns_time_feat, inplace=True)
    df_pct.drop(columns=columns_time_feat, inplace=True)

    return df_delta, df_pct


print("🔄 Loading and preprocessing demographics data...")
df_demo = demo_test(pd.read_csv("../anonymized_Labels_refeeding.csv", sep='\t'), pd.read_csv("BEP_imputed.csv"))
print("✅ Demographics data preprocessed.\n")

print("🔄 Loading and preprocessing lab data...")
df_lab = lab_test(pd.read_csv("../anonymized_Labels_refeeding_lab.csv", sep='\t'), df_demo)
print("✅ Lab data preprocessed.\n")

print("🔄 Loading and preprocessing vitals data...")
df_vitals = vitals_test(pd.read_csv("../anonymized_Labels_refeeding_metingen.csv", sep='\t' ), df_demo)
print("✅ Vitals data preprocessed.\n")

print("🔗 Merging datasets together...")
df_merge= merge_test(df_demo, df_lab, df_vitals)
print("✅ Datasets merged.\n")

print("📊 Here's a quick look at the merged dataset (first 10 rows):\n")
print(df_merge.head(10))
print()

print("⚙️ Starting MICE imputation on selected clinical features...")
df_impute = mice_imputation(df_merge)
print("✅ MICE imputation complete.\n")

print("🛠️ Performing final simple imputations for AGE, HEIGHT, and SEX...")
print("🛠️ Adding labels for RFS...")
print("🛠️ Saving the imputed dataset to a CSV file.")
df_final = final_imputations_and_export(df_impute)
df_final = RFS_labels(df_final)
df_final.to_csv(output_imputed_test, index=False)
print(f"✅ Final dataset saved to: {output_imputed_test}\n")

print("⏱️ Generating time-based features (delta and percent change)...")
df_deltas, df_pct_changes = add_time_features(df_final)
df_deltas.to_csv(output_delta_test, index=False)
df_pct_changes.to_csv(output_percentage_change_test, index=False)
print("✅ Time-based delta and percent change features created.\n")
print(f"✅ Delta dataset saved to: {output_delta_test}\n")
print(f"✅ Percentage change dataset saved to: {output_percentage_change_test}\n")

