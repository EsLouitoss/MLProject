import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('known_exploited_vulnerabilities.csv')

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Basic dataset information
print("=== DATASET OVERVIEW ===")
print(f"Number of vulnerabilities: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print(f"Date range: {df['dateAdded'].min()} to {df['dateAdded'].max()}")

# Display basic info
print("\n=== DATASET INFO ===")
print(df.info())

# Display first few rows
print("\n=== FIRST 5 ROWS ===")
display(df.head())

# Check for missing values
print("\n=== MISSING VALUES ===")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_info[missing_info['Missing Count'] > 0])

# Basic statistics for numerical columns
print("\n=== BASIC STATISTICS ===")
print(df.describe())

# Key metrics analysis
print("=== KEY METRICS ===")
print(f"Unique vendors: {df['vendorProject'].nunique()}")
print(f"Unique products: {df['product'].nunique()}")
print(f"Vulnerabilities with known ransomware use: {df['knownRansomwareCampaignUse'].value_counts().get('Known', 0)}")
print(f"Vulnerabilities with unknown ransomware use: {df['knownRansomwareCampaignUse'].value_counts().get('Unknown', 0)}")

# Vendor distribution
print("\n=== TOP 10 VENDORS BY VULNERABILITY COUNT ===")
top_vendors = df['vendorProject'].value_counts().head(10)
print(top_vendors)

# Date analysis
df['dateAdded'] = pd.to_datetime(df['dateAdded'])
df['dueDate'] = pd.to_datetime(df['dueDate'])
df['days_to_due'] = (df['dueDate'] - df['dateAdded']).dt.days

print(f"\nAverage days from addition to due date: {df['days_to_due'].mean():.1f} days")

# Data cleaning and feature engineering
print("=== DATA PRE-PROCESSING ===")

# Create a copy for preprocessing
df_processed = df.copy()

# Handle missing values
print("Handling missing values...")
df_processed['cwes'] = df_processed['cwes'].fillna('Unknown')
df_processed['notes'] = df_processed['notes'].fillna('No notes')

# Feature engineering
print("Creating new features...")

# Extract year and month
df_processed['year_added'] = df_processed['dateAdded'].dt.year
df_processed['month_added'] = df_processed['dateAdded'].dt.month

# Create binary target for ransomware use
df_processed['ransomware_target'] = (df_processed['knownRansomwareCampaignUse'] == 'Known').astype(int)

# Create severity indicators based on vulnerability type
def extract_severity_indicator(vuln_name):
    vuln_name_lower = str(vuln_name).lower()
    if any(term in vuln_name_lower for term in ['remote code execution', 'arbitrary code execution', 'code injection']):
        return 'Critical'
    elif any(term in vuln_name_lower for term in ['privilege escalation', 'buffer overflow', 'sql injection']):
        return 'High'
    elif any(term in vuln_name_lower for term in ['denial of service', 'authentication bypass', 'information disclosure']):
        return 'Medium'
    else:
        return 'Low'

df_processed['severity_indicator'] = df_processed['vulnerabilityName'].apply(extract_severity_indicator)

# Create vendor risk profile (vendors with many vulnerabilities might be higher risk)
vendor_risk = df_processed['vendorProject'].value_counts()
df_processed['vendor_risk_score'] = df_processed['vendorProject'].map(vendor_risk)

# Text length features (proxy for complexity)
df_processed['short_desc_length'] = df_processed['shortDescription'].str.len()
df_processed['vuln_name_length'] = df_processed['vulnerabilityName'].str.len()

print("Pre-processing completed!")
print(f"Processed dataset shape: {df_processed.shape}")
