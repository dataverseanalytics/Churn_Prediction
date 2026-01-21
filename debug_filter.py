from data_processor import DataProcessor

# Initialize
dp = DataProcessor('membership-records from Gomask.csv', 'subscription-billing from GoMask.csv')
dp.load_data()
dp.merge_data()

# Check filtering
df = dp.merged_df
df['invoice_balance'] = df.apply(dp.calculate_invoice_balance, axis=1)
df['is_individual'] = df['email'].apply(dp.is_individual_bill_to)

print("="*60)
print("DATA FILTER ANALYSIS")
print("="*60)
print(f"\nTotal records: {len(df)}")
print(f"Records with invoice_balance > 0: {len(df[df['invoice_balance'] > 0])}")
print(f"Records marked as individual: {len(df[df['is_individual'] == True])}")
print(f"Records meeting both conditions: {len(df[(df['invoice_balance'] > 0) & (df['is_individual'] == True)])}")

print("\n" + "="*60)
print("SAMPLE EMAILS AND THEIR CLASSIFICATION")
print("="*60)
sample = df[['email', 'is_individual', 'invoice_balance']].head(30)
print(sample.to_string())

print("\n" + "="*60)
print("UNPAID INVOICES")
print("="*60)
unpaid = df[df['invoice_balance'] > 0][['email', 'is_individual', 'invoice_balance']].head(20)
print(unpaid.to_string())
