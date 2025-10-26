import pandas as pd

excel_file = "pet_pri_gnd_a_epmr_pte_dpgal_w.xls"

xls = pd.ExcelFile(excel_file)

# Loop through all sheets except the metadata sheet
for sheet_name in xls.sheet_names:
    if sheet_name.lower() == "workbook contents":
        continue  # skip the metadata sheet

    # Read the sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Clean up empty columns if any
    df = df.dropna(axis=1, how='all')
    
    # Save to CSV
    csv_file = f"{sheet_name}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved {csv_file}")
