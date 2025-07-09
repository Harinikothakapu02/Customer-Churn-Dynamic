import pyodbc
import pandas as pd
from pathlib import Path

def fetch_data():
    try:
        server = r'DELL-BVOELMRLLM\SQLEXPRESS'
        database = 'db_churn'
        table_name = 'stg_Churn'  # Table name
        
        # Connect to SQL Server
        conn = pyodbc.connect(
            f'DRIVER={{SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            'Trusted_Connection=yes;'
        )
        
        # Check if table exists
        tables = pd.read_sql("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES", conn)
        if table_name not in tables['TABLE_NAME'].values:
            raise ValueError(f"Table '{table_name}' not found in database")
        
        # Get count
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        count = pd.read_sql(count_query, conn).iloc[0, 0]
        print(f"Found {count} records in {table_name}")
        
        if count == 0:
            raise ValueError("Table is empty - no data to export")
        
        print("Fetching data...")
        chunks = []
        chunk_size = 1000
        
        for i, chunk in enumerate(pd.read_sql(f"SELECT * FROM {table_name}", conn, chunksize=chunk_size), start=1):
            chunks.append(chunk)
            print(f"Fetched {i * chunk_size} rows...")
        
        df = pd.concat(chunks)
        
        # Save to Desktop CustomerChurn_Dynamic/data/fetched_data.xlsx
        output_path = Path(r"C:\Users\vihaan\Desktop\CustomerChurn_Dynamic\data\fetched_data.xlsx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving Excel file at: {output_path}")
        df.to_excel(output_path, index=False)
        
        if output_path.stat().st_size < 1024:
            raise ValueError("Exported file is too small - possible export failure")
        
        print(f"✅ Success! {len(df)} records saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify table name is correct")
        print("2. Check SQL Server Management Studio for data")
        print("3. Try a simpler query like 'SELECT TOP 10 * FROM table'")
        print("4. Check file permissions and if the Excel file is open")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    fetch_data()
