"""
seed_events.py - Seed Macro Events from CSV to Supabase
======================================================
Reads data/macro_events_india.csv and uploads it to Supabase public.macro_events.

Usage:
  python seed_events.py
"""

import os
import sys
import pandas as pd

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from supabase_db import save_macro_events, SUPABASE_ENABLED

def main():
    if not SUPABASE_ENABLED:
        print("[-] Error: Supabase connection is not configured or disabled.")
        print("    Ensure SUPABASE_URL and SUPABASE_KEY are defined in your .env file.")
        sys.exit(1)
        
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "macro_events_india.csv")
    if not os.path.exists(csv_path):
        print(f"[-] Error: Event calendar CSV not found at: {csv_path}")
        sys.exit(1)
        
    print(f"[*] Reading events from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[-] Error reading CSV: {e}")
        sys.exit(1)
        
    required_cols = ['Date', 'Event_Type', 'Outcome']
    for col in required_cols:
        if col not in df.columns:
            print(f"[-] Error: Missing required column '{col}' in CSV.")
            sys.exit(1)
            
    # Convert dataframe to records
    records = df.to_dict('records')
    print(f"[*] Found {len(records)} events. Uploading to Supabase...")
    
    success = save_macro_events(records)
    if success:
        print("[+] Success: All macro events seeded successfully into Supabase!")
    else:
        print("[-] Error: Failed to seed macro events into Supabase. Check logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
