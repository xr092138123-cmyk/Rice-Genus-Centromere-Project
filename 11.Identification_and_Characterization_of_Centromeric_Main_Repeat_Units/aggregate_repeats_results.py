#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_repeats_results.py

Aggregates moddotplot analysis results from a single directory.
It parses metadata (material, chromosome, window size) from filenames
and outputs a single, sorted, tab-delimited summary file.
"""

import os
import pandas as pd
import re
import argparse
import sys

def parse_repeats_file(file_path):
    """
    Reads and parses a single _repeats.txt file.
    Returns a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        df.columns = [col.strip() for col in df.columns]
        
        if 'Percentage_of_Centromere' in df.columns:
            df['Percentage_of_Centromere'] = df['Percentage_of_Centromere'].astype(str).str.replace('%', '').astype(float)
        if 'Cumulative_Percentage' in df.columns:
            df['Cumulative_Percentage'] = df['Cumulative_Percentage'].astype(str).str.replace('%', '').astype(float)
            
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found {file_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"WARNING: Could not parse file {file_path}. Error: {e}", file=sys.stderr)
        return None

def parse_filename(filename):
    """
    Parses material, chromosome, and window size from a filename.
    Expected format: AA_Ogla_hap1.Chr01.w1000_top20_repeats.txt
    """
    window_match = re.search(r'\.w(\d+)_', filename)
    if not window_match:
        return None, None, None
    
    window_size = int(window_match.group(1))
    
    parts = filename.split('.')
    if len(parts) < 3:
        return None, None, None
        
    material = parts[0]
    chromosome = parts[1]
    
    return material, chromosome, window_size

def aggregate_data(input_dir, output_file):
    """
    Main function to traverse directory, parse files, aggregate, sort, and save data.
    """
    all_data = []
    
    print(f"--- Starting aggregation from directory '{input_dir}' ---")

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory '{input_dir}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    file_count = 0
    parsed_count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith("_repeats.txt"):
            file_count += 1
            
            material, chromosome, window_size = parse_filename(filename)
            
            if material is None:
                print(f"WARNING: Filename format mismatch, skipping: {filename}", file=sys.stderr)
                continue
            
            file_path = os.path.join(input_dir, filename)
            df = parse_repeats_file(file_path)
            
            if df is not None:
                df['Material'] = material
                df['Chromosome'] = chromosome
                df['WindowSize'] = window_size
                all_data.append(df)
                parsed_count += 1

    if not all_data:
        print("ERROR: No data was successfully aggregated. Check directory contents and filename format.", file=sys.stderr)
        sys.exit(1)

    print(f"\nScanned {file_count} '_repeats.txt' files, successfully parsed and aggregated {parsed_count}.")

    # Combine all data into a single DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # --- Sorting Logic ---
    # Create a temporary numeric column for proper sorting of 'Top' values (e.g., 1, 2, ..., 10, 11)
    # This prevents alphabetical sorting where 'Top10' comes before 'Top2'
    if 'Top' in final_df.columns:
        final_df['Top_Number'] = final_df['Top'].str.replace('Top', '').astype(int)
        
        # Sort by Material, Chromosome, WindowSize, and then by the new numeric Top_Number
        print("Sorting data numerically by Top rank...")
        final_df.sort_values(by=['Material', 'Chromosome', 'WindowSize', 'Top_Number'], inplace=True)
        
        # Drop the temporary helper column before saving
        final_df.drop(columns=['Top_Number'], inplace=True)
    else:
        # Fallback sorting if 'Top' column doesn't exist for some reason
        final_df.sort_values(by=['Material', 'Chromosome', 'WindowSize'], inplace=True)


    # --- Column Ordering and Saving ---
    # Define the desired final column order
    cols_order = ['Material', 'Chromosome', 'WindowSize', 'Top', 'Seed_Region', 'Copy_Number', 
                  'Family_Cumulative_Length', 'Percentage_of_Centromere', 'Cumulative_Percentage']
    
    # Reorder columns, keeping only those that exist in the DataFrame to avoid errors
    final_cols = [col for col in cols_order if col in final_df.columns]
    final_df = final_df[final_cols]

    # Save to a tab-separated file
    try:
        final_df.to_csv(output_file, sep='\t', index=False)
        print(f"\n--- Aggregation complete! Data saved to '{output_file}' (tab-separated) ---")
        print("Preview of the first 5 rows of aggregated data:")
        print(final_df.head().to_string())
    except IOError as e:
        print(f"ERROR: Could not write to output file '{output_file}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate moddotplot `_repeats.txt` results from a directory. "
                    "Parses metadata from filenames and outputs a single, sorted, tab-delimited file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_dir", 
        help="Directory containing all `..._repeats.txt` result files."
    )
    parser.add_argument(
        "output_file", 
        help="Path for the final aggregated tab-separated output file.\n"
             "Example: 'aggregated_summary.tsv'"
    )

    args = parser.parse_args()
    aggregate_data(args.input_dir, args.output_file)