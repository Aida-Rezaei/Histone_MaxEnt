"""
Extracting the average signal from all bigWig files
Focusing on all chromosomes for test with bin size 200 bp
A function called extract_signal is made for this purpose.
"""

# packages
from pathlib import Path #for specifying the paths much easier
import pandas as pd #for dataframe creation and modification
import pybigtools #for reading bigWig files
import pyarrow #for Apache Arrow
import fastparquet # for parquet files
from concurrent.futures import ProcessPoolExecutor #parallel processing
import shutil #changing the directory of the files

PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
DATA_DIR = PROJECT_ROOT / "data" / "raw" #the path to the data
OUTPUT_DIR = PROJECT_ROOT / "results" #the path to where the output file would
OUTPUT_DIR.mkdir(exist_ok = True) #creates the directory if not available. if exists, no error

BIN_SIZE = 200
###
def extract_signal(bigWig_file, OUTPUT_DIR, BIN_SIZE = 200):
    '''
    For exctracting the signals (and other information to create a metadata) of the bigWig files and all of the chromosomes.
    The arguments:
    - bigWig_file: the bigWig file/ path
    - OUTPUT_DIR: the path where the output is saved
    - BIN_SIZE: sizes of sections of the chromosome for finding signal. By default: 200 bp
    The output files are for each bigWig file and in parquet format (to save space compared to CSV)
    '''
    output_file = OUTPUT_DIR / f"{bigWig_file.stem}_signal.parquet"
    if output_file.exists():
        print(f"Skipping {bigWig_file.name}")
        return
    chroms = list(range(1,23)) + ["X", "Y"]
    #for bigWig_file in bigWig:
    print(f"\nOpening {bigWig_file.name}.")
    bw = pybigtools.open(bigWig_file)
    rows = []
    bin_id = 0
    for chrom_num in chroms:
        chrom_length = bw.chroms()[f"chr{chrom_num}"]
        print(f"Processing  chr{chrom_num} ({chrom_length} bp)")
        for start in range(0, chrom_length, BIN_SIZE):
            end = min(start + BIN_SIZE, chrom_length)
            values = bw.values(
                chrom= f"chr{chrom_num}",
                start= start, 
                end = end,
                fillna= 0 #if there are any NA values put 0 instead of them
            )
            rows.append({
                "bin_id": bin_id,
                "chromosome": f"chr{chrom_num}",
                "start": start,
                "end": end,
                "signal": values.mean()
            }) #adding the information to the rows
            bin_id += 1
    df = pd.DataFrame(rows)
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} bins")
    print(f"Saved {output_file}")
    bw.close() #closing the file
if __name__ == "__main__":
    bigWig = sorted(DATA_DIR.glob("*.bigWig"))
    print(f"Found {len(bigWig)} bigWig files.")
    print([file_name.name for file_name in bigWig]) 
    NUM_WORKERS = 3
    with ProcessPoolExecutor(max_workers = NUM_WORKERS) as executor:
        list(
            executor.map(
                extract_signal,
                bigWig,
                [OUTPUT_DIR] * len(bigWig),
                [BIN_SIZE] * len(bigWig)
            )
        )
    print("\nDone!")

METADATA_FILE = PROJECT_ROOT / "results" / "encode_metadata.csv"
df = pd.read_csv(METADATA_FILE)
# print(df.columns)
# ['Cell_Type', 'Histone_Mark', 'Experiment_Accession', 'File_Accession',
#        'Assembly', 'Output_Type', 'File_Format', 'Biological_Replicates',
#        'Download_URL']
# we want Cell_Type_Histone_Mark_signal.parquet


file_counter = 0
for p_file in OUTPUT_DIR.iterdir():
    #print(p_file.name[:11])
    if p_file.name[:11] in set(df["File_Accession"]): #if the first 11 letters of the files's name is found in the "File_Accession" of df table
        print(f"Found {p_file.name}")
        file_counter += 1
        row_index = df[df["File_Accession"] == p_file.name[:11]].index[0]
        #print(row_index)
        cell_type = df.loc[row_index, "Cell_Type"]
        print(f"The cell type is {cell_type}")
        histone_mark = df.loc[row_index, "Histone_Mark"]
        print(f"The histone mark is {histone_mark}")
        newFile = p_file.rename(f"{cell_type}_{histone_mark}_signal.parquet")
        print(f"New name: {newFile}\n")
print(f"{file_counter} files were renamed.")

#changing the directory of the files
NEW_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "signals"
for file_path in PROJECT_ROOT.glob("*.parquet"):
    shutil.move(str(file_path), 
                str(NEW_OUTPUT_DIR / file_path.name))
    print(f"File moved!")