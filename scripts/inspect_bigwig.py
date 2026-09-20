"""
Inspect one ENCODE bigWig file using pyBigTools.
"""
# packages
from pathlib import Path
import pybigtools
import matplotlib.pyplot as plt #plotting histograms
import numpy as np #for log


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Pick the first bigWig file
bigwig_file = sorted(DATA_DIR.glob("*.bigWig"))[0]

print("=" * 60)
print("Opening")
print(bigwig_file.name)
print("=" * 60)

# Open file
bw = pybigtools.open(str(bigwig_file))

# Chromosomes and coverage
chroms = bw.chroms()
chrom_num = []
coverage = []
#print(chroms)
for keys in chroms.keys():
    for num in [1, 10, "X", "Y"]:
        if f"chr{num}" == keys:
            print(f"chr{num}")
            chr_values = bw.values(chrom = f"chr{num}",
                                   fillna = 0)
            print(f"{chr_values}")
            not_0 = 0
            for a in chr_values:
                if a > 0:
                    not_0 += 1
            coverage.append(not_0/len(chr_values) * 100)
            chrom_num.append(str(num))
#print(coverage)
#print(chrom_num)
plt.bar(chrom_num, coverage)
plt.title("Coverage per chromosomes example")
plt.xlabel("Chromosome")
plt.ylabel("Coverage %")
plt.ylim(bottom = 99.98,
         top = 100)
plt.tight_layout()
plt.show()


#HISTOGRAM
## x axis: signal, y axis: frequency
#chr1_values = bw.values("chr1", 0, 248956422, fillna=0)
plt.hist(np.log10(chr1_values + 1), edgecolor = "black")
plt.xlabel("Signal")
plt.ylabel("Frequency")
plt.title("Histogram of signal values")
plt.tight_layout()
plt.show()

#Signal along chromosome CHART
##x axis: genome position
##y axis: signal: chr1_values
genome_position = np.arange(1, 50001)
plt.plot(genome_position, np.log10(chr1_values[:50000] + 1))
plt.xlabel("Genome Position (bp)")
plt.ylabel("Signal")
plt.title("Signal along chromosome")
plt.tight_layout()
plt.show()

values = bw.values("chr1", 0, 100000, fillna=0)
print(f"Min: {min(values)}")
print(f"Max: {max(values)}")
print(f"Mean: {values.mean()}")
print(f"Median: {np.median(values)}")
print(f"Datatype: {type(values)}")
