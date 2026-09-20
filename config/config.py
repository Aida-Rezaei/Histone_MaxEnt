"""
Project configuration
"""
# Genome assembly
GENOME = "hg38"

# Histone modifications
HISTONE_MARKS = [
    "H3K4me3",
    "H3K27me3",
    "H3K4me1",
    "H3K27ac",
    "H3K36me3",
    "H3K9me3",
    "H3K9ac"
]

# Cell types
CELL_TYPES = [
    "H1",
    "GM12878",
    "K562",
    "HepG2",
    "IMR90"
]

# Resolution
BIN_SIZE = 200 # bp
WINDOW_SIZE = 10000 # bp