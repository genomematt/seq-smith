import numpy as np
from seq_align import align

seqa = "AGAGAGAGAG"
seqb = "AGCAGCAGCA"
map_bytes = [0] * 256
for i, c in enumerate("ACGT"):
    map_bytes[ord(c)] = i

score_matrix = np.array(
    [
        [1, -1, -1, -1],
        [-1, 1, -1, -1],
        [-1, -1, 1, -1],
        [-1, -1, -1, 1],
    ],
    dtype=np.int32,
)

gap_open = -1
gap_extend = -1

alignment = align(seqa, seqb, map_bytes, score_matrix, gap_open, gap_extend)
print(f"Alignment score: {alignment.score}")
print(f"Alignment fragments: {alignment.align_frag}")
