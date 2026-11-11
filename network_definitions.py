"""
Network definitions on the AAL-116 atlas (1-based AAL indices, converted to 0-based below).

AAL has no network labels. The groupings below are an AAL-based approximation chosen by the
author and are meant to be pre-specified, not treated as ground truth. Published DMN/FPN/salience
mappings onto AAL differ between studies. Anatomical labels follow the standard AAL ordering
(odd = left, even = right).
"""
def _z(idx):
    return [i - 1 for i in idx]

# Default mode: medial superior frontal (23,24), medial orbitofrontal (25,26), posterior cingulate (35,36),
# hippocampus (37,38), parahippocampal (39,40), angular (65,66), precuneus (67,68), middle temporal (85,86)
DMN = _z([23, 24, 25, 26, 35, 36, 37, 38, 39, 40, 65, 66, 67, 68, 85, 86])
# Frontoparietal control (approximation): dorsolateral superior frontal (3,4), middle frontal (7,8),
# inferior frontal triangularis (13,14), inferior parietal (61,62)
FPN = _z([3, 4, 7, 8, 13, 14, 61, 62])
# Salience (approximation): insula (29,30), anterior cingulate (31,32), mid cingulate (33,34)
SAL = _z([29, 30, 31, 32, 33, 34])
# Amygdala (41,42)
AMY = _z([41, 42])
# vmPFC proxy: orbital superior frontal (5,6), orbital middle frontal (9,10), rectus (27,28)
VMPFC = _z([5, 6, 9, 10, 27, 28])
# Cerebellum and vermis (91-116)
CER = _z(list(range(91, 117)))

BLOCKS = {  # name -> (set A, set B or None for within-set)
    "DMN_within": (DMN, None),
    "FPN_within": (FPN, None),
    "SAL_within": (SAL, None),
    "DMN_FPN": (DMN, FPN),
    "AMY_vmPFC": (AMY, VMPFC),
    "CER_FPN": (CER, FPN),
}
