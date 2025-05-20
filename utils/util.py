import math
import os
import csv
import random
import torch
from torch.utils import data
import numpy as np
from dateutil import parser

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK','MAS',
    ]

aa2num= {x:i for i,x in enumerate(num2aa)}

aa1to3 =  \
    {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS',
     'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN',
     'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP',
     'A': 'ALA', 'V':'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET',
     'X': 'UNK'}

# minimal sc atom representation (Nx8)
aa2short=[
    (" N  "," CA "," C  "," CB ",  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," CB "," CG "," CD "," NE "," CZ "), # arg
    (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), # asn
    (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), # asp
    (" N  "," CA "," C  "," CB "," SG ",  None,  None,  None), # cys
    (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), # gln
    (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), # glu
    (" N  "," CA "," C  ",  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," CB "," CG "," ND1",  None,  None), # his
    (" N  "," CA "," C  "," CB "," CG1"," CD1",  None,  None), # ile
    (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), # leu
    (" N  "," CA "," C  "," CB "," CG "," CD "," CE "," NZ "), # lys
    (" N  "," CA "," C  "," CB "," CG "," SD "," CE ",  None), # met
    (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), # phe
    (" N  "," CA "," C  "," CB "," CG "," CD ",  None,  None), # pro
    (" N  "," CA "," C  "," CB "," OG ",  None,  None,  None), # ser
    (" N  "," CA "," C  "," CB "," OG1",  None,  None,  None), # thr
    (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), # trp
    (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), # tyr
    (" N  "," CA "," C  "," CB "," CG1",  None,  None,  None), # val
    (" N  "," CA "," C  "," CB ",  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," CB ",  None,  None,  None,  None), # mask
]

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # mask
]

# build the "alternate" sc mapping
aa2longalt=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH2"," NH1",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD2"," OD1",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE2"," OE1",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG2"," CG1",  None,  None,  None,  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # mask
]
max_n_atoms = len(aa2long[0])
aa2natom = [14 - atoms.count(None) for atoms in aa2long]

aa2tip = [
        " CB ", # ala
        " CZ ", # arg
        " ND2", # asn
        " CG ", # asp
        " SG ", # cys
        " NE2", # gln
        " CD ", # glu
        " CA ", # gly
        " NE2", # his
        " CD1", # ile
        " CG ", # leu
        " NZ ", # lys
        " SD ", # met
        " CZ ", # phe
        " CG ", # pro
        " OG ", # ser
        " OG1", # thr
        " CH2", # trp
        " OH ", # tyr
        " CB ", # val
        " CB ", # unknown (gap etc)
        " CB " # masked
        ]


torsions=[
    [ None, None, None, None ],  # ala
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," NE "], [" CG "," CD "," NE "," CZ "] ],  # arg
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," OD1"], None, None ],  # asn
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," OD1"], None, None ],  # asp
    [ [" N  "," CA "," CB "," SG "], None, None, None ],  # cys
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," OE1"], None ],  # gln
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," OE1"], None ],  # glu
    [ None, None, None, None ],  # gly
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," ND1"], None, None ],  # his
    [ [" N  "," CA "," CB "," CG1"], [" CA "," CB "," CG1"," CD1"], None, None ],  # ile
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # leu
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," CE "], [" CG "," CD "," CE "," NZ "] ],  # lys
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," SD "], [" CB "," CG "," SD "," CE "], None ],  # met
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # phe
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], None, None ],  # pro
    [ [" N  "," CA "," CB "," OG "], None, None, None ],  # ser
    [ [" N  "," CA "," CB "," OG1"], None, None, None ],  # thr
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # trp
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # tyr
    [ [" N  "," CA "," CB "," CG1"], None, None, None ],  # val
    [ None, None, None, None ],  # unk
    [ None, None, None, None ],  # mask
]

# ideal N, CA, C initial coordinates
init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
init_CA = torch.zeros_like(init_N)
init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
INIT_CRDS = torch.full((14, 3), np.nan)
INIT_CRDS[:3] = torch.stack((init_N, init_CA, init_C), dim=0) # (3,3)

norm_N = init_N / (torch.norm(init_N, dim=-1, keepdim=True) + 1e-5)
norm_C = init_C / (torch.norm(init_C, dim=-1, keepdim=True) + 1e-5)
cos_ideal_NCAC = torch.sum(norm_N*norm_C, dim=-1) # cosine of ideal N-CA-C bond angle

#fd Rosetta vals
ideal_coords = [
    [ # 0 ala
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
    ],
    [ # 1 arg
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5042,-0.7698,-1.2118)],
        [' CG ', 4, (0.6396,1.3794, 0.000)],
        [' CD ', 5, (0.5492,1.3801, 0.000)],
        [' NE ', 6, (0.5423,1.3491, 0.000)],
        [' NH1', 7, (0.2012,2.2965, 0.000)],
        [' NH2', 7, (2.0824,1.0030, 0.000)],
        [' CZ ', 7, (0.7650,1.1090, 0.000)],
    ],
    [ # 2 asn
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5341,-0.7799,-1.1874)],
        [' CG ', 4, (0.5778,1.3881, 0.000)],
        [' ND2', 5, (0.5839,-1.1711, 0.000)],
        [' OD1', 5, (0.6331,1.0620, 0.000)],
    ],
    [ # 3 asp
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5162,-0.7757,-1.2144)],
        [' CG ', 4, (0.5926,1.4028, 0.000)],
        [' OD1', 5, (0.5746,1.0629, 0.000)],
        [' OD2', 5, (0.5738,-1.0627, 0.000)],
    ],
    [ # 4 cys
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5046,-0.7727,-1.2189)],
        [' SG ', 4, (0.7386,1.6511, 0.000)],
    ],
    [ # 5 gln
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5226,-0.7776,-1.2109)],
        [' CG ', 4, (0.6225,1.3857, 0.000)],
        [' CD ', 5, (0.5788,1.4021, 0.000)],
        [' NE2', 6, (0.5908,-1.1895, 0.000)],
        [' OE1', 6, (0.6347,1.0584, 0.000)],
    ],
    [ # 6 glu
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5197,-0.7737,-1.2137)],
        [' CG ', 4, (0.6287,1.3862, 0.000)],
        [' CD ', 5, (0.5850,1.3849, 0.000)],
        [' OE1', 6, (0.5752,1.0618, 0.000)],
        [' OE2', 6, (0.5741,-1.0635, 0.000)],
    ],
    [ # 7 gly
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
    ],
    [ # 8 his
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5163,-0.7809,-1.2129)],
        [' CG ', 4, (0.6016,1.3710, 0.000)],
        [' CD2', 5, (0.8918,-1.0184, 0.000)],
        [' ND1', 5, (0.7437,1.1615, 0.000)],
        [' CE1', 5, (2.0299,0.8564, 0.000)],
        [' NE2', 5, (2.1458,-0.4589, 0.000)],
    ],
    [ # 9 ile
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5140,-0.7885,-1.2184)],
        [' CG1', 4, (0.5339,1.4348,0.000)],
        [' CG2', 4, (0.5319,-0.7693,-1.1994)],
        [' CD1', 5, (0.6106,1.3829, 0.000)],
    ],
    [ # 10 leu
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.525, -0.000, -0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5175,-0.7692,-1.2220)],
        [' CG ', 4, (0.6652,1.3823, 0.000)],
        [' CD1', 5, (0.5083,1.4353, 0.000)],
        [' CD2', 5, (0.5079,-0.7600,1.2163)],
    ],
    [ # 11 lys
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5259,-0.7785,-1.2069)],
        [' CG ', 4, (0.6291,1.3869, 0.000)],
        [' CD ', 5, (0.5526,1.4174, 0.000)],
        [' CE ', 6, (0.5544,1.4170, 0.000)],
        [' NZ ', 7, (0.5566,1.3801, 0.000)],
    ],
    [ # 12 met
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5331,-0.7727,-1.2048)],
        [' CG ', 4, (0.6298,1.3858,0.000)],
        [' SD ', 5, (0.6953,1.6645,0.000)],
        [' CE ', 6, (0.3383,1.7581,0.000)],
    ],
    [ # 13 phe
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5150,-0.7729,-1.2156)],
        [' CG ', 4, (0.6060,1.3746, 0.000)],
        [' CD1', 5, (0.7078,1.1928, 0.000)],
        [' CD2', 5, (0.7084,-1.1920, 0.000)],
        [' CE1', 5, (2.0900,1.1940, 0.000)],
        [' CE2', 5, (2.0897,-1.1939, 0.000)],
        [' CZ ', 5, (2.7809, 0.000, 0.000)],
    ],
    [ # 14 pro
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5649,-0.5888,-1.2966)],
        [' CG ', 4, (0.3657,1.4451,0.0000)],
        [' CD ', 5, (0.3744,1.4582, 0.0)],
    ],
    [ # 15 ser
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5146,-0.7595,-1.2073)],
        [' OG ', 4, (0.5021,1.3081, 0.000)],
    ],
    [ # 16 thr
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5172,-0.7952,-1.2130)],
        [' CG2', 4, (0.5334,-0.7239,-1.2267)],
        [' OG1', 4, (0.4804,1.3506,0.000)],
    ],
    [ # 17 trp
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5136,-0.7712,-1.2173)],
        [' CG ', 4, (0.5984,1.3741, 0.000)],
        [' CD1', 5, (0.8151,1.0921, 0.000)],
        [' CD2', 5, (0.8753,-1.1538, 0.000)],
        [' CE2', 5, (2.1865,-0.6707, 0.000)],
        [' CE3', 5, (0.6541,-2.5366, 0.000)],
        [' NE1', 5, (2.1309,0.7003, 0.000)],
        [' CH2', 5, (3.0315,-2.8930, 0.000)],
        [' CZ2', 5, (3.2813,-1.5205, 0.000)],
        [' CZ3', 5, (1.7521,-3.3888, 0.000)],
    ],
    [ # 18 tyr
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5305,-0.7799,-1.2051)],
        [' CG ', 4, (0.6104,1.3840, 0.000)],
        [' CD1', 5, (0.6936,1.2013, 0.000)],
        [' CD2', 5, (0.6934,-1.2011, 0.000)],
        [' CE1', 5, (2.0751,1.2013, 0.000)],
        [' CE2', 5, (2.0748,-1.2011, 0.000)],
        [' OH ', 5, (4.1408, 0.000, 0.000)],
        [' CZ ', 5, (2.7648, 0.000, 0.000)],
    ],
    [ # 19 val
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5105,-0.7712,-1.2317)],
        [' CG1', 4, (0.5326,1.4252, 0.000)],
        [' CG2', 4, (0.5177,-0.7693,1.2057)],
    ],
    [ # 20 unk
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
    ],
    [ # 21 mask
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
    ],
]

# van der Waals radii (from wikipedia)
vdw_radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
vdw_radii_per_aa = torch.full((22,14), 0.0)
for i in range(22):
    for i_atm, atm_name in enumerate(aa2long[i]):
        if atm_name == None:
            break
        element = atm_name.strip()[0]
        vdw_radii_per_aa[i,i_atm] = vdw_radii[element]

# define nonbonded pair within residue
nonbonded_pair = {}
#ALA
#ARG
nonbonded_pair['ARG'] = {}
nonbonded_pair['ARG'][' N  '] = [' CD ', ' NE ', ' CZ ', ' NH1', ' NH2'] 
nonbonded_pair['ARG'][' CA '] = [' NE ', ' CZ ', ' NH1', ' NH2'] 
nonbonded_pair['ARG'][' C  '] = [' CD ', ' NE ', ' CZ ', ' NH1', ' NH2'] 
nonbonded_pair['ARG'][' O  '] = [' CG ', ' CD ', ' NE ', ' CZ ', ' NH1', ' NH2'] 
nonbonded_pair['ARG'][' CB '] = [' CZ ', ' NH1', ' NH2']
nonbonded_pair['ARG'][' CG' ] = [' NH1', ' NH2']
#ASN
nonbonded_pair['ASN'] = {}
nonbonded_pair['ASN'][' N  '] = [' OD1', ' ND2'] 
nonbonded_pair['ASN'][' C  '] = [' OD1', ' ND2'] 
nonbonded_pair['ASN'][' O  '] = [' CG ', ' OD1', ' ND2'] 
#ASP
nonbonded_pair['ASP'] = {}
nonbonded_pair['ASP'][' N  '] = [' OD1', ' OD2'] 
nonbonded_pair['ASP'][' C  '] = [' OD1', ' OD2'] 
nonbonded_pair['ASP'][' O  '] = [' CG ', ' OD1', ' OD2'] 
#CYS
nonbonded_pair['CYS'] = {}
nonbonded_pair['CYS'][' O  '] = [' SG '] 
#GLN
nonbonded_pair['GLN'] = {}
nonbonded_pair['GLN'][' N  '] = [' CD ', ' OE1', ' NE2'] 
nonbonded_pair['GLN'][' CA '] = [' OE1', ' NE2']
nonbonded_pair['GLN'][' C  '] = [' CD ', ' OE1', ' NE2'] 
nonbonded_pair['GLN'][' O  '] = [' CG ', ' CD ', ' OE1', ' NE2'] 
#GLU
nonbonded_pair['GLU'] = {}
nonbonded_pair['GLU'][' N  '] = [' CD ', ' OE1', ' OE2'] 
nonbonded_pair['GLU'][' CA '] = [' OE1', ' OE2']
nonbonded_pair['GLU'][' C  '] = [' CD ', ' OE1', ' OE2'] 
nonbonded_pair['GLU'][' O  '] = [' CG ', ' CD ', ' OE1', ' OE2'] 
#GLY
#HIS
nonbonded_pair['HIS'] = {}
nonbonded_pair['HIS'][' N  '] = [' ND1', ' CD2', ' CE1', ' NE2'] 
nonbonded_pair['HIS'][' CA '] = [' CE1', ' NE2']
nonbonded_pair['HIS'][' C  '] = [' ND1', ' CD2', ' CE1', ' NE2'] 
nonbonded_pair['HIS'][' O  '] = [' CG ', ' ND1', ' CD2', ' CE1', ' NE2']
#ILE
nonbonded_pair['ILE'] = {}
nonbonded_pair['ILE'][' N  '] = [' CD1'] 
nonbonded_pair['ILE'][' C  '] = [' CD1'] 
nonbonded_pair['ILE'][' O  '] = [' CG1', ' CG2', ' CD1'] 
#LEU
nonbonded_pair['LEU'] = {}
nonbonded_pair['LEU'][' N  '] = [' CD1', ' CD2'] 
nonbonded_pair['LEU'][' C  '] = [' CD1', ' CD2'] 
nonbonded_pair['LEU'][' O  '] = [' CG ', ' CD1', ' CD2'] 
#LYS
nonbonded_pair['LYS'] = {}
nonbonded_pair['LYS'][' N  '] = [' CD ', ' CE ', ' NZ '] 
nonbonded_pair['LYS'][' CA '] = [' CE ', ' NZ '] 
nonbonded_pair['LYS'][' C  '] = [' CD ', ' CE ', ' NZ '] 
nonbonded_pair['LYS'][' O  '] = [' CG ', ' CD ', ' CE ', ' NZ '] 
nonbonded_pair['LYS'][' CB '] = [' NZ ']
#MET
nonbonded_pair['MET'] = {}
nonbonded_pair['MET'][' N  '] = [' SD ', ' CE '] 
nonbonded_pair['MET'][' CA '] = [' CE '] 
nonbonded_pair['MET'][' C  '] = [' SD ', ' CE '] 
nonbonded_pair['MET'][' O  '] = [' CG ', ' SD ', ' CE '] 
#PHE
nonbonded_pair['PHE'] = {}
nonbonded_pair['PHE'][' N  '] = [' CD1', ' CD2', ' CE1', ' CE2', ' CZ ']
nonbonded_pair['PHE'][' CA '] = [' CE1', ' CE2', ' CZ ']
nonbonded_pair['PHE'][' C  '] = [' CD1', ' CD2', ' CE1', ' CE2', ' CZ ']
nonbonded_pair['PHE'][' O  '] = [' CG ', ' CD1', ' CD2', ' CE1', ' CE2', ' CZ ']
nonbonded_pair['PHE'][' CB '] = [' CZ ']
#PRO
nonbonded_pair['PRO'] = {}
nonbonded_pair['PRO'][' O  '] = [' CG ', ' CD ']
#SER
nonbonded_pair['SER'] = {}
nonbonded_pair['SER'][' O  '] = [ ' OG ']
#THR
nonbonded_pair['THR'] = {}
nonbonded_pair['THR'][' O  '] = [' OG1', ' CG2']
#TRP
nonbonded_pair['TRP'] = {}
nonbonded_pair['TRP'][' N  '] = [' CD1', ' CD2', ' NE1', ' CE2', ' CE3', ' CZ2', ' CZ3', ' CH2']
nonbonded_pair['TRP'][' CA '] = [' NE1', ' CE2', ' CE3', ' CZ2', ' CZ3', ' CH2']
nonbonded_pair['TRP'][' C  '] = [' CD1', ' CD2', ' NE1', ' CE2', ' CE3', ' CZ2', ' CZ3', ' CH2']
nonbonded_pair['TRP'][' O  '] = [' CG ', ' CD1', ' CD2', ' NE1', ' CE2', ' CE3', ' CZ2', ' CZ3', ' CH2']
nonbonded_pair['TRP'][' CB '] = [' CZ2', ' CZ3', ' CH2']
nonbonded_pair['TRP'][' CG '] = [' CH2']
#TYR
nonbonded_pair['TYR'] = {}
nonbonded_pair['TYR'][' N  '] = [' CD1', ' CD2', ' CE1', ' CE2', ' CZ ', ' OH ']
nonbonded_pair['TYR'][' CA '] = [' CE1', ' CE2', ' CZ ', ' OH ']
nonbonded_pair['TYR'][' C  '] = [' CD1', ' CD2', ' CE1', ' CE2', ' CZ ', ' OH ']
nonbonded_pair['TYR'][' O  '] = [' CG ', ' CD1', ' CD2', ' CE1', ' CE2', ' CZ ', ' OH ']
nonbonded_pair['TYR'][' CB '] = [' CZ ', ' OH ']
nonbonded_pair['TYR'][' CG '] = [' OH ']
#VAL
nonbonded_pair['VAL'] = {}
nonbonded_pair['VAL'][' O  '] = [' CG1', ' CG2']
#UNK
#MASK

nonbonded_within_residue = torch.full((22,14,14), False)
for i in range(22):
    res_name = num2aa[i]
    if not res_name in nonbonded_pair:
        continue
    for i_atm, atm_name in enumerate(aa2long[i]):
        if not atm_name in nonbonded_pair[res_name]:
            continue
        for tgt_name in nonbonded_pair[res_name][atm_name]:
            j_atm = aa2long[i].index(tgt_name)
            nonbonded_within_residue[i,i_atm,j_atm] = True

# resolve tip atom indices
tip_indices = torch.full((22,), 0)
for i in range(22):
    tip_atm = aa2tip[i]
    atm_long = aa2long[i]
    tip_indices[i] = atm_long.index(tip_atm)

# resolve torsion indices
torsion_indices = torch.full((22,4,4),0)
torsion_can_flip = torch.full((22,10),False,dtype=torch.bool)
for i in range(22):
    i_l, i_a = aa2long[i], aa2longalt[i]
    for j in range(4):
        if torsions[i][j] is None:
            continue
        for k in range(4):
            a = torsions[i][j][k]
            torsion_indices[i,j,k] = i_l.index(a)
            if (i_l.index(a) != i_a.index(a)):
                torsion_can_flip[i,3+j] = True ##bb tors never flip

# build the mapping from atoms in the full rep (Nx14) to the "alternate" rep
allatom_mask = torch.zeros((22,14), dtype=torch.bool)
long2alt = torch.zeros((22,14), dtype=torch.long)
for i in range(22):
    i_l, i_lalt = aa2long[i],  aa2longalt[i]
    for j,a in enumerate(i_l):
        if (a is None):
            long2alt[i,j] = j
        else:
            long2alt[i,j] = i_lalt.index(a)
            allatom_mask[i,j] = True

def th_ang_v(ab,bc,eps:float=1e-8):
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)
    ab, bc = th_N(ab),th_N(bc)
    cos_angle = torch.clamp( (ab*bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih_v(ab,bc,cd):
    def th_cross(a,b):
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
    n1 = th_N( th_cross(ab,bc) )
    n2 = th_N( th_cross(bc,cd) )
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)
    cos_angle = (n1*n2).sum(-1)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih(a,b,c,d):
    return th_dih_v(a-b,b-c,c-d)

# More complicated version splits error in CA-N and CA-C (giving more accurate CB position)
# It returns the rigid transformation from local frame to global frame
def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    #N, Ca, C - [B,L, 3]
    #R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B,L = N.shape[:2]
    
    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix
    
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.sum(e1*v2, dim=-1) # cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
        Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel
    
        R = torch.einsum('blij,bljk->blik', R,Rp)

    return R, Ca

def get_tor_mask(seq, torsion_indices, atom_mask=None):
    B,L = seq.shape[:2]
    tors_mask = torch.ones((B,L,10), dtype=torch.bool, device=seq.device)
    tors_mask[...,3:7] = torsion_indices[seq,:,-1] > 0
    tors_mask[:,0,1] = False
    tors_mask[:,-1,0] = False

    # mask for additional angles
    tors_mask[:,:,7] = seq!=aa2num['GLY']
    tors_mask[:,:,8] = seq!=aa2num['GLY']
    tors_mask[:,:,9] = torch.logical_and( seq!=aa2num['GLY'], seq!=aa2num['ALA'] )
    tors_mask[:,:,9] = torch.logical_and( tors_mask[:,:,9], seq!=aa2num['UNK'] )
    tors_mask[:,:,9] = torch.logical_and( tors_mask[:,:,9], seq!=aa2num['MAS'] )

    if atom_mask == None:
        return tors_mask

    # masking missing atoms
    ti0 = torch.gather(atom_mask,2,torsion_indices[seq,:,0]) # first atoms to define chi angles (B, L, n_chi)
    ti1 = torch.gather(atom_mask,2,torsion_indices[seq,:,1])
    ti2 = torch.gather(atom_mask,2,torsion_indices[seq,:,2])
    ti3 = torch.gather(atom_mask,2,torsion_indices[seq,:,3])
    
    mask = torch.stack((ti0,ti1,ti2,ti3), dim=-1)

    tors_mask[...,3:7] = torch.logical_and(tors_mask[...,3:7], mask.all(dim=-1))

    return tors_mask

def get_torsions(xyz_in, seq, torsion_indices, torsion_can_flip, ref_angles, atom_mask=None):
    B,L = xyz_in.shape[:2]
    
    tors_mask = get_tor_mask(seq, torsion_indices, atom_mask=atom_mask)

    # idealize given xyz coordinates before computing torsion angles
    xyz = xyz_in.clone()
    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:])
    Nideal = torch.tensor([-0.5272, 1.3593, 0.000], device=xyz_in.device)
    Cideal = torch.tensor([1.5233, 0.000, 0.000], device=xyz_in.device)
    xyz[...,0,:] = torch.einsum('brij,j->bri', Rs, Nideal) + Ts
    xyz[...,2,:] = torch.einsum('brij,j->bri', Rs, Cideal) + Ts

    torsions = torch.zeros( (B,L,10,2), device=xyz.device )
    # omega
    torsions[:,:-1,0,:] = th_dih(xyz[:,:-1,1,:],xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:])
    # phi
    torsions[:,1:,1,:] = th_dih(xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:],xyz[:,1:,2,:])
    # psi
    torsions[:,:,2,:] = -1 * th_dih(xyz[:,:,0,:],xyz[:,:,1,:],xyz[:,:,2,:],xyz[:,:,3,:])

    # chis
    ti0 = torch.gather(xyz,2,torsion_indices[seq,:,0,None].repeat(1,1,1,3))
    ti1 = torch.gather(xyz,2,torsion_indices[seq,:,1,None].repeat(1,1,1,3))
    ti2 = torch.gather(xyz,2,torsion_indices[seq,:,2,None].repeat(1,1,1,3))
    ti3 = torch.gather(xyz,2,torsion_indices[seq,:,3,None].repeat(1,1,1,3))
    torsions[:,:,3:7,:] = th_dih(ti0,ti1,ti2,ti3)
    
    # CB bend
    NC = 0.5*( xyz[:,:,0,:3] + xyz[:,:,2,:3] )
    CA = xyz[:,:,1,:3]
    CB = xyz[:,:,4,:3]
    t = th_ang_v(CB-CA,NC-CA)
    t0 = ref_angles[seq][...,0,:]
    torsions[:,:,7,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    # CB twist
    NCCA = NC-CA
    NCp = xyz[:,:,2,:3] - xyz[:,:,0,:3]
    NCpp = NCp - torch.sum(NCp*NCCA, dim=-1, keepdim=True)/ torch.sum(NCCA*NCCA, dim=-1, keepdim=True) * NCCA
    t = th_ang_v(CB-CA,NCpp)
    t0 = ref_angles[seq][...,1,:]
    torsions[:,:,8,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )

    # CG bend
    CG = xyz[:,:,5,:3]
    t = th_ang_v(CG-CB,CA-CB)
    t0 = ref_angles[seq][...,2,:]
    torsions[:,:,9,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )

    # alt chis
    torsions_alt = torsions.clone()
    torsions_alt[torsion_can_flip[seq,:]] *= -1

    return torsions, torsions_alt, tors_mask

def get_tips(xyz, seq):
    B,L = xyz.shape[:2]

    xyz_tips = torch.gather(xyz, 2, tip_indices.to(xyz.device)[seq][:,:,None,None].expand(-1,-1,-1,3)).reshape(B, L, 3)
    if torch.isnan(xyz_tips).any(): # replace NaN tip atom with virtual Cb atom
        # three anchor atoms
        N  = xyz[:,:,0]
        Ca = xyz[:,:,1]
        C  = xyz[:,:,2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    

        xyz_tips = torch.where(torch.isnan(xyz_tips), Cb, xyz_tips)
    return xyz_tips

def make_frame(X, Y):
    Xn = X / torch.linalg.norm(X)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    Z = torch.cross(Xn,Yn)
    Zn =  Z / torch.linalg.norm(Z)

    return torch.stack((Xn,Yn,Zn), dim=-1)

def cross_product_matrix(u):
    B, L = u.shape[:2]
    matrix = torch.zeros((B, L, 3, 3), device=u.device)
    matrix[:,:,0,1] = -u[...,2]
    matrix[:,:,0,2] = u[...,1]
    matrix[:,:,1,0] = u[...,2]
    matrix[:,:,1,2] = -u[...,0]
    matrix[:,:,2,0] = -u[...,1]
    matrix[:,:,2,1] = u[...,0]
    return matrix

# process ideal frames
base_indices = torch.full((22,14),0, dtype=torch.long)
xyzs_in_base_frame = torch.ones((22,14,4))
RTs_by_torsion = torch.eye(4).repeat(22,7,1,1)
reference_angles = torch.ones((22,3,2))

for i in range(22):
    i_l = aa2long[i]
    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i,idx] = base
        xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

    # omega frame
    RTs_by_torsion[i,0,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,0,:3,3] = torch.zeros(3)

    # phi frame
    RTs_by_torsion[i,1,:3,:3] = make_frame(
        xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3],
        torch.tensor([1.,0.,0.])
    )
    RTs_by_torsion[i,1,:3,3] = xyzs_in_base_frame[i,0,:3]

    # psi frame
    RTs_by_torsion[i,2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,2,:3] - xyzs_in_base_frame[i,1,:3],
        xyzs_in_base_frame[i,1,:3] - xyzs_in_base_frame[i,0,:3]
    )
    RTs_by_torsion[i,2,:3,3] = xyzs_in_base_frame[i,2,:3]

    # chi1 frame
    if torsions[i][0] is not None:
        a0,a1,a2 = torsion_indices[i,0,0:3]
        RTs_by_torsion[i,3,:3,:3] = make_frame(
            xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
            xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3],
        )
        RTs_by_torsion[i,3,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # chi2 frame
    for j in range(1,4):
        if torsions[i][j] is not None:
            a2 = torsion_indices[i,j,2]
            RTs_by_torsion[i,3+j,:3,:3] = make_frame( 
                xyzs_in_base_frame[i,a2,:3],
                torch.tensor([-1.,0.,0.]), )
            RTs_by_torsion[i,3+j,:3,3] = xyzs_in_base_frame[i,a2,:3]
            
    # CB/CG angles
    NCr = 0.5*(xyzs_in_base_frame[i,0,:3]+xyzs_in_base_frame[i,2,:3])
    CAr = xyzs_in_base_frame[i,1,:3]
    CBr = xyzs_in_base_frame[i,4,:3]
    CGr = xyzs_in_base_frame[i,5,:3]
    reference_angles[i,0,:]=th_ang_v(CBr-CAr,NCr-CAr)
    NCp = xyzs_in_base_frame[i,2,:3]-xyzs_in_base_frame[i,0,:3]
    NCpp = NCp - torch.dot(NCp,NCr)/ torch.dot(NCr,NCr) * NCr
    reference_angles[i,1,:]=th_ang_v(CBr-CAr,NCpp)
    reference_angles[i,2,:]=th_ang_v(CGr,torch.tensor([-1.,0.,0.]))

def make_deterministic(seed=0):
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

def assert_no_nan(t):
    if torch.any(torch.isnan(t)):
        ic(t)
        ic(t.shape)
        ic(torch.isnan(t).nonzero())
        raise Exception('nans found')

class Timer:
    def __init__(self, name):
        self.name = name
        self.last = None
        self.elapsed = 0
    def tick(self):
        now = timer()
        if self.last is not None:
            self.elapsed = now - self.last
        self.last = now

    def format(self):
        return f'Timer {self.name}: {self.elapsed:.3f}'

    def tick_and_print(self, msg=''):
        self.tick()
        out = ''
        if msg:
            out = msg + ': '
        out += self.format()
        print(out)

def garbage_collect_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    tensors.append(obj)
        except:
            pass
    return tensors

def tensor_memory(t):
    return t.element_size() * t.nelement()

def format_tensors(tensors):
    msgs = []
    for t in tensors:
        msgs.append(f'Size: {tensor_memory(t)}, Memory: {sys.getsizeof(t) / 1024 / 1024 / 1024} Gb')
    return '\n'.join(msgs)

def memory_snapshot():
    msg = ''
    msg += torch.cuda.memory_summary() + '\n'
    msg += 'Garbage collected tensors:\n' + format_tensors(garbage_collect_tensors())
    return msg

# Sets the default cdist compute_mode to 'use_mm_for_euclid_dist' to
# preserve memory contiguity for backpropagation.
def monkey_patch_cdist():
    def wrap(original):
        def safe_cdist(*args, **kwargs):
            return original(*args, **{**{'compute_mode': 'use_mm_for_euclid_dist'}, **kwargs})
        return safe_cdist
    torch.cdist = wrap(torch.cdist)

def assert_all_atoms_present(seq, xyz, idx_pdb=None, msg='', assert_not_too_many=False, assert_all_present=False, verbose=False):
    '''
    Asserts that all atoms given in seq have non-nan positions in xyz.
        Parameters:
            seq: (L) categorical aa
            xyz: (L, 14, 3) atom coordinates
    '''
    assert len(seq) == len(xyz), f'shape mismatch between seq and xyz first dimension: seq: {seq.shape}, xyz: {xyz.shape}'
    if idx_pdb is not None:
        assert len(seq) ==  len(idx_pdb)
    else:
        idx_pdb = list(range(len(seq)))
    too_many = []
    not_all_present = []
    for aa_i, co, i in zip(seq, xyz, idx_pdb):
        n_expected = aa2natom[aa_i]
        nan_idx = torch.logical_or(co.isnan(), co == 0)
        nan_atom_count = torch.sum(~torch.any(nan_idx, dim=1))
        o = f'{msg}| {nan_atom_count}\t/{n_expected} at idx_pdb:{i}, aa:{num2aa[aa_i]}'
        print(o)
        if nan_atom_count > n_expected:
            too_many.append(o)
            if verbose:
                ic(co)
                ic(nan_idx)
                ic(nan_atom_count)
        if nan_atom_count != n_expected:
            not_all_present.append(o)

    for do_assert, errors, prefix in (
            (assert_not_too_many, too_many, 'too many atoms for some residues'),
            (assert_all_present, not_all_present, 'not all atoms present for some residues'),
            ):
        m = f'{prefix}:{len(errors)} / {len(seq)}'
        if do_assert and len(errors) > 0:
            raise Exception(m)
        print(m)

