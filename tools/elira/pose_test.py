"""Pose test: rotate bones about their pivots, as an animation would, and render the result.

    python3 tools/elira/pose_test.py elira/elira.geo.json elira/elira.png out.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from geo import GeoModel
from render import sheet

POSE = {   # Blockbench degrees added to the rest rotation
    'waist': (0, -8, 0), 'chest': (4, -10, 0), 'neck': (0, 8, 0), 'head': (-6, 22, 4),
    'arm_r': (-62, 0, 24), 'forearm_r': (-58, 0, 0), 'hand_r': (0, 0, -12),
    'finger1_1_r': (0, 0, -40), 'finger2_1_r': (0, 0, -40), 'finger3_1_r': (0, 0, -40), 'finger4_1_r': (0, 0, -40),
    'finger1_2_r': (0, 0, -45), 'finger2_2_r': (0, 0, -45), 'finger3_2_r': (0, 0, -45), 'finger4_2_r': (0, 0, -45),
    'arm_l': (22, 0, -14), 'forearm_l': (-24, 0, 0),
    'leg_l': (-36, 0, 0), 'shin_l': (44, 0, 0), 'foot_l': (-6, 0, 0), 'leg_r': (14, 0, 0), 'shin_r': (10, 0, 0),
    'skirt_fl': (-26, 0, 0), 'skirt_fl_lo': (-10, 0, 0), 'skirt_m': (-18, 0, 0), 'skirt_m_lo': (-8, 0, 0),
    'skirt_fr': (10, 0, 0), 'skirt_back': (12, 0, 0), 'skirt_back_lo': (8, 0, 0), 'sash': (0, 0, 8), 'sash_lo': (0, 0, 8),
    'hair_back': (10, 0, 0), 'hair_back_lo': (10, 0, 0), 'hair_wing_r': (4, 0, 8), 'hair_wing_r_lo': (0, 0, 8),
    'hair_wing_l': (4, 0, -6), 'lock_r': (-6, 0, 4), 'lock_r_lo': (-8, 0, 0), 'lock_l': (-4, 0, -3),
}


def main():
    src, tex, out = sys.argv[1:4]
    m = GeoModel(src, pose=POSE)
    missing = sorted(set(POSE) - set(m.bones))
    assert not missing, missing
    sheet(m.quads(), Image.open(tex).convert('RGBA'), [(20, 8), (110, 5), (200, 8), (300, 8)], size=(430, 760)).save(out)


if __name__ == '__main__':
    main()
