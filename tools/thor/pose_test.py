"""Pose test: rotate bones about their pivots (what an animation does) and render."""
import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import Model, sheet
d = json.load(open(sys.argv[1]))
pose = {
    'torso': (0, -12, 0), 'chest': (6, -10, 0), 'head': (-8, 28, 0),
    'right arm': (-70, 0, 35), 'bone43': (-65, 0, 0), 'RightThumb': (0, 0, -15),
    'RightIndex1': (0, 0, -55), 'RightMiddle1': (0, 0, -55), 'RightRing1': (0, 0, -55), 'RightPinky1': (0, 0, -55),
    'RightIndex2': (0, 0, -60), 'RightMiddle2': (0, 0, -60), 'RightRing2': (0, 0, -60), 'RightPinky2': (0, 0, -60),
    'left_arm5': (25, 0, -12), 'bone169': (-25, 0, 0),
    'hip_right': (-38, 0, 0), 'rightLeg': (48, 0, 0), 'RightFoot': (-8, 0, 0),
    'hip_left2': (18, 0, 0), 'leftLeg2': (12, 0, 0),
    'hair_long': (14, 0, 0), 'hair21': (8, 0, 0), 'hair22': (8, 0, 0), 'hair27': (8, 0, 0), 'hair24': (8, 0, 0), 'hair16': (8, 0, 0), 'hair12': (8, 0, 0),
    'tor_front_flap_1': (-30, 0, 0), 'tor_skirt02': (-22, 0, 0), 'tor_skirt02g': (-26, 0, 0), 'tor_skirt03': (-14, 0, 0),
}
found = set()
for g in d['groups']:
    if g['name'] in pose:
        g['rotation'] = [a + b for a, b in zip(g['rotation'], pose[g['name']])]
        found.add(g['name'])
print('missing', sorted(set(pose) - found))
m = Model(data=d)
sheet(m.quads(), m.texture(), [(20, 8), (110, 5), (200, 8), (300, 8)], size=(430, 760)).save(sys.argv[2])
