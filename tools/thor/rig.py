"""Skeleton for the Thor boss. Every legacy bone name is kept; new helper bones are marked NEW."""

# landmarks (model units, sole of the boot at 3.22 like the source model)
SOLE, ANKLE, KNEE, HIP = 3.22, 5.7, 15.9, 26.3
WAIST, CHEST, SHOULDER, NECK, CHIN, CROWN = 29.4, 33.4, 39.7, 42.3, 42.5, 48.4
ARM_X, ELBOW, WRIST = 8.05, 31.3, 24.55
LEG_X = 2.65
TOGA_SIDE = -1         # robe over his left shoulder (-X), as in the source model and reference art

# right-side bone -> left-side bone
LEFT = {
    'right arm': 'left_arm5', 'bone16': 'bone162', 'bone14': 'bone163', 'bone4': 'bone164', 'bone5': 'bone165',
    'bone24': 'bone166', 'bone15': 'bone167', 'bone2': 'bone168', 'bone43': 'bone169', 'bone23': 'bone170',
    'bone41': 'bone171', 'bone42': 'bone172', 'bone26': 'bone173', 'bone27': 'bone174', 'bone28': 'bone175',
    'bone29': 'bone176', 'bone34': 'bone177', 'bone30': 'bone178', 'bone31': 'bone179', 'bone32': 'bone180',
    'bone33': 'bone181', 'bone35': 'bone182', 'bone36': 'bone183', 'bone37': 'bone184', 'bone38': 'bone185',
    'bone39': 'bone186', 'bone40': 'bone187', 'RightThumb': 'LeftThumb6',
    'hip_right': 'hip_left2', 'rightLeg': 'leftLeg2', 'RightFoot': 'LeftFoot2',
    'body': 'body7', 'body2': 'body8', 'body5': 'body3',
    'Eye_r': 'Eye_l', 'RightEyelidBase2': 'LeftEyelidBase2', 'new_eye_iris_1': 'new_eye_iris_2',
    'RightEyelidBase3': 'LeftEyelidBase4',
}
FINGERS = ['Index', 'Middle', 'Ring', 'Pinky']
for f in FINGERS:
    for k in (1, 2):
        LEFT[f'Right{f}{k}'] = f'Left{f}{k}'
LEFT['RightThumbBase'] = 'LeftThumbBase'
LEFT['RightThumbTip'] = 'LeftThumbTip'

FINGER_Z = {'Index': -0.86, 'Middle': -0.27, 'Ring': 0.33, 'Pinky': 0.90}
HAND_Y = 21.95        # knuckle line


def build_rig(B):
    b = B.bone
    B.names = LEFT
    b('root', None, (0, -3.4, 0))
    # hips and legs
    b('pelvis_connection', 'root', (0, 26.0, 0.2))
    # waist cloth
    b('tor_sash_mid2', 'root', (0, 28.8, 0.2))
    b('thor_sash2', 'tor_sash_mid2', (0, 28.8, 0.2))
    b('thor_belt', 'thor_sash2', (0, 28.8, 0.2))
    b('belt_inner_lining', 'thor_belt', (0, 28.8, 0.2))
    b('tor_tunic_gathered_waist', 'thor_sash2', (0, 30.4, 0.2))
    b('tunic_soft_waist_71', 'tor_tunic_gathered_waist', (0, 27.3, 0.2))
    b('tor_skirt', 'tor_sash_mid2', (0, 27.5, 0.2))
    b('tor_sash_mid3', 'root', (0, 28.8, 0.2))
    b('thor_sash3', 'tor_sash_mid3', (0, 28.8, 0.2))
    b('thor_belt2', 'thor_sash3', (0, 28.8, -3.0))
    b('belt_inner_lining2', 'thor_belt2', (0, 28.8, -3.0))
    b('tor_tunic_gathered_waist2', 'thor_sash3', (0, 30.4, 0.2))
    b('tunic_soft_waist_2', 'tor_tunic_gathered_waist2', (0, 30.4, 0.2))
    # spine (NEW torso + chest so the upper body can bend and twist as one)
    b('torso', 'root', (0, WAIST, 0.3))
    b('chest', 'torso', (0, CHEST, 0.5))
    t = TOGA_SIDE
    b('tor_sash_mid', 'chest', (4.8 * t, 41.0, 0.3))
    b('sash_mid', 'tor_sash_mid', (4.8 * t, 41.0, 0.3))
    b('sash_mid_cloth_04', 'sash_mid', (4.6 * t, 40.4, -2.4))
    b('sash_mid_cloth_36', 'sash_mid', (4.4 * t, 40.2, 2.6))
    # head
    b('head', 'chest', (0, NECK, 0.5))
    b('LeftEyelidBase', 'head', (0, 45.9, -2.6))                  # nose
    b('RightEyelidBase3', 'LeftEyelidBase', (0.45, 46.15, -2.6))  # right brow
    b('Eye_r', 'head', (1.02, 45.5, -2.4))
    b('RightEyelidBase2', 'Eye_r', (1.02, 45.5, -2.4))
    b('new_eye_iris_1', 'RightEyelidBase2', (0.98, 45.5, -2.5))
    b('head3', 'head', (0, 44.0, -1.0))                           # lower face
    b('RightEyelidBase4', 'head3', (0, 46.15, -2.6))
    b('LeftEyelidBase4', 'RightEyelidBase4', (-0.45, 46.15, -2.6))  # left brow
    b('thor_mouth', 'head3', (0, 43.9, -2.3))
    b('head2', 'head', (0, NECK, 0.5))
    b('Eye_l', 'head2', (-1.02, 45.5, -2.4))
    b('LeftEyelidBase2', 'Eye_l', (-1.02, 45.5, -2.4))
    b('new_eye_iris_2', 'LeftEyelidBase2', (-0.98, 45.5, -2.5))
    for side in (1, -1):
        B.side = side
        b('body', 'torso', (0, WAIST, 0.3))
        b('body5', 'body', (3.6, 30.8, 0.2))
        b('body2', 'chest', (0, CHEST, 0.5))
        # arm: shoulder -> elbow -> wrist -> fingers
        b('right arm', 'chest', (7.6, SHOULDER, 0.2), (0, 0, 7.5))
        b('bone16', 'right arm', (7.6, SHOULDER, 0.2))
        b('bone14', 'bone16', (ARM_X, 39.0, 0.2))
        b('bone4', 'bone14', (8.2, 38.6, 1.6))
        b('bone5', 'bone14', (9.2, 37.2, 0.2))
        b('bone24', 'bone16', (ARM_X, 38.2, 0.2))
        b('bone15', 'bone24', (ARM_X, 38.2, 0.2))
        b('bone2', 'bone15', (ARM_X, 35.4, -1.3))
        b('bone43', 'bone24', (ARM_X, ELBOW, 0.35), (10, 0, 0))
        b('bone23', 'bone43', (ARM_X, ELBOW, 0.35))
        b('bone34', 'bone23', (ARM_X, 30.0, 0.3))
        for n in ('bone30', 'bone31', 'bone32', 'bone33'):
            b(n, 'bone34', (ARM_X, 30.0, 0.3))
        for n in ('bone26', 'bone27', 'bone28', 'bone29'):
            b(n, 'bone23', (ARM_X, 27.2, 0.3))
        for p, c in (('bone35', 'bone36'), ('bone37', 'bone38'), ('bone39', 'bone40'), ('bone41', 'bone42')):
            b(p, 'bone23', (ARM_X, 25.0, 0.3))
            b(c, p, (ARM_X, 25.0, 0.3))
        b('RightThumb', 'bone23', (ARM_X, WRIST, 0.3))            # whole hand (legacy name)
        for f in FINGERS:                                          # NEW finger joints
            b(f'Right{f}1', 'RightThumb', (ARM_X + 0.05, HAND_Y, FINGER_Z[f]), (0, 0, -10))
        b('RightThumbBase', 'RightThumb', (ARM_X - 0.2, 23.75, -0.95), (32, 0, -8))
        # leg: hip -> knee -> ankle
        b('hip_right', 'pelvis_connection', (LEG_X - 0.1, HIP, 0.1))
        b('rightLeg', 'hip_right', (LEG_X, KNEE, -0.1))
        b('RightFoot', 'rightLeg', (LEG_X, ANKLE, 0.35))
    B.side = 1
