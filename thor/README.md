# Thor boss model

Redesign of the Thor boss for the Record of Ragnarok mod, generated from the original Blockbench
project in `source/`.

| File | What it is |
| --- | --- |
| `thor_boss.bbmodel` | The redesigned model, same bone hierarchy as the source project. Use this one. |
| `thor_boss.png` | Its 4096 x 4096 texture (also embedded in the `.bbmodel`). UV space is still 1024 x 1024. |
| `thor_boss_spine_rig.bbmodel` | Same cubes and texture, but head, arms, chest and hair hang from a `torso > chest` spine. Optional. |
| `source/tor_hair_reference_v3.bbmodel` | The original project, unchanged. The generator reads its bones and paints from its texture. |
| `preview/` | Renders: turnaround, close-ups, before/after and a posed rig test. |

## Compatibility with the existing rig

- Every one of the 356 groups in the source project is still present with the same name and UUID,
  including the empty ones, so animations and code that look bones up by name keep resolving.
- `thor_boss.bbmodel` keeps the source parent/child layout: `root` holds head, body, cloth and legs,
  and `right arm` / `left_arm5` stay top-level groups.
- Feet rest on the same ground plane (y = 3.22), the height differs by 0.3 units and the footprint is
  slightly smaller, so the hitbox and render scale do not need to change.
- Pivots moved to the anatomical joints (shoulder, elbow, wrist, hip, knee, ankle, neck). Rotation
  keyframes keep working but will bend at the new joint positions.

New bones, all additive: `torso`, `chest`, `hair_long`, `hair30`, `hair31`, two joints per finger
(`RightIndex1/2` ... `LeftPinky1/2`) and `Right/LeftThumbBase`, `Right/LeftThumbTip`.

The long hair is rigged as six columns of three bones (for example `hair19 > hair21 > hair23`), the
skirt as 21 two-joint panels (`tor_skirt02 > tor_skirt02b`, ...) plus the two front flaps.

## Rebuilding

Needs Python 3 with `numpy` and `pillow`.

```
python3 tools/thor/build.py                                          # thor_boss.bbmodel + .png (about a minute)
python3 tools/thor/build.py --spine-rig --name thor_boss_spine_rig
python3 tools/thor/build.py --fast --out /tmp/thor                   # quick low-res pass while editing shapes
python3 tools/thor/check.py thor/source/tor_hair_reference_v3.bbmodel thor/thor_boss.bbmodel
python3 tools/thor/preview.py thor/thor_boss.bbmodel thor/preview --compare thor/source/tor_hair_reference_v3.bbmodel
python3 tools/thor/pose_test.py thor/thor_boss_spine_rig.bbmodel thor/preview/pose_test.png
```

Shapes live in `tools/thor/{rig,body,head,hair,cloth}.py`, paint in `materials.py`.
