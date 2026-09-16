# Sim vs real response-speed comparison

Signal: `action`; FPS: 30. Speed is `abs(action[t] - action[t-1]) * FPS`, computed only within the same episode.

| Joint | Sim mean speed | Real mean speed | Real - Sim | Change % | Sim p90 | Real p90 | p90 Change % |
|---|---:|---:|---:|---:|---:|---:|---:|
| shoulder_pan.pos | 12.17 | 7.44 | -4.73 | -38.9% | 44.84 | 26.37 | -41.2% |
| shoulder_lift.pos | 21.69 | 18.22 | -3.48 | -16.0% | 71.21 | 47.47 | -33.3% |
| elbow_flex.pos | 17.31 | 15.54 | -1.77 | -10.2% | 55.38 | 44.84 | -19.0% |
| wrist_flex.pos | 10.36 | 6.33 | -4.03 | -38.9% | 42.20 | 26.37 | -37.5% |
| wrist_roll.pos | 3.88 | 2.38 | -1.50 | -38.6% | 7.91 | 2.64 | -66.7% |
| gripper.pos | 7.32 | 4.98 | -2.34 | -32.0% | 15.88 | 9.37 | -41.0% |

Interpretation: positive change means the real dataset changes that joint faster on average; negative means real is slower/smoother than sim for that joint.