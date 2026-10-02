# Robots

[← README](../README.md)

| Preset | Drive | v max (m/s) | ω max (rad/s) | Height (m) | LiDAR FOV / range |
|---|---|---|---|---|---|
| `agilex_limo` | differential | 1.0 | 1.0 | 0.25 | 360° / 8 m |
| `clearpath_jackal` | differential | 2.0 | 1.0 | 0.25 | 360° / 12 m |
| `jetauto_pro` | omnidirectional | 0.6 | 1.0 | 0.63 | 360° / 12 m |
| `kobuki` | differential | 0.7 | 3.14 | 0.12 | 360° / 12 m |
| `pal_tiago` | differential | 1.0 | 1.0 | 1.45 | 270° / 10 m |
| `turtlebot3_burger` | differential | 0.22 | 2.84 | 0.19 | 360° / 8 m |
| `turtlebot3_waffle_pi` | differential | 0.26 | 1.82 | 0.14 | 360° / 8 m |
| `turtlebot4` | differential | 0.31 | 1.9 | 0.35 | 360° / 12 m |
| `turtlebot4_lite` | differential | 0.31 | 1.9 | 0.19 | 360° / 12 m |

Every number in a preset (JSON) has a source: `official` with a URL, or `estimated` with the
method. Footprints of 8 robots are real sections of the manufacturers' URDF; JetAuto Pro is a
32.4 × 26 cm box. Robots are grouped into height classes (`h36`, `h63`, `h145`) that decide
which obstacles block them (a TurtleBot passes under a table, TIAGo does not).

```bash
hm3d robots                                  # list the presets
hm3d robots check my_robot.json              # validate your own preset
hm3d robots render --out robots.svg          # draw every footprint to scale
hm3d robots import-urdf turtlebot4 --dry-run # section a manufacturer URDF
```

Your own presets go into `<workspace>/robots/`; they override the packaged ones.
