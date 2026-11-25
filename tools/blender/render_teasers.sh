#!/usr/bin/env bash
# Path-traced (Blender Cycles) versions of the six teaser views, from the same cameras as the
# benchmark_config_vis_teaser_<Scene>.json presets (1200 x 900). Needs the textured meshes in mesh/.
#
#   tools/blender/render_teasers.sh [samples] [scene ...]       e.g.  tools/blender/render_teasers.sh 512 SanMiguel
#
# Output: benchmark/teaser_renders/render_<Scene>.png and .log (OUT overrides the folder, BLENDER the executable).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BLENDER=${BLENDER:-blender}
OUT=${OUT:-$ROOT/benchmark/teaser_renders}
SPP=${1:-512}
[ $# -gt 0 ] && shift
SCENES=${*:-"VokseliaSpawn Lumberyard_all SanMiguel RungHolt EmeraldSquare PowerPlant"}

# Per-scene look. --sun auto picks the sun direction that puts the most direct light on what the camera sees.
args_for() {
  case "$1" in
    VokseliaSpawn)  echo "--pixel-art --bevel 0.04 --tex-bump 0.25 --haze 0.0015 --sun auto --exposure 0.3" ;;
    RungHolt)       echo "--pixel-art --bevel 0.04 --tex-bump 0.25 --haze 0.0015 --sun auto --exposure 0.3" ;;
    Lumberyard_all) echo "--sun auto --exposure 0.6" ;;
    SanMiguel)      echo "--sun auto --sky-strength 0.8 --exposure 1.5" ;;
    EmeraldSquare)  echo "--sun auto --exposure 0.2" ;;
    PowerPlant)     echo "--sun auto --exposure 0.3" ;;
  esac
}

mkdir -p "$OUT"
for s in $SCENES; do
  t0=$(date +%s)
  # Pipe the log: the Blender snap writes nothing when its stdout is redirected to a file.
  "$BLENDER" -b --factory-startup -P "$ROOT/tools/blender/render_teaser.py" -- \
      --preset "presets/benchmark_config_vis_teaser_$s.json" --out "$OUT/render_$s.png" --samples "$SPP" \
      $(args_for "$s") < /dev/null 2>&1 | cat > "$OUT/render_$s.log"
  echo "$s: $(( $(date +%s) - t0 )) s -> $OUT/render_$s.png $(grep -q 'render\] done' "$OUT/render_$s.log" && echo ok || echo FAILED)"
  grep -E '^\[render\] (auto sun|sun elevation)' "$OUT/render_$s.log" | sed 's/^/    /'
done
