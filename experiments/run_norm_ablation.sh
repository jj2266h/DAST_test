#!/usr/bin/env bash
# Normalization ablation for the CIIE paper (RQ2, plus the RQ3 on/off arms).
# Same engine-disjoint protocol as run_clean_sweep.sh: selection on validation
# engines, one evaluation on holdout engines, official test set never read.
#
#   conda activate dast && bash experiments/run_norm_ablation.sh 1      # required
#   bash experiments/run_norm_ablation.sh 1 2 3                         # everything
#
# Phase 1 (required, 40 runs): FD002/FD004 x {global_minmax, global_z, oc_minmax, oc_z}, clip/warmup off
# Phase 2 (control,  20 runs): FD001/FD003 x {global_minmax, global_z}, clip/warmup off
#                              (oc_z == global_z on single-condition data, so not rerun)
# Phase 3 (RQ3,      30 runs): FD002/FD004 x oc_z x {warmup only, clip only, both}
#                              with c=5, w=5 (the validation-selected setting of the clean sweep)
# Safe to re-run: runs already present in summary.jsonl are skipped.

set -euo pipefail

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then PHASES=(1); fi

OUT=experiments/norm_ablation_results
SPLIT=split_dataset
SEEDS="20 42 100 7 2026"
CLIP_ON=5
WARMUP_ON=5

mkdir -p "$OUT"
SUMMARY="$OUT/summary.jsonl"
touch "$SUMMARY"

prepare() {  # dataset norm
  local stem="${1}_${2}_global"
  [ "$2" = "oc_z" ] && stem="${1}_global"
  if [ ! -f "$SPLIT/${stem}_trainX.mat" ]; then
    python experiments/prepare_splits.py --dataset "$1" --norm "$2" --out-dir "$SPLIT" --variants global
  fi
}

run() {  # dataset norm clip warmup seed
  # tag format must match train_split.py (clip is printed as a float)
  local tag="${1}_${2}_global_clip$(python -c "print(float($3))")_warmup${4}_seed${5}"
  if grep -q "\"tag\": \"$tag\"" "$SUMMARY"; then
    echo "--- skip (done): $tag"
    return
  fi
  echo "--- $tag ---"
  python experiments/train_split.py --dataset "$1" --norm "$2" --variant global \
    --split-dir "$SPLIT" --seed "$5" --clip "$3" --warmup "$4" \
    --out-dir "$OUT" --eval-holdout
}

for phase in "${PHASES[@]}"; do
  echo "=== phase $phase ==="
  case "$phase" in
    1)
      for ds in FD002 FD004; do
        for norm in global_minmax global_z oc_minmax oc_z; do
          prepare "$ds" "$norm"
          for seed in $SEEDS; do run "$ds" "$norm" 0 0 "$seed"; done
        done
      done ;;
    2)
      for ds in FD001 FD003; do
        for norm in global_minmax global_z; do
          prepare "$ds" "$norm"
          for seed in $SEEDS; do run "$ds" "$norm" 0 0 "$seed"; done
        done
      done ;;
    3)
      for ds in FD002 FD004; do
        prepare "$ds" oc_z
        for seed in $SEEDS; do
          run "$ds" oc_z 0 "$WARMUP_ON" "$seed"
          run "$ds" oc_z "$CLIP_ON" 0 "$seed"
          run "$ds" oc_z "$CLIP_ON" "$WARMUP_ON" "$seed"
        done
      done ;;
    *) echo "unknown phase: $phase"; exit 1 ;;
  esac
done

echo
echo "DONE. Runs recorded: $(wc -l < "$SUMMARY")"
echo "Summarize: python experiments/summarize_norm_ablation.py $SUMMARY"
