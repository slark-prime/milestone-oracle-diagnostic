#!/usr/bin/env bash
# Set up the directory layout the analysis scripts under code/scripts/ expect.
#
# The scripts were written against the original repo, where data lives under
# data/logs/rl/. In this release tarball, data files sit at the top of data/.
# This script creates a one-time data/logs/rl/ subdirectory of symlinks (or
# copies, for environments that do not support symlinks) so that scripts of
# the form
#
#     REPO = Path(__file__).resolve().parent.parent.parent
#     AUDIT = REPO / "data/logs/rl/audit_consensus.jsonl"
#
# resolve correctly without script modification.
#
# Run from inside this directory.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p data/logs/rl

# Top-level JSONL files
for f in data/*.jsonl; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    target="data/logs/rl/$base"
    if [ -e "$target" ] || [ -L "$target" ]; then
        rm -f "$target"
    fi
    ln -s "../../$base" "$target"
done

# oracle_panel_16k subdir
mkdir -p data/logs/rl/oracle_panel_16k
for f in data/oracle_panel_16k/*.jsonl; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    target="data/logs/rl/oracle_panel_16k/$base"
    if [ -e "$target" ] || [ -L "$target" ]; then
        rm -f "$target"
    fi
    ln -s "../../../oracle_panel_16k/$base" "$target"
done

# stage0_panel and stage0_panel_16k subdirs
for sub in stage0_panel stage0_panel_16k; do
    [ -d "data/$sub" ] || continue
    mkdir -p "data/logs/rl/$sub"
    for f in data/"$sub"/*.jsonl; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        target="data/logs/rl/$sub/$base"
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm -f "$target"
        fi
        ln -s "../../../$sub/$base" "$target"
    done
done

# Compatibility aliases: scripts use legacy filenames that the public release
# renamed. Each pair below points the legacy name at the equivalent release file.
declare -a aliases=(
    "diagnostic_multi_families_repaired.jsonl|diagnostic_354_families.jsonl"
    "repaired_decomp.jsonl|repaired_decomp_leak_safe.jsonl"
)
for pair in "${aliases[@]}"; do
    legacy="${pair%%|*}"
    actual="${pair##*|}"
    target="data/logs/rl/$legacy"
    if [ ! -e "$target" ] && [ ! -L "$target" ]; then
        ln -s "../../$actual" "$target"
    fi
done

# Scripts under code/scripts/ compute REPO = Path(__file__).resolve().parent.parent
# which resolves to code/, not the release root. They then look for
# REPO/data/logs/rl/<file>, which becomes code/data/logs/rl/<file>. Add a
# single symlink so this resolves to the same data/ tree built above.
if [ ! -e code/data ] && [ ! -L code/data ]; then
    ln -s ../data code/data
fi

echo "Layout ready. Scripts under code/scripts/ now resolve REPO/data/logs/rl/* to the released files."
echo ""
echo "Tip: rerun this script if you delete files inside data/logs/rl/."
