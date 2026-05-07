#!/usr/bin/env bash
# Build a submission archive that is safe for double-blind review.
#
# Excludes:
#   - .git/        (would expose the private GitHub remote URL,
#                  i.e. it would deanonymize the submission)
#   - .DS_Store    (macOS Finder metadata; nothing to do with the dataset)
#   - __pycache__  (compiled Python; not useful for review)
#   - *.pyc        (same)
#
# Run from inside this directory. Output is written one level up:
#   ../milestone-oracle-diagnostic-submission.tar.gz
#   ../milestone-oracle-diagnostic-submission.zip
#
# Either archive can be uploaded as supplementary material on OpenReview.
# Both contain identical contents.
set -euo pipefail

cd "$(dirname "$0")"

OUT_TAR="../milestone-oracle-diagnostic-submission.tar.gz"
OUT_ZIP="../milestone-oracle-diagnostic-submission.zip"

# Remove any symlink-farm artifacts left behind by setup_release_paths.sh; we
# do not want them in the submission archive (would duplicate data and bloat
# the zip when symlinks are followed).
rm -rf data/logs code/data 2>/dev/null || true

# Tar.gz
# --owner=0 --group=0 --numeric-owner strips real-user metadata so the archive
# does not embed our local username/group (deanonymization risk).
# --no-xattrs prevents macOS extended-attribute leakage (e.g., AppleDouble).
tar --exclude='.git' --exclude='.DS_Store' --exclude='__pycache__' --exclude='*.pyc' \
    --owner=0 --group=0 --numeric-owner --no-xattrs \
    -czf "$OUT_TAR" .

# Zip (some reviewers prefer zip; -x excludes globs)
# -y stores symlinks as symlinks rather than following; -X strips extra file
# attributes (uid/gid, AppleDouble, etc.) for anonymity.
rm -f "$OUT_ZIP"
zip -r -q -y -X "$OUT_ZIP" . \
    -x '.git/*' -x '*.DS_Store' -x '*__pycache__*' -x '*.pyc'

echo "Wrote:"
ls -lh "$OUT_TAR" "$OUT_ZIP"

echo ""
echo "Contents are identical between the two archives. Either is safe for upload."
echo "Sanity check: neither archive contains .git or .DS_Store:"
if tar -tzf "$OUT_TAR" | grep -E '\.git/|\.DS_Store' >/dev/null; then
    echo "  WARNING: tar still contains banned paths"
    exit 1
fi
if unzip -l "$OUT_ZIP" | grep -E '\.git/|\.DS_Store' >/dev/null; then
    echo "  WARNING: zip still contains banned paths"
    exit 1
fi
echo "  OK"
