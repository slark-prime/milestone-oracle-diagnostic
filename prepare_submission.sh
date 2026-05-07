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

# Tar.gz
tar --exclude='.git' --exclude='.DS_Store' --exclude='__pycache__' --exclude='*.pyc' \
    -czf "$OUT_TAR" .

# Zip (some reviewers prefer zip; -x excludes globs)
rm -f "$OUT_ZIP"
zip -r -q "$OUT_ZIP" . \
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
