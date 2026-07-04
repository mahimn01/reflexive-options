#!/bin/sh
# arxiv_build.sh — build a reproducible arXiv source tarball for the
# "Reflexivity in Options Markets" paper (v0.3.11).
#
# Output: dist/arxiv_v0.3.11.tar.gz
# Log:    dist/arxiv_build.log
#
# Idempotent: same source -> bit-identical tarball
# (uses SOURCE_DATE_EPOCH=0 + sorted tar listing + gzip -n).
#
# Does NOT modify paper/main.tex, paper/references.bib, or any
# file under paper/figures/. Read-only access to those.

set -eu

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper"
DIST_DIR="$REPO_ROOT/dist"
BUILD_DIR="$DIST_DIR/arxiv_build"
EXTRACT_TEST_DIR="$DIST_DIR/arxiv_extract_test"
LOG="$DIST_DIR/arxiv_build.log"
VERSION="v0.3.11"
TARBALL="$DIST_DIR/arxiv_${VERSION}.tar.gz"
EXPECTED_PAGES=40
MAX_TARBALL_BYTES=$((5 * 1024 * 1024))   # 5 MB hard ceiling
PDF_TOLERANCE_BYTES=8192                  # 8 KB drift tolerance (build-timestamp)

mkdir -p "$DIST_DIR"
: > "$LOG"

# Reproducibility: pin all build timestamps to epoch 0.
export SOURCE_DATE_EPOCH=0
export TZ=UTC
export LC_ALL=C
export LANG=C

# ----------------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------------
log() {
    printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"
}
fail() {
    printf '[%s] FAIL: %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG" >&2
    exit 1
}

log "arxiv_build.sh START (version=$VERSION)"
log "  REPO_ROOT  = $REPO_ROOT"
log "  PAPER_DIR  = $PAPER_DIR"
log "  BUILD_DIR  = $BUILD_DIR"
log "  TARBALL    = $TARBALL"

# ----------------------------------------------------------------------------
# Pre-flight: required tools
# ----------------------------------------------------------------------------
for tool in pdflatex bibtex tar gzip grep awk find sort; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        fail "required tool not found: $tool"
    fi
done

# pdfinfo is optional but preferred for the page-count check.
HAVE_PDFINFO=0
if command -v pdfinfo >/dev/null 2>&1; then
    HAVE_PDFINFO=1
fi

# ----------------------------------------------------------------------------
# Step 1: clean & recreate the build dir
# ----------------------------------------------------------------------------
log "Step 1: prepare build dir"
rm -rf "$BUILD_DIR" "$EXTRACT_TEST_DIR"
mkdir -p "$BUILD_DIR/figures"

# ----------------------------------------------------------------------------
# Step 2: discover referenced figures via grep on main.tex
# (only ship what's actually referenced; no extras)
# ----------------------------------------------------------------------------
log "Step 2: discover referenced figures + inputs"
REFERENCED="$(grep -oE 'figures/[a-zA-Z0-9_./-]+' "$PAPER_DIR/main.tex" \
    | sort -u)"

if [ -z "$REFERENCED" ]; then
    fail "no figures referenced from main.tex — grep returned empty"
fi

log "  referenced asset list:"
printf '%s\n' "$REFERENCED" | sed 's/^/    /' | tee -a "$LOG" >/dev/null

# ----------------------------------------------------------------------------
# Step 3: copy source files into build dir
# ----------------------------------------------------------------------------
log "Step 3: copy main.tex, references.bib, figures/*"
cp "$PAPER_DIR/main.tex"      "$BUILD_DIR/main.tex"
cp "$PAPER_DIR/references.bib" "$BUILD_DIR/references.bib"

COPIED_COUNT=0
MISSING_COUNT=0
printf '%s\n' "$REFERENCED" | while IFS= read -r rel; do
    src="$PAPER_DIR/$rel"
    dst="$BUILD_DIR/$rel"
    if [ ! -f "$src" ]; then
        printf '  MISSING: %s\n' "$rel" | tee -a "$LOG" >&2
        MISSING_COUNT=$((MISSING_COUNT + 1))
        continue
    fi
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    COPIED_COUNT=$((COPIED_COUNT + 1))
done

# Verify nothing was missing (the subshell loses the counter; re-check).
for rel in $REFERENCED; do
    if [ ! -f "$BUILD_DIR/$rel" ]; then
        fail "referenced file not copied: $rel"
    fi
done

# Bibstyle file is shipped from TeXLive at arXiv-build time; no need to copy.

# ----------------------------------------------------------------------------
# Step 4: build PDF locally (pdflatex x2, bibtex, pdflatex x2)
# ----------------------------------------------------------------------------
log "Step 4: build PDF in $BUILD_DIR"
cd "$BUILD_DIR"

run_latex() {
    log "  running: pdflatex (pass $1)"
    if ! pdflatex -interaction=nonstopmode -halt-on-error main.tex \
            >>"$LOG" 2>&1; then
        log "  pdflatex pass $1 FAILED — tail of log:"
        tail -40 "$LOG" >&2
        fail "pdflatex pass $1 failed"
    fi
}

run_latex 1
log "  running: bibtex"
if ! bibtex main >>"$LOG" 2>&1; then
    log "  bibtex FAILED — tail of log:"
    tail -40 "$LOG" >&2
    fail "bibtex failed"
fi
run_latex 2
run_latex 3

# ----------------------------------------------------------------------------
# Step 5: sanity check the built PDF vs the master
# ----------------------------------------------------------------------------
log "Step 5: validate built PDF"
if [ ! -f main.pdf ]; then
    fail "main.pdf not produced"
fi
if [ ! -f main.bbl ]; then
    fail "main.bbl not produced"
fi

BUILT_BYTES=$(wc -c < main.pdf | tr -d ' ')
MASTER_PDF="$PAPER_DIR/main.pdf"
if [ -f "$MASTER_PDF" ]; then
    MASTER_BYTES=$(wc -c < "$MASTER_PDF" | tr -d ' ')
    DIFF_BYTES=$((BUILT_BYTES - MASTER_BYTES))
    [ $DIFF_BYTES -lt 0 ] && DIFF_BYTES=$((-DIFF_BYTES))
    log "  built=$BUILT_BYTES bytes, master=$MASTER_BYTES bytes, diff=$DIFF_BYTES bytes"
    if [ $DIFF_BYTES -gt $PDF_TOLERANCE_BYTES ]; then
        log "  WARN: PDF size diverges from master by > $PDF_TOLERANCE_BYTES bytes"
        log "        (typical cause: master rebuilt under different timestamp;"
        log "         not a blocker for arXiv but worth a manual diff)"
    fi
fi

# Page count
PAGES=""
if [ $HAVE_PDFINFO -eq 1 ]; then
    PAGES="$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')"
else
    # Fallback: parse the LaTeX log for "Output written ... (N pages"
    PAGES="$(grep -oE 'Output written on main\.pdf \([0-9]+ pages' main.log \
        | grep -oE '[0-9]+' | head -1)"
fi
log "  page count = $PAGES (expected $EXPECTED_PAGES)"
if [ "$PAGES" != "$EXPECTED_PAGES" ]; then
    fail "page count mismatch: got '$PAGES', expected $EXPECTED_PAGES"
fi

# ----------------------------------------------------------------------------
# Step 6: strip build detritus & macOS junk before tarring
# ----------------------------------------------------------------------------
log "Step 6: strip build artefacts"
# Keep: main.tex, main.bbl, references.bib, figures/*
# Drop: main.pdf, main.aux, main.log, main.out, main.toc, main.synctex.gz,
#       main.fls, main.fdb_latexmk, blg, *.DS_Store, *.gz, *.png (none needed)
rm -f main.pdf main.aux main.log main.out main.toc \
      main.synctex.gz main.fls main.fdb_latexmk main.blg
find "$BUILD_DIR" -name '.DS_Store' -type f -delete
# Defensively scrub any png files that might have slipped into figures/
find "$BUILD_DIR/figures" -type f \! \( -name '*.pdf' -o -name '*.tex' \) -print -delete \
    | sed 's/^/  scrubbed: /' | tee -a "$LOG" >/dev/null || true

log "  build dir contents (final):"
(cd "$BUILD_DIR" && find . -type f | LC_ALL=C sort) | sed 's/^/    /' | tee -a "$LOG" >/dev/null

# Force every shipped file to mtime=epoch 0 so tar serialises deterministically
# across runs / hosts (BSD tar ignores --mtime flags; touch is portable).
log "  pinning mtimes to 1970-01-01T00:00:00Z for reproducibility"
find "$BUILD_DIR" -exec touch -t 197001010000.00 {} +

# ----------------------------------------------------------------------------
# Step 7: bundle the tarball (deterministic ordering, no timestamps)
# ----------------------------------------------------------------------------
log "Step 7: bundle tarball"
cd "$BUILD_DIR"

# Build a sorted file list (the tar -T trick + --no-recursion gives bit-identical
# archives across runs / hosts). Sort excludes the leading "./".
FILE_LIST="$(mktemp -t arxivlist.XXXXXX)"
find . -type f \! -name '.DS_Store' | LC_ALL=C sort > "$FILE_LIST"

# Detect tar flavour — BSD tar (default on macOS) and GNU tar differ.
TAR_FLAVOUR="bsd"
if tar --version 2>/dev/null | grep -qi 'gnu tar'; then
    TAR_FLAVOUR="gnu"
fi
log "  tar flavour: $TAR_FLAVOUR"

# Pack to a .tar first, then gzip -n separately (so the gz header carries
# no embedded filename / timestamp).
TAR_PLAIN="$DIST_DIR/arxiv_${VERSION}.tar"
rm -f "$TAR_PLAIN" "$TARBALL"

if [ "$TAR_FLAVOUR" = "gnu" ]; then
    tar --sort=name \
        --mtime='@0' \
        --owner=0 --group=0 --numeric-owner \
        --format=ustar \
        --no-recursion \
        -T "$FILE_LIST" \
        -cf "$TAR_PLAIN"
else
    # BSD tar (macOS)
    tar --uid 0 --gid 0 --uname '' --gname '' \
        --format ustar \
        -cf "$TAR_PLAIN" \
        -T "$FILE_LIST"
fi

rm -f "$FILE_LIST"

gzip -n -9 -c "$TAR_PLAIN" > "$TARBALL"
rm -f "$TAR_PLAIN"

TAR_BYTES=$(wc -c < "$TARBALL" | tr -d ' ')
log "  tarball size = $TAR_BYTES bytes"
if [ $TAR_BYTES -gt $MAX_TARBALL_BYTES ]; then
    fail "tarball too large: $TAR_BYTES bytes (cap $MAX_TARBALL_BYTES)"
fi

# Sanity: list contents
log "  tarball contents:"
gzip -dc "$TARBALL" | tar -tf - | LC_ALL=C sort | sed 's/^/    /' | tee -a "$LOG" >/dev/null

# ----------------------------------------------------------------------------
# Step 8: simulated arXiv sanity-check — extract to a fresh tmpdir,
# pdflatex the source, confirm 30-page output.
# ----------------------------------------------------------------------------
log "Step 8: simulated arXiv sanity-check (extract + pdflatex)"
mkdir -p "$EXTRACT_TEST_DIR"
TMP_EXTRACT="$(mktemp -d "$EXTRACT_TEST_DIR/arxiv_sanity.XXXXXX")"
log "  extract dir: $TMP_EXTRACT"

(cd "$TMP_EXTRACT" && gzip -dc "$TARBALL" | tar -xf -)

# arXiv AutoTeX uses the existing .bbl rather than re-running bibtex (recommended).
# Mimic that exactly: ONE pdflatex pass with the shipped .bbl, then a second
# pass to resolve cross-refs.
cd "$TMP_EXTRACT"
log "  arXiv-style build: pdflatex pass 1"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >>"$LOG" 2>&1 \
    || fail "arXiv-sim pdflatex pass 1 failed"
log "  arXiv-style build: pdflatex pass 2"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >>"$LOG" 2>&1 \
    || fail "arXiv-sim pdflatex pass 2 failed"

SIM_PAGES=""
if [ $HAVE_PDFINFO -eq 1 ]; then
    SIM_PAGES="$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')"
else
    SIM_PAGES="$(grep -oE 'Output written on main\.pdf \([0-9]+ pages' main.log \
        | grep -oE '[0-9]+' | head -1)"
fi
log "  arXiv-sim built PDF: $SIM_PAGES pages (expected $EXPECTED_PAGES)"
if [ "$SIM_PAGES" != "$EXPECTED_PAGES" ]; then
    fail "arXiv-sim page count mismatch: got '$SIM_PAGES'"
fi

SIM_BYTES=$(wc -c < main.pdf | tr -d ' ')
log "  arXiv-sim PDF bytes: $SIM_BYTES"
if [ -f "$MASTER_PDF" ]; then
    DIFF_SIM=$((SIM_BYTES - MASTER_BYTES))
    [ $DIFF_SIM -lt 0 ] && DIFF_SIM=$((-DIFF_SIM))
    log "  arXiv-sim vs master diff = $DIFF_SIM bytes"
fi

# ----------------------------------------------------------------------------
# Step 9: report sha256 for transparency
# ----------------------------------------------------------------------------
log "Step 9: tarball checksum"
if command -v shasum >/dev/null 2>&1; then
    SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
    SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"
else
    SHA="(no sha256 tool available)"
fi
log "  sha256 = $SHA"

log "arxiv_build.sh DONE"
log ""
log "Deliverable: $TARBALL"
log "  size:    $TAR_BYTES bytes"
log "  sha256:  $SHA"
log "  pages:   $PAGES (verified via simulated arXiv build)"
log "Full log: $LOG"
