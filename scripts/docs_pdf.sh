#!/bin/bash
# The docs (mkdocs nav order) as one PDF: build/grace-al-docs.pdf.
# Needs pandoc and xelatex. Links between pages become links inside the PDF.
set -euo pipefail
cd "$(dirname "$0")/.."
out=build/grace-al-docs.pdf
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
pages=(index quickstart concepts hpc troubleshooting sites/bwunicluster3 developer config)
version=$(python -c "import grace_al; print(grace_al.__version__)" 2>/dev/null || echo dev)

for p in "${pages[@]}"; do
    # first heading of the page = its chapter; give it an id the other pages can link to
    id="page-${p//\//-}"
    sed -E "1s/^# (.*)$/# \1 {#$id}/" "docs/$p.md" |
        # links to other pages: (x.md#anchor) -> (#anchor), (x.md) -> (#page-x)
        sed -E 's/\]\(([a-z0-9_/]+)\.md#([^)]*)\)/](#\2)/g; s/\]\(([a-z0-9_]+)\/([a-z0-9_]+)\.md\)/](#page-\1-\2)/g; s/\]\(([a-z0-9_]+)\.md\)/](#page-\1)/g'
    printf '\n\n'
done > "$tmp/all.md"

cat > "$tmp/header.tex" <<'TEX'
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
\fvset{breaklines,breakanywhere}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.15}
TEX

mkdir -p build
pandoc -f markdown-smart "$tmp/all.md" -o "$out" --pdf-engine=xelatex --toc --toc-depth=2 \
    --top-level-division=chapter --highlight-style=tango --include-in-header="$tmp/header.tex" \
    -V documentclass=report -V geometry:a4paper,margin=2.2cm -V mainfont="Liberation Serif" \
    -V sansfont="Liberation Sans" -V monofont="Liberation Mono" -V fontsize=10pt \
    -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
    -M title="grace-al" -M subtitle="Foundation-model-driven active learning for GRACE fine-tuning" \
    -M date="version $version, $(date +%Y-%m-%d)"
echo "$out"
