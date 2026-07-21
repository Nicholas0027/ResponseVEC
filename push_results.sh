#!/usr/bin/env bash
# push_results.sh — commit ResponseVec code + paper + experiment artifacts to GitHub.
#
# SECURITY: the GitHub token is read ONLY from the environment variable
# GITHUB_TOKEN at runtime. It is never written to disk, never committed, and
# is stripped from the remote URL after the push. Do NOT hardcode a token here.
#
# Usage:
#   GITHUB_TOKEN=<your_token> bash push_results.sh ["commit message"]
#
# The .gitignore already excludes artifacts/ caches and secrets; this script
# force-adds ONLY the curated result subset (metrics, figures, gate decisions,
# cost table, smoke summary) so large model caches never reach the repo.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/lab/responsevec}"
REPO_URL_HOST="github.com/Nicholas0027/ResponseVEC.git"
GIT_USER_NAME="${GIT_USER_NAME:-ResponseVec Lab Bot}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-responsevec-bot@users.noreply.github.com}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${1:-chore: update ResponseVec results ($(date -u +%Y-%m-%dT%H:%M:%SZ))}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: GITHUB_TOKEN environment variable is not set." >&2
  echo "Run with:  GITHUB_TOKEN=<token> bash push_results.sh" >&2
  exit 2
fi

cd "$REPO_DIR"

# One-time init if this is not yet a git repo.
if [[ ! -d .git ]]; then
  git init -q
  git branch -M "$BRANCH"
fi

git config user.name  "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

# Curate the result subset that IS allowed into the repo even though
# artifacts/ is gitignored. Everything else under artifacts/ (model caches,
# raw representation shards) stays local.
CURATED=(
  "artifacts/metrics"
  "artifacts/figures"
  "artifacts/smoke/smoke_summary.json"
)

# Stage source + paper (tracked normally) ...
git add -A
# ... then force-add only the curated results.
for path in "${CURATED[@]}"; do
  if [[ -e "$path" ]]; then
    git add -f "$path" 2>/dev/null || true
  fi
done

# Safety net: refuse to commit anything that looks like a token/secret.
if git diff --cached --name-only | grep -Eiq '(\.env$|token|secret|credential)'; then
  echo "ERROR: a staged file name looks like a secret — aborting." >&2
  git reset -q
  exit 3
fi

if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -q -m "$COMMIT_MSG"

# Build an authenticated remote URL at runtime only; never persist the token.
AUTH_URL="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL_HOST}"
git remote remove origin 2>/dev/null || true
git remote add origin "$AUTH_URL"

# Push; fall back to setting upstream on first push.
if ! git push -q origin "$BRANCH" 2>/dev/null; then
  git push -q -u origin "$BRANCH"
fi

# Immediately scrub the token from the stored remote so `git remote -v`
# and .git/config never retain it.
git remote set-url origin "https://${REPO_URL_HOST}"

echo "Pushed to https://${REPO_URL_HOST%.git} on branch $BRANCH"
echo "Commit: $(git rev-parse --short HEAD) — $COMMIT_MSG"
