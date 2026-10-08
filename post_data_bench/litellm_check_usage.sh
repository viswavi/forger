#!/usr/bin/env bash
# Read token spend for a LiteLLM user.
# Usage: echo "<parent_key>" | ./litellm_check_usage.sh <base_url> <user_id> <api_key> [token_id]
# Keep the parent key on stdin, out of process listings. Match token_id
# exactly when supplied; otherwise match the masked key_name’s last four
# characters against api_key.

set -euo pipefail

usage() {
  echo "Usage: echo \"<parent_key>\" | $0 <base_url> <user_id> <api_key> [token_id]" >&2
  exit 1
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
fi

BASE_URL="$1"
USER_ID="$2"
API_KEY="$3"
TOKEN_ID="${4:-}"
API_KEY_LAST4="${API_KEY: -4}"

# Accept EOF without a newline: read captures the key but returns nonzero.
# Read one line so interactive callers need not send EOF.
IFS= read -r PARENT_KEY || true
if [[ -z "$PARENT_KEY" ]]; then
  echo "Error: no parent key provided on stdin." >&2
  exit 1
fi

# Strip any trailing slash from the base URL so we don't end up with "//".
BASE_URL="${BASE_URL%/}"

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required but not installed (brew install jq)." >&2
  exit 1
fi

# Pass curl’s Authorization header through stdin config, never visible argv.
RESPONSE="$(printf 'header = "Authorization: Bearer %s"\n' "$PARENT_KEY" \
  | curl -s -K - --location "${BASE_URL}/user/info?user_id=${USER_ID}")"

if [[ -z "$RESPONSE" ]]; then
  echo "Error: empty response from ${BASE_URL}/user/info" >&2
  exit 1
fi

# Make sure we actually got JSON back (and not e.g. an HTML error page).
if ! jq -e . >/dev/null 2>&1 <<<"$RESPONSE"; then
  echo "Error: response was not valid JSON:" >&2
  echo "$RESPONSE" >&2
  exit 1
fi

# Match exact token/token_id, or fall back to the masked key_name suffix.
SPEND="$(jq -r --arg token_id "$TOKEN_ID" --arg last4 "$API_KEY_LAST4" '
  .keys // [] | map(select(
    if $token_id != "" then
      (.token // "") == $token_id or
      (.token_id // "") == $token_id
    else
      ((.key_name // "")[-4:]) == $last4
    end
  )) | .[0].spend // empty
' <<<"$RESPONSE")"

if [[ -z "$SPEND" ]]; then
  if [[ -n "$TOKEN_ID" ]]; then
    echo "Error: no key with token id '${TOKEN_ID}' found for user '${USER_ID}'." >&2
  else
    echo "Error: no key with key_name ending in '${API_KEY_LAST4}' found for user '${USER_ID}'." >&2
  fi
  exit 1
fi

echo "$SPEND"
