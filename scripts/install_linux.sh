#!/usr/bin/env bash
#
# Install Earshot on Linux: global command + desktop integration.
#
#   - Installs the `earshot` command system-wide (editable, via uv tool)
#   - Creates a .desktop entry, an icon, and a launcher wrapper so
#     Earshot appears in your application launcher
#
# The installed launcher runs the editable source from this project
# directory, so editing the code and relaunching picks up your changes
# immediately.
#
# Usage:
#   scripts/install_linux.sh             # install
#   scripts/install_linux.sh --uninstall # remove
#

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/64x64/apps"
BIN_DIR="$HOME/.local/bin"
DESKTOP_FILE="$APP_DIR/earshot.desktop"
ICON_FILE="$ICON_DIR/earshot.png"
WRAPPER="$BIN_DIR/earshot-launch"

uninstall=false

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//' >&2
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --uninstall) uninstall=true ;;
        --help|-h)   usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 2 ;;
    esac
    shift
done

# Resolve uv to its full path so the launcher works in GUI sessions
# where the user's shell PATH may not be set.
UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
    echo "uv not found on PATH.  Install it: https://docs.astral.sh/uv/" >&2
    exit 1
fi

# ── uninstall ─────────────────────────────────────────────────────────

if $uninstall; then
    rm -f "$DESKTOP_FILE" "$ICON_FILE" "$WRAPPER"
    update-desktop-database "$APP_DIR" 2>/dev/null || true
    uv tool uninstall earshot 2>/dev/null || true
    echo "Removed desktop entry, icon, launcher, and global earshot command."
    exit 0
fi

# ── system dependencies ──────────────────────────────────────────────
#
# All apt packages Earshot needs at build- and run-time.  PyGObject
# (pystray's GTK/AppIndicator backend → native tray menu) needs the
# dev headers and typelibs; the rest are runtime tools.

SYS_PACKAGES=(
    libgirepository1.0-dev
    libcairo2-dev
    gir1.2-gtk-3.0
    gir1.2-ayatanaappindicator3-0.1
    gnome-shell-extension-appindicator
    libnotify-bin
)
# Typing tool is session-dependent: xdotool for X11, wtype for Wayland.
case "${XDG_SESSION_TYPE:-}" in
    wayland) SYS_PACKAGES+=(wtype) ;;
    *)       SYS_PACKAGES+=(xdotool) ;;
esac
MISSING=()
for pkg in "${SYS_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING+=("$pkg")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Installing system packages: ${MISSING[*]}"
    sudo apt-get update
    sudo apt-get install -y "${MISSING[@]}"
fi

# ── global command ────────────────────────────────────────────────────

uv tool install --editable "$PROJECT_DIR"

# ── desktop integration ───────────────────────────────────────────────

mkdir -p "$APP_DIR" "$ICON_DIR" "$BIN_DIR"

# icon ──────────────────────────────────────────────────────────────

"$UV" run --project "$PROJECT_DIR" python3 -c "
from earshot.config import COLOR_IDLE
from earshot.icon import make_icon

make_icon(COLOR_IDLE).save('$ICON_FILE')
print('Icon  -> $ICON_FILE')
"

# launcher wrapper ──────────────────────────────────────────────────
#
# Use the full path to uv + --project so the editable source in this
# directory is always used, regardless of PATH (which GUI sessions
# may not inherit from the shell).

cat > "$WRAPPER" << EOF
#!/usr/bin/env bash
exec "$UV" run --project "$PROJECT_DIR" earshot "\$@"
EOF
chmod +x "$WRAPPER"
echo "Launcher -> $WRAPPER"

# .desktop entry ────────────────────────────────────────────────────

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Earshot
Comment=Speech-to-text transcription
Exec=$WRAPPER
Icon=earshot
Terminal=false
Categories=Utility;Accessibility;
Keywords=speech;voice;dictation;transcription;
EOF

update-desktop-database "$APP_DIR" 2>/dev/null || true
echo "Desktop -> $DESKTOP_FILE"

echo ""
echo "Done.  Earshot should now appear in your application launcher"
echo "and be available as the \`earshot\` command."
echo "Edit the code in $PROJECT_DIR and relaunch to pick up changes."
