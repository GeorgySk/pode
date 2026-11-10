#!/bin/bash
# Give the container user the uid and gid that own the workspace, so files
# created in the container belong to the host user. Safe to re-run: on Linux
# the dev container CLI usually matches the uid and gid already
# (updateRemoteUserUID), and then this does nothing.
set -eux

uid=$(stat -c '%u' "$(pwd)")
gid=$(stat -c '%g' "$(pwd)")
user=$(whoami)

# Never change root's ids: another user (vscode) may already own them.
if [ "$user" = root ]; then
    exit 0
fi

# Root-owned workspace (Docker Desktop on Windows): nothing to match.
if [ "$uid" -eq 0 ] && [ "$gid" -eq 0 ]; then
    exit 0
fi

if [ "$(id -g "$user")" -ne "$gid" ]; then
    # Reuse the group that already has this gid, or create one.
    group=$(getent group "$gid" | cut -d: -f1 || true)
    if [ -z "$group" ]; then
        group=hostgroup
        sudo groupadd -g "$gid" "$group"
    fi
    sudo usermod -g "$group" "$user"
fi

if [ "$(id -u "$user")" -ne "$uid" ]; then
    sudo usermod -u "$uid" "$user"
fi
