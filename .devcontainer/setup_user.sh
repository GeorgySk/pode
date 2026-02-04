#!/bin/bash
set -eux

uid=$(stat -c '%u' "$(pwd)")
gid=$(stat -c '%g' "$(pwd)")
current_user=$(whoami)

if [ "$gid" -ne 1000 ]; then
    sudo groupadd -g "$gid" hostgroup
    if [ "$uid" -ne 1000 ]; then
        sudo usermod -u "$uid" -g hostgroup "$current_user"
    else
        sudo usermod -g hostgroup "$current_user"
    fi
elif [ "$uid" -ne 1000 ]; then
    sudo usermod -u "$uid" "$current_user"
fi
