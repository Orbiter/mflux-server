#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Step 1: create a plist file in LaunchDaemons

# Get the username of the user who invoked the script (not root if run with sudo)
if [[ -n "$SUDO_USER" ]]; then
    USERNAME="$SUDO_USER"
else
    USERNAME=$(whoami)
fi

# Get the parent directory of the current working directory and append "run.sh"
RUNSCRIPT="$(dirname "$(pwd)")/run.sh"

# Define the template and destination file paths
TEMPLATE_FILE="de.anomic.mflux-server.plist.template"
DEST_FILE="/Library/LaunchDaemons/de.anomic.mflux-server.plist"

# Check if the template file exists
if [[ ! -f "$TEMPLATE_FILE" ]]; then
    echo "Template file $TEMPLATE_FILE does not exist. Please provide the template file."
    exit 1
fi

# Check if the script is run as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root. Use sudo."
    exit 1
fi

# Use sed for macOS-compatible substitutions
sed -e "s|\\\$USERNAME\\\$|$USERNAME|g" -e "s|\\\$RUNSCRIPT\\\$|$RUNSCRIPT|g" "$TEMPLATE_FILE" > "$DEST_FILE"

# Set the correct permissions for the destination file
chmod 644 "$DEST_FILE"

echo "Plist file has been created at $DEST_FILE"

# Step 2: Set permissions for the plist file
echo "Setting permissions for $DEST_FILE..."
chown root:wheel "$DEST_FILE"
chmod 644 "$DEST_FILE"

# Step 3: Load the service
echo "Loading the service..."
launchctl unload "$DEST_FILE" 2>/dev/null || true
launchctl load -w "$DEST_FILE"

# Step 4: Verify the service
echo "Verifying the service..."
if launchctl list | grep -q "de.anomic.mflux-server"; then
    echo "Service successfully loaded."
else
    echo "Error: Service did not load properly."
    exit 1
fi

# Step 5: Logging Information
echo "Service logs will be available at:"
echo "  /tmp/de.anomic.mflux-server.log"

echo "Deployment completed successfully."
