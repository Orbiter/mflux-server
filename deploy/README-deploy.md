# Service Deployment on Mac Server

This guide provides step-by-step instructions for installing and configuring a service to run automatically on a Macintosh.
This service will start at boot, run independently of user login, and restart automatically if it crashes.

## Prerequisites

- Ensure you have administrative access to the Macintosh.
- The service scripts are located at `triton_services/bin`.
- The plist files are located at `triton_services/LaunchDaemons/`.

## Installation Steps

### Move the plist File to the Correct Location
Copy the plit files from `service.plist` to `/Library/LaunchDaemons/`. This requires administrative privileges:

```bash
sudo cp ~/git/triton_services/LaunchDaemons/*.plist /Library/LaunchDaemons/
```

### Set File Permissions
Set the correct permissions for the plist file:

```bash
sudo chown root:wheel /Library/LaunchDaemons/de.anomic.*.plist
sudo chmod 644 /Library/LaunchDaemons/de.anomic.*.plist
```

### Load the Service
Load the service using the launchctl command:

```bash
sudo launchctl load -w /Library/LaunchDaemons/de.anomic.*.plist
```

### Verify the Service
Check if the service is loaded and running:

```bash
sudo launchctl list | grep de.anomic
```

### Logging
The service's standard output and error will be logged to:

- Standard Output: /tmp/de.anomic.*.out.log
- Standard Error: /tmp/de.anomic.*.err.log

### Modifying the launch daemon

After making changes to the .plist file, reload the daemon to apply the changes:

```bash
sudo launchctl unload /Library/LaunchDaemons/de.anomic.*.plist
sudo launchctl load -w /Library/LaunchDaemons/de.anomic.*.plist
```
