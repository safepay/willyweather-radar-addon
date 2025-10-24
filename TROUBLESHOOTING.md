# Repository Setup Troubleshooting

If Home Assistant says "Not a valid add-on repository", try these solutions:

## Solution 1: Verify Repository Structure

Your repository root should contain these files:
```
willyweather-radar-addon/
├── config.yaml          ← Required
├── Dockerfile           ← Required
├── build.yaml           ← Required for multi-arch
├── README.md
├── CHANGELOG.md
└── rootfs/
    └── app/
        ├── run.sh
        └── server.py
```

## Solution 2: Check config.yaml Format

Ensure `config.yaml` has all required fields:
- `name`
- `version`
- `slug`
- `description`
- `arch` (array of architectures)

## Solution 3: Verify build.yaml is YAML (not JSON)

The `build.yaml` should look like:
```yaml
build_from:
  aarch64: "ghcr.io/home-assistant/aarch64-base:3.19"
  amd64: "ghcr.io/home-assistant/amd64-base:3.19"
  ...
```

NOT like JSON:
```json
{
  "build_from": {
    ...
  }
}
```

## Solution 4: Wait for GitHub Pages

After pushing to GitHub, wait 1-2 minutes for GitHub to process the repository before adding it to Home Assistant.

## Solution 5: Use Correct Repository URL

In Home Assistant, add the FULL GitHub URL:
```
https://github.com/yourusername/willyweather-radar-addon
```

NOT:
- `yourusername/willyweather-radar-addon`
- `github.com/yourusername/willyweather-radar-addon`

## Solution 6: Check Branch Name

Home Assistant looks for the `main` or `master` branch by default. Ensure your default branch is named correctly.

## Solution 7: Make Repository Public

Home Assistant cannot access private repositories unless you have Home Assistant Cloud with GitHub integration.

## Solution 8: Force Refresh

Sometimes Home Assistant caches the repository list:
1. Remove the repository from Home Assistant
2. Wait 30 seconds
3. Close Settings completely
4. Reopen Settings → Add-ons → Add-on Store
5. Re-add the repository

## Still Not Working?

### Check the Supervisor Logs

1. Settings → System → Logs
2. Select "Supervisor" from dropdown
3. Look for errors related to repository fetching

### Test with curl

On your Home Assistant machine:
```bash
curl -I https://raw.githubusercontent.com/yourusername/willyweather-radar-addon/main/config.yaml
```

Should return `200 OK` and the file content.

### Common Error Messages

**"Not a valid add-on repository"**
- Missing or invalid `config.yaml`
- Wrong repository structure
- Repository is private

**"Failed to fetch repository"**
- Network/firewall issue
- GitHub is down
- Wrong URL

**"No add-ons found"**
- Files are in wrong directory
- Branch name mismatch

## Development/Testing Without GitHub

For local development, you can install directly:

1. SSH into Home Assistant
2. Create directory: `/addons/willyweather-radar/`
3. Copy all files to this directory
4. Restart Supervisor: `ha supervisor restart`
5. The add-on will appear in "Local add-ons"

This bypasses the repository entirely.
