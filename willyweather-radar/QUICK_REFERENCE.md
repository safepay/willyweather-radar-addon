# Quick Setup Reference

## ✅ Correct Repository Structure (Single Add-on)

```
your-repository/
├── config.yaml          # Add-on configuration
├── Dockerfile           # Container definition  
├── build.yaml           # Multi-architecture builds (YAML format!)
├── README.md
├── CHANGELOG.md
├── LICENSE
└── rootfs/
    └── app/
        ├── run.sh
        └── server.py
```

**Important:** For a single add-on repository, you don NOT need `repository.yaml` in the root!

## ✅ Adding to Home Assistant

1. **Settings** → **Add-ons** → **Add-on Store**
2. Click **⋮** (three dots menu)
3. Select **Repositories**
4. Add: `https://github.com/YOUR_USERNAME/willyweather-radar-addon`
5. Click **Add**
6. Close and reopen Add-on Store
7. Your add-on should appear!

## ✅ Pre-Flight Checklist

Before pushing to GitHub:

- [ ] `config.yaml` exists and has all required fields
- [ ] `Dockerfile` exists
- [ ] `build.yaml` is in YAML format (not JSON)
- [ ] `rootfs/` directory exists with app files
- [ ] Repository will be **public** (or you have HA Cloud)
- [ ] Default branch is `main` or `master`

## ✅ Required config.yaml Fields

```yaml
name: WillyWeather Radar          # Display name
version: "1.0.0"                  # Version string
slug: willyweather_radar          # Unique identifier (lowercase, underscores)
description: Short description    # One line description
arch:                             # Supported architectures
  - aarch64
  - amd64
  - armv7
  - armhf
  - i386
```

## ✅ build.yaml Format (YAML not JSON!)

```yaml
build_from:
  aarch64: "ghcr.io/home-assistant/aarch64-base:3.19"
  amd64: "ghcr.io/home-assistant/amd64-base:3.19"
  armhf: "ghcr.io/home-assistant/armhf-base:3.19"
  armv7: "ghcr.io/home-assistant/armv7-base:3.19"
  i386: "ghcr.io/home-assistant/i386-base:3.19"
```

## ❌ Common Mistakes

1. **Using repository.yaml** - Not needed for single add-on!
2. **JSON in build.yaml** - Must be YAML format
3. **Private repository** - Must be public (unless HA Cloud)
4. **Wrong URL** - Must include `https://github.com/`
5. **Add-on in subdirectory** - Files should be in root
6. **Wrong branch** - Default should be `main` or `master`

## 🔧 Local Testing (Skip GitHub)

```bash
# SSH into Home Assistant
mkdir -p /addons/willyweather-radar
# Copy files to /addons/willyweather-radar/
ha supervisor restart
```

Your add-on will appear in "Local add-ons" - no repository needed!

## 📞 Still Not Working?

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.
