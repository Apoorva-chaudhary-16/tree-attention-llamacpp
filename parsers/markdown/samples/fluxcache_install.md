# FluxCache Installation Guide

FluxCache is an in-memory cache server written in Rust. This guide covers installation and the first start.

## Requirements

### Hardware
- At least 2 CPU cores
- 4 GB of RAM (16 GB recommended for production)
- 10 GB of free disk space for snapshots

### Software
- Linux kernel 5.10 or newer
- OpenSSL 3.0
- Rust 1.78 toolchain (only needed when building from source)

## Installation

### Using the package manager
```bash
sudo apt install fluxcache
sudo systemctl enable fluxcache
```
The package installs the binary to /usr/bin/fluxcached and the config file to /etc/fluxcache/fluxcache.toml.

### Building from source
```bash
git clone https://example.org/fluxcache.git
cd fluxcache
cargo build --release
```
The build takes about 6 minutes on a laptop. The binary is created in target/release.

## Configuration

| Option | Default | Meaning |
|---|---|---|
| port | 7421 | TCP port the server listens on |
| max_memory | 2gb | Memory limit before eviction starts |
| snapshot_interval | 300 | Seconds between disk snapshots |

### Security
Set `tls = true` and provide a certificate path. The default admin password is "changeme" and must be replaced before the first start.

## Troubleshooting

### Port already in use
If the server fails to start, another program may be using the configured port. Change the port option or stop that program.

### Slow snapshots
Snapshots slow down when the disk is nearly full. Keep at least 20 percent of the disk free.
