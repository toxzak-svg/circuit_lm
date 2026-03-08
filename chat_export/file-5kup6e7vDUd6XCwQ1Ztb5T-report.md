# Setting Up and Managing a GPU Rig via SSH with a WireGuard Proxy

Managing a powerful GPU‑equipped workstation from anywhere is easier when it’s secured behind a private VPN.  This guide walks through the entire process—assembling a GPU server, installing the necessary drivers and tools, creating a secure WireGuard tunnel and connecting via SSH, and using monitoring tools to manage workloads remotely.  It is written for beginners and assumes the server runs Ubuntu 22.04 LTS; commands can be adapted to other distributions.

![Network diagram showing a WireGuard tunnel between a remote laptop and a GPU rig via the internet. Each device has its own WireGuard address (10.0.0.1 for the GPU rig and 10.0.0.2 for the laptop). The blue line indicates the VPN tunnel connecting the two through the cloud.]({{file:file-V3FzL1jCo1kPDkM2jAoQYS}})

## 1 Prerequisites and Planning

Before you build a remote‑accessible GPU server you need some preparation:

- **Hardware:** A desktop or rack‑mount workstation with a CUDA‑capable NVIDIA GPU (e.g., RTX 3060/4090).  At least 8 GB of RAM and a multi‑core CPU are recommended for machine‑learning workloads【983002689695046†L170-L175】.  Connect the server to your router via Ethernet for stable connectivity.
- **Operating System:** Ubuntu 22.04 LTS Server or Desktop.  The instructions here assume Ubuntu; commands for other distributions may differ.
- **Public network access:** Ideally the server sits behind a router with a public IP or dynamic DNS.  You’ll forward a single UDP port (51820 by default) to the server to allow WireGuard connections.  If your ISP provides CG‑NAT or blocks inbound connections, you can instead host WireGuard on a small VPS and tunnel into your home network.
- **Security considerations:** Keep the OS updated, use strong passwords/passphrases, and avoid exposing SSH directly on the internet.  WireGuard encrypts traffic and authenticates peers using public‑key cryptography【408329616742216†L44-L59】.

## 2 Preparing the GPU Server

1. **Install Ubuntu and OpenSSH.**  During the Ubuntu installation, select “OpenSSH server” so you can administer the machine remotely.  After installation, update the system:

   ```sh
   sudo apt update
   sudo apt upgrade
   ```

2. **Install OpenSSH manually (if not already installed).**  The GitHub step‑by‑step guide on personal GPU servers recommends installing the OpenSSH server with `sudo apt install openssh-server`【756501007729830†L108-L116】.  Enable and start the service so it launches automatically:

   ```sh
   sudo systemctl enable --now ssh
   sudo systemctl status ssh  # should show “Active: active (running)”【756501007729830†L120-L130】
   ```

3. **Configure the firewall.**  Ubuntu’s UFW makes it easy to allow only necessary ports.  Permit SSH access on port 22 and enable the firewall:

   ```sh
   sudo ufw allow ssh  # adds 22/tcp rule【756501007729830†L134-L152】
   sudo ufw enable
   ```

   You can verify the rules with `sudo ufw status` (you should see SSH allowed).  To avoid your workstation’s IP changing, assign a static IP in your router’s DHCP settings or edit `/etc/netplan/*.yaml` as shown in the MJUN server setup article【419934184118671†L114-L136】.

4. **Install essential tools.**  A remote server benefits from tools for process management, file transfers and code editing.  The MJUN guide recommends packages like `htop`, `tmux`, `screen`, `git`, `wget`, etc.  Install them with:

   ```sh
   sudo apt install -y git wget curl htop tmux screen avahi-daemon ffmpeg imagemagick iputils-ping net-tools zsh
   ```

   These tools enable remote editing (`vim`/`emacs`), real‑time system monitoring (`htop`), persistent sessions (`tmux`/`screen`), and local name resolution via mDNS (`avahi-daemon`)【419934184118671†L70-L107】.

5. **Set up SSH keys for passwordless login.**  Generate a key pair on your laptop with `ssh-keygen` and copy the public key to the server’s `~/.ssh/authorized_keys`.  Disable password logins by editing `/etc/ssh/sshd_config` (`PasswordAuthentication no`) and restart the SSH service.  Key‑based authentication improves security.

## 3 Installing NVIDIA Drivers and CUDA

A GPU server needs the proprietary NVIDIA driver and CUDA libraries.  Cherry Servers’ installation tutorial provides step‑by‑step commands【983002689695046†L185-L307】:

1. **Upgrade the system.**  Ensure your system is up to date:

   ```sh
   sudo apt update
   sudo apt upgrade
   ```

2. **Identify the recommended driver.**  Install the `ubuntu-drivers-common` package and list available drivers:

   ```sh
   sudo apt install ubuntu-drivers-common
   sudo ubuntu-drivers devices  # lists GPU model and recommended driver
   ```

   The output lists your GPU and recommends a driver (e.g., `nvidia-driver-535`【983002689695046†L199-L218】).

3. **Install the NVIDIA driver.**  Install the recommended driver package:

   ```sh
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

   After rebooting, verify the installation with `nvidia-smi`; it should display the driver and CUDA version【983002689695046†L247-L264】.

4. **Install GCC (required for CUDA).**  Install the GNU compiler collection:

   ```sh
   sudo apt install gcc
   gcc -v  # confirms installation【983002689695046†L269-L289】
   ```

5. **Install the CUDA toolkit.**  Download and install the CUDA repository key ring, update apt, and install CUDA:

   ```sh
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

   This procedure installs the full CUDA toolkit【983002689695046†L291-L310】.  If dependency errors occur, run `sudo apt --fix-broken install`.

6. **Set environment variables.**  Append the following lines to your `~/.bashrc` to expose CUDA binaries and libraries:

   ```sh
   echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
   source ~/.bashrc
   ```

7. **Verify CUDA installation.**  Test the CUDA compiler:

   ```sh
   nvcc -V  # displays the CUDA compiler version【983002689695046†L350-L355】
   ```

At this point your GPU rig has a working NVIDIA driver and CUDA toolkit.

## 4 Setting Up WireGuard on the GPU Server

### 4.1 What is WireGuard?

WireGuard is a modern VPN protocol that provides encrypted tunnels using a compact codebase and strong cryptography【823650699554180†L78-L115】.  Its design only responds to authenticated clients【408329616742216†L44-L59】, making it resistant to port scans and simpler than legacy VPNs like OpenVPN or IPsec.  It assigns each peer an internal IP address and uses public/private keys for authentication.

### 4.2 Install WireGuard on Ubuntu

WireGuard is available in Ubuntu’s repositories.  To install the server tools, run:

```sh
sudo apt update
sudo apt install wireguard
```

The tutorial from OnlineHashCrack confirms that WireGuard packages can be installed directly via apt on Ubuntu/Debian【823650699554180†L156-L162】.

### 4.3 Generate Server Keys

WireGuard uses Curve25519 keys.  Generate a private key and the corresponding public key, storing them securely (don’t share the private key).  Use a restrictive umask so only root can read the keys:

```sh
umask 077
wg genkey | tee /etc/wireguard/server_private.key | wg pubkey > /etc/wireguard/server_public.key【823650699554180†L195-L203】
```

### 4.4 Create the Server Configuration

Create `/etc/wireguard/wg0.conf` and set the interface address, port and keys.  Replace placeholders with actual key contents:

```ini
[Interface]
Address = 10.0.0.1/24  # WireGuard tunnel address for the server
ListenPort = 51820     # default UDP port
PrivateKey = <server_private_key>

# SaveConfig = true   # optional: automatically save runtime changes

# Client peer example (will be filled after creating client keys)
#[Peer]
#PublicKey = <client_public_key>
#AllowedIPs = 10.0.0.2/32
```

#### Firewall and System Settings

- **Open port 51820/udp.**  Allow WireGuard’s UDP port through the firewall:

  ```sh
  sudo ufw allow 51820/udp【823650699554180†L223-L229】
  ```

- **Enable IP forwarding.**  To route traffic through the VPN, enable IPv4 forwarding:

  ```sh
  echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf【823650699554180†L229-L233】
  sudo sysctl -p
  ```

- **Persist network configuration.**  If your server is behind NAT, forward UDP 51820 on your router to the GPU server’s LAN IP.

### 4.5 Start and Enable WireGuard

Start the interface using the helper script and enable it to start on boot:

```sh
sudo systemctl start wg-quick@wg0
sudo systemctl enable wg-quick@wg0【823650699554180†L239-L243】
```

Check the status with `sudo systemctl status wg-quick@wg0`.  You should see the interface `wg0` listed in `ip addr`.

## 5 Configuring a WireGuard Client

The client can be a laptop running Linux, Windows or macOS.  The OnlineHashCrack guide covers client key generation and configuration【823650699554180†L248-L276】.

1. **Install WireGuard.**  On Linux, install via package manager (e.g., `sudo apt install wireguard` on Ubuntu).  On Windows and macOS, download the official client from [wireguard.com](https://www.wireguard.com)【823650699554180†L170-L183】.  Mobile apps are available via the iOS and Android stores【823650699554180†L184-L188】.

2. **Generate client keys.**  On the client, generate a key pair:

   ```sh
   umask 077
   wg genkey | tee ~/client_private.key | wg pubkey > ~/client_public.key【823650699554180†L252-L257】
   ```

3. **Create the client configuration** (usually `~/wg0.conf` or imported via GUI).  Replace placeholders with your keys and server IP (public IP or dynamic DNS name).  The `AllowedIPs` field determines which traffic goes through the VPN:

   ```ini
   [Interface]
   PrivateKey = <client_private_key>
   Address = 10.0.0.2/24    # client’s WireGuard IP
   DNS = 1.1.1.1           # optional DNS server

   [Peer]
   PublicKey = <server_public_key>
   Endpoint = <server_public_ip>:51820  # your home IP or domain
   AllowedIPs = 0.0.0.0/0, ::/0        # route all traffic through the tunnel
   PersistentKeepalive = 25           # sends keepalive every 25 s to maintain NAT
   ```

4. **Add the client to the server’s config.**  On the GPU server, append a `[Peer]` section in `wg0.conf` with the client’s public key and assigned IP:

   ```ini
   [Peer]
   PublicKey = <client_public_key>
   AllowedIPs = 10.0.0.2/32【823650699554180†L286-L290】
   ```

   Reload the configuration:

   ```sh
   sudo systemctl restart wg-quick@wg0
   ```

5. **Bring up the interface.**  On the client, activate the tunnel:

   ```sh
   sudo wg-quick up wg0  # Linux command【823650699554180†L276-L283】
   ```

   On Windows/macOS/mobile, import the configuration file into the WireGuard app and toggle the connection.  Check the connection status with `sudo wg show`【823650699554180†L298-L305】.

6. **Test connectivity.**  Ping the server’s WireGuard IP from the client (`ping 10.0.0.1`).  If replies are received, the tunnel is established.  You can now SSH into the GPU server using its WireGuard IP:

   ```sh
   ssh user@10.0.0.1  # connect via WireGuard tunnel
   ```

   To avoid typing long commands, add an entry to your `~/.ssh/config` on the client:

   ```
   Host gpu-server
       HostName 10.0.0.1
       User your_username
       IdentityFile ~/.ssh/id_rsa
   ```

   Then connect with `ssh gpu-server`.

## 6 Monitoring and Managing the GPU Server

### 6.1 Checking GPU Utilization

Use NVIDIA’s command‑line tools to monitor the GPU:

- `nvidia-smi` displays driver version, memory usage and running processes.  Run it any time via SSH.  It provides a snapshot of GPU usage【983002689695046†L247-L264】.

- **nvtop:** A more interactive monitor similar to `htop`.  The MangoHost article explains that nvtop polls NVIDIA’s management library and displays live GPU, memory and temperature graphs【828723739171723†L73-L89】.  Install it on Ubuntu with:

  ```sh
  sudo apt update
  sudo apt install nvtop【828723739171723†L115-L128】
  nvtop  # run the interactive monitor
  ```

  Use arrow keys to navigate, `h` for help and `q` to exit【828723739171723†L150-L156】.

### 6.2 Managing Long‑Running Tasks

Remote training or rendering jobs can run for hours.  Use `tmux` or `screen` to keep sessions alive after you disconnect:

```sh
sudo apt install tmux screen
```

Launch a session (`tmux new -s train`), run your command and detach with `Ctrl‑b` followed by `d`.  You can reattach later with `tmux attach -t train`.  The MJUN guide notes that `tmux`/`screen` are invaluable for maintaining shells across logouts【419934184118671†L70-L107】.

### 6.3 Monitoring CPU and Memory

Install `htop` for a colour‑coded view of CPU cores, memory and processes【419934184118671†L99-L103】:

```sh
sudo apt install htop
htop
```

### 6.4 Transferring Files

- Use `scp` or `rsync` over SSH to move datasets and models between your local machine and the server.
- For large files or multiple transfers, `rsync -avzP` provides resume and progress features.

### 6.5 Docker and Containers (Optional)

Running workloads inside containers keeps your system clean.  MJUN’s guide suggests installing Docker and the NVIDIA Container Toolkit【419934184118671†L148-L239】:

1. **Install Docker:**

   ```sh
   curl -fsSL https://get.docker.com -o get-docker.sh【419934184118671†L155-L161】
   sudo sh get-docker.sh
   sudo gpasswd -a $USER docker  # allow non‑root Docker usage【419934184118671†L166-L168】
   ```

2. **Install NVIDIA Container Toolkit:**

   ```sh
   sudo apt-get install -y nvidia-container-toolkit【419934184118671†L233-L239】
   sudo nvidia-ctk runtime configure --runtime=docker【419934184118671†L240-L243】
   sudo systemctl restart docker
   ```

3. **Run a GPU-enabled container:**

   ```sh
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

   This command pulls an official CUDA container and runs `nvidia-smi` inside it, verifying that Docker can access the GPU.

## 7 Security Best Practices and Troubleshooting

- **Use strong keys and restrict peers.**  Never share private keys; restrict each `[Peer]` to specific `AllowedIPs` ranges【823650699554180†L286-L297】.  Regularly rotate keys and remove unused peers.
- **Keep software up to date.**  Update your OS, WireGuard and GPU drivers regularly to patch vulnerabilities【823650699554180†L143-L148】.
- **Harden SSH.**  Disable root login (`PermitRootLogin no`), require key authentication and consider multi‑factor authentication via PAM.  Change the default SSH port if desired.
- **Firewall rules.**  Only open necessary ports (22/tcp for SSH, 51820/udp for WireGuard).  Use `ufw` or `iptables` to block unsolicited traffic.
- **Troubleshooting WireGuard:**
  - **Connection timeouts:** ensure port 51820 is forwarded and allowed through firewalls【823650699554180†L223-L229】.
  - **Key mismatch:** verify that the server has the client’s public key and vice versa【823650699554180†L252-L257】.
  - **IP conflicts:** choose a WireGuard subnet different from your LAN (e.g., 10.0.0.0/24) to avoid overlapping addresses【408329616742216†L106-L119】.
  - **Monitoring:** use `sudo wg show` to inspect peer status and data transfer【823650699554180†L300-L305】.

## 8 Conclusion

By following the steps above, you build a powerful GPU workstation that can be accessed securely from anywhere.  WireGuard provides a lightweight VPN tunnel so your SSH traffic remains encrypted and the server’s local IP stays hidden.  Combining reliable SSH service, updated NVIDIA drivers, and monitoring tools like `nvtop` and `tmux` ensures that training models or rendering scenes remotely is stable and efficient.  With Docker and the NVIDIA Container Toolkit you can encapsulate complex environments and share them with others.  Keep the system patched, rotate keys regularly, and enjoy remote control of your GPU rig wherever you work.
