# What is Docker?

- Think of it like a package manager that works on any os
- Dockerfile is the recipe for the image creation
  - it starts with **FROM** that specifies from where
  - you create layers which get moved around
  - **Run:** Runs the command necessary to install
  - Each command spawns a mini layer to add back into the host linux image
  - docker build creates the image

## Docker Registry

- Universal app distrubition
- Standard for how containers are moved around
- Each image has a unique SHA Hash for the layers/metadata for that specific image
- **docker push** will push this image to the registry
- **docker pull** will pull the image to the host machine

## Docker Engine

- Runs in the background verifying image
- Image is run through **docker run**
- An app is ran in its own container
  - Built in a **namespace** which its ran in its own niche
  - Its cutoff from the rest of the is system
  - gets its own file system
- You can create two of the same containers but they won't be able to communicate
- Client communicates through various ways
  - Sockets
  - TCP/TLS
  - SSH Tunnel
- Separate containers need to run through a different host port when published
- **docker run -d -p 8800:80  public input port:  private listening
- docker run also pulls

## Why Docker

### The problem of Isolation

- Servers can waste a lot of resources if not fully utilized
- They can be hard to manage with so many different versions/apps on one app
- Hard to isolate these apps since they all share the same filesystem
- VM's allowed abstracion
- Containers allow Isolation similar to VM where they get their own IP, process space, and filesystem
  - but you don't need the full OS, MUCH simpler and safer  :()

### The Problem of Environments

- "it works on my machine"
- You may not have all the environments to test even though it works on yours
- For every part of an app and type of environemtn, there are different configurations
- consistent standard format now, OCI woot woot

### The Problem of Speed

- The speed of business is King
- The ability for business to execute their ideas
  - big part of the process lifecycle
- Distrubeted Architecture is King

## Commands

- **docker version:** Lets you know the version
- **docker info:** Returns a lot more info about Docker Engine
- **docker:** Type this to check list of commands
- **Management Commands:** Think of this as admin commands
  - You can use commands like **debug** or **init** followed by subcommands or information
- **docker container top:** process list in a container
- **docker container inspect:** details of a container config
- **docker container stats:** stats for all containers and their performance
- **docker container exec:** can run executive comamnds within container

## Image Vs. Containers

- Image: Binaries and libraries and source code that make up the source code
- Containers: instance of that image running
- docker container run --publish 80:80 nginx
  - Downloaded nginx image from Docker Hub
  - Started a new container for that image
  - Opened port 80 on the host IP
    - traffic forwards automatically from browser to the container
- **--detach** allows it to run in the background
- **--name** allows you to name container
- **logs** lets you see logs of specified container
- **-it** flag and **bash** can let you create a shell inside the container without SSH

## Package Management Essentials

- [Good Resource](https://www.digitalocean.com/community/tutorials/package-management-basics-apt-yum-dnf-pkg)
- Know your system!
  - **Debian:** Use apt and dpkg to install .deb packages
    - Systems like Ubuntu, Linux Minto, and Rasbian
  - **Rocky/Fedora/RHEL:** .rpm packages installed by yum
  - **FreeBSD:** .txz packages installed by pkg

### Updating package lists

- For Debian / Ubuntu: sudo apt update
- For Rocky / Fedora / RHEL: dnf check-update
- For FreeBSD Packages: sudo pkg update
- For FreeBSD Ports: sudo portsnap fetch update

## Docker Networks

- **docker network ls:** shows networks
- **docker network inspect:** inspect networks
- **docker network create --driver:**  create a network
- **docker network connect:** connect container to network
- **docker network disconnect:** disconnect container from network
- bridge network is default network that connects to firewall
- Create apps so frontend/backend are in same docker network!
  - Their intercommunication never leaves the host
  - All externally exposed ports are closed by default

### DNS and how it Effects Containers

- DNS is key to inter-container communication
- Cannot rely on IPAddresses to communicate since they are too dynamic
- Docker uses container names as host names for communication
- 

## Review

- Nat types, ports, and interfaces

## Resources

- [Docker Internals](https://www.youtube.com/watch?v=sK5i-N34im8&list=PLBmVKD7o3L8v7Kl_XXh3KaJl9Qw2lyuFl)
- [Docker Mac Commands](https://www.bretfisher.com/docker-for-mac-commands-for-getting-into-local-docker-vm/)
- [Docker Windows Commands](https://www.bretfisher.com/getting-a-shell-in-the-docker-for-windows-vm/)
- [Package Management Essentials](https://www.digitalocean.com/community/tutorials/package-management-basics-apt-yum-dnf-pkg)
- [Round Robin DNS](https://en.wikipedia.org/wiki/Round-robin_DNS)
