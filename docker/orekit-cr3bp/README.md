# Earth-Moon Halo Orbit in Docker

How to run Orekit's EarthMoonHaloOrbit tutorial (a northern halo orbit around Earth-Moon L1, computed in the CR3BP) inside a container, with Java, all libraries, and Orekit's data already in. This also runs different CR3BP tutorials from the same image.

## Folder layout

```
halo-docker/
 -  Dockerfile
 -  .dockerignore
 -  orekit-tutorials/
 -  orekit-data/        
```

## 1. Install Docker

Install Docker Desktop and make sure it's running:

```
docker --version
```

## 2. Get the tutorial source code

```
cd halo-docker
git clone https://gitlab.orekit.org/orekit/orekit-tutorials.git
cd orekit-tutorials
git tag            # list releases
git checkout 13.1  # Latest version, subject to change
cd ..
```

## 3. Get orekit-data

Download the orekit-data archive from https://gitlab.orekit.org/orekit/orekit-data, unzip it inside `halo-docker/`, and rename the folder to `orekit-data`.

## 4. Build the image

```
docker build -t halo-sim .
```

## 5. Run it

```
docker run --rm halo-sim
```

Run a different CR3BP tutorial from the same image:

```
docker run --rm halo-sim org.orekit.tutorials.propagation.cr3bp.ManifoldTransfer
docker run --rm halo-sim org.orekit.tutorials.propagation.cr3bp.PropagationInCR3BP
```

## 6. Share it

People can either build from this folder, or receive the finished image:

```
docker save halo-sim -o halo-sim.tar    # Sender
docker load -i halo-sim.tar             # Receiver
docker run --rm halo-sim
```