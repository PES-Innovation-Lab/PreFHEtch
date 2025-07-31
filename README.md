# PreFHEtch

Pre-Filtering Homomorphically Encrypted queries for Triage and Candidate Handling

# Building the project

- Install the dependencies

    - [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
    - [JsonCpp](https://github.com/open-source-parsers/jsoncpp)
    - [Protocol Buffers](https://developers.google.com/protocol-buffers)
    - [Drogon](https://github.com/drogonframework/drogon)
    - [cpr](https://github.com/libcpr/cpr)
    - [Boost](https://www.boost.org/)

- Download the dataset

```bash
./dataset.sh
```

- Configure the build system

```bash
cd PreFHEtch
cmake -S . -B build
```

- Build the project

```bash
cmake --build build
```

- Build with docker
 
```bash 
# to build 
docker build -t prefhetch .

# start the container in the background
docker run -dit --name prefhetch-container -p 8080:8080 prefhetch

# in one terminal (for running the PreFHEtch-server)
# this will start a wait loop for the server.
docker exec -it prefhetch-container bash
cd build
./PreFHEtch-server

# in another terminal (for running the PreFHEtch-client)
docker exec -it prefhetch-container bash
cd build
./PreFHEtch_client --nq 5 --nprobe 10 --coarse-probe 200 --k 100
```
If any changes are made, you will have to run the build command again, ie:
```bash
cmake --build build
```
