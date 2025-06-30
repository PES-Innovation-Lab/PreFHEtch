# PreFHEtch

Pre-Filtering Homomorphically Encrypted queries for Triage and Candidate Handling

# Building the project

- Install the dependencies

  - [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
  - [JsonCpp](https://github.com/open-source-parsers/jsoncpp)
  - [Drogon](https://github.com/drogonframework/drogon)
  - [cpr](https://github.com/libcpr/cpr)

- Configure the build system

```bash
cd PreFHEtch
cmake -S . -B build
```

- Build the project

```bash
cmake --build build
```
