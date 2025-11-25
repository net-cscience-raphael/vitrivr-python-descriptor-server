# vitrivr-python-descriptor-server

## Dependencies 
Software    | Version |
------------|---------|
Python      | 3.9     |

## In docker
```bash
git clone git@github.com:vitrivr/vitrivr-python-descriptor-server.git
cd ./vitrivr-python-descriptor-server
```

The following command builds and runs the docker container. Further it installs all python dependencies.
```bash
docker pull netcscienceraphael/vitrivr-python-descriptor-server
sudo docker run --rm -p 8888:8888 netcscienceraphael/vitrivr-python-descriptor-server:latest
```
> [!NOTE]
>  On first start the models will be downloaded and output is quite for some time
