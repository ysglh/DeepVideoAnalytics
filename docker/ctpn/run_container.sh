#!/usr/bin/env bash
nvidia-docker run -p 8888:8888 --name ctpn -d -it akshayubhat/dva_ctpn
nvidia-docker cp ocr.ipynb ctpn:/root/DVA/