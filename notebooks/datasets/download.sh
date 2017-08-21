#!/usr/bin/env bash
aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/COCO_Text.zip  COCO_Text.zip
aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/captions.zip  captions.zip
aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/instances.zip  instances.zip
aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/persons.zip  persons.zip
unzip COCO_Text.zip
unzip captions.zip
unzip instances.zip
unzip persons.zip