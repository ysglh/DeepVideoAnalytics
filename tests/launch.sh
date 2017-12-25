#!/usr/bin/env sh
set -xe
export LAUNCH_BY_NAME_indexer_inception=1
export LAUNCH_BY_NAME_indexer_facenet=1
export LAUNCH_BY_NAME_retriever_inception=1
export LAUNCH_BY_NAME_retriever_facenet=1
export LAUNCH_BY_NAME_detector_coco=1
export LAUNCH_BY_NAME_detector_face=1
export LAUNCH_BY_NAME_analyzer_tagger=1
export LAUNCH_Q_qclusterer=1
export LAUNCH_Q_qextract=1
export LAUNCH_SCHEDULER=1
cd ../server/ && fab launch_workers_and_scheduler_from_environment:0
