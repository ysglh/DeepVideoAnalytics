Q_MANAGER = 'qmanager'
Q_EXTRACTOR = 'qextract'
Q_CLUSTER = 'qclusterer'
Q_TRAINER = 'qtrainer'
Q_LAMBDA = 'qlambda'

TASK_NAMES_TO_QUEUE = {
    "perform_region_import":Q_EXTRACTOR,
    "perform_model_import":Q_EXTRACTOR,
    "perform_video_segmentation":Q_EXTRACTOR,
    "perform_video_decode":Q_EXTRACTOR,
    "perform_frame_download": Q_EXTRACTOR,
    "perform_dataset_extraction":Q_EXTRACTOR,
    "perform_transformation":Q_EXTRACTOR,
    "perform_export":Q_EXTRACTOR,
    "perform_deletion":Q_EXTRACTOR,
    "perform_sync":Q_EXTRACTOR,
    "perform_detector_import":Q_EXTRACTOR,
    "perform_import":Q_EXTRACTOR,
    "perform_retriever_creation": Q_CLUSTER,
    "perform_detector_training": Q_TRAINER,
    "perform_video_decode_lambda": Q_LAMBDA
}