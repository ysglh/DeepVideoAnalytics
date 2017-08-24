import os

Q_MANAGER = 'qmanager'
Q_EXTRACTOR = 'qextract'
Q_INDEXER = 'qindexer'
Q_DETECTOR = 'qdetector'
Q_RETRIEVER = 'qretriever'
Q_FACE_RETRIEVER = 'qfaceretriever'
Q_FACE_DETECTOR = 'qfacedetector'
Q_CLUSTER = 'qclusterer'
Q_TRAINER = 'qtrainer'
Q_OCR = 'qocr'
Q_VGG = 'qvgg'

QUEUES = [Q_EXTRACTOR,Q_INDEXER,Q_DETECTOR,Q_RETRIEVER,Q_FACE_RETRIEVER,Q_FACE_DETECTOR,Q_CLUSTER,Q_TRAINER,Q_OCR,Q_VGG,Q_MANAGER]
INDEXER_TASKS = {'perform_indexing',}

TASK_NAMES_TO_QUEUE = {
    "perform_video_segmentation":Q_EXTRACTOR,
    "perform_video_decode":Q_EXTRACTOR,
    "perform_dataset_extraction":Q_EXTRACTOR,
    "perform_transformation":Q_EXTRACTOR,
    "perform_export":Q_EXTRACTOR,
    "perform_deletion":Q_EXTRACTOR,
    "perform_sync":Q_EXTRACTOR,
    "perform_detector_import":Q_EXTRACTOR,
    "perform_import":Q_EXTRACTOR,
    "perform_clustering": Q_CLUSTER,
    "perform_detector_training": Q_TRAINER,
}

VIDEO_TASK = 'video'
QUERY_TASK = 'query'
S3_TASK = 's3task'
CLUSTER_TASK = 'cluster'
TRAIN_TASK = 'trainining'
IMPORT_TASK = 'import'

TASK_NAMES_TO_TYPE = {
    "perform_video_segmentation": VIDEO_TASK,
    "perform_video_decode": VIDEO_TASK,
    "perform_dataset_extraction":VIDEO_TASK,
    "perform_detector_import":IMPORT_TASK,
    "perform_deletion": VIDEO_TASK,
    "perform_clustering": CLUSTER_TASK,
    "perform_detector_training": TRAIN_TASK,
}


VISUAL_INDEXES = {
    'inception':
        {
            'indexer_task':"perform_indexing",
            'indexer_queue':Q_INDEXER,
            'retriever_queue':Q_RETRIEVER,
            'detection_specific':False
        },
    'facenet':
        {
            'indexer_task': "perform_indexing",
            'indexer_queue': Q_FACE_RETRIEVER,
            'retriever_queue': Q_FACE_RETRIEVER,
            'detection_specific': True
        },
    }


ANALYZERS = {
    'crnn':
        {
            'queue':"qcrnn",
        },
    'tag':
        {
            'queue': "qtag",
        },
}


DETECTORS = {
    'face':
        {
            'task':"perform_detection",
            'queue':Q_FACE_DETECTOR,
        },
    'coco':
        {
            'task':"perform_detection",
            'queue':Q_DETECTOR,
        },
    'textbox':
        {
            'task':"perform_detection",
            'queue':Q_OCR,
        },
    }
