import os

Q_EXTRACTOR = 'qextract'
Q_INDEXER = 'qindexer'
Q_DETECTOR = 'qdetector'
Q_RETRIEVER = 'qretriever'
Q_FACE_RETRIEVER = 'qfaceretriever'
Q_FACE_DETECTOR = 'qfacedetector'
Q_CLUSTER = 'qclusterer'
Q_TRAINER = 'qtrainer'
Q_OCR = 'qocr'

QUEUES = [Q_EXTRACTOR,Q_INDEXER,Q_DETECTOR,Q_RETRIEVER,Q_FACE_RETRIEVER,Q_FACE_DETECTOR,Q_CLUSTER,Q_TRAINER,Q_OCR]

TASK_NAMES_TO_QUEUE = {
    "inception_index_by_id":Q_INDEXER,
    "vgg_index_by_id":Q_INDEXER,
    "inception_index_regions_by_id":Q_INDEXER,
    "extract_frames_by_id":Q_EXTRACTOR,
    "perform_ssd_detection_by_id":Q_DETECTOR,
    "detect_custom_objects":Q_DETECTOR,
    "perform_face_detection":Q_FACE_DETECTOR,
    "perform_face_indexing":Q_FACE_RETRIEVER,  # to save GPU memory, ideally they should be on different queue
    "alexnet_index_by_id":Q_INDEXER,
    "alexnet_query_by_image":Q_RETRIEVER,
    "export_video_by_id":Q_EXTRACTOR,
    "import_vdn_file":Q_EXTRACTOR,
    "import_vdn_s3":Q_EXTRACTOR,
    "delete_video_by_id":Q_EXTRACTOR,
    "backup_video_to_s3":Q_EXTRACTOR,
    "sync_bucket_video_by_id":Q_EXTRACTOR,
    "import_vdn_detector_file":Q_EXTRACTOR,
    "push_video_to_vdn_s3":Q_EXTRACTOR,
    "import_video_by_id":Q_EXTRACTOR,
    "import_video_from_s3":Q_EXTRACTOR,
    "perform_clustering": Q_CLUSTER,
    "assign_open_images_text_tags_by_id": Q_DETECTOR,
    "train_yolo_detector": Q_TRAINER,
    "perform_textbox_detection_by_id": Q_OCR,
    "perform_text_recognition_by_id": Q_OCR,
}


VIDEO_TASK = 'video'
QUERY_TASK = 'query'
S3_TASK = 's3task'
CLUSTER_TASK = 'cluster'
TRAIN_TASK = 'trainining'
IMPORT_TASK = 'import'

TASK_NAMES_TO_TYPE = {
    "inception_index_by_id":VIDEO_TASK,
    "vgg_index_by_id":VIDEO_TASK,
    "inception_index_regions_by_id":VIDEO_TASK,
    "extract_frames_by_id":VIDEO_TASK,
    "import_vdn_file":VIDEO_TASK,
    "import_vdn_detector_file":IMPORT_TASK,
    "import_vdn_s3":VIDEO_TASK,
    "perform_ssd_detection_by_id":VIDEO_TASK,
    "detect_custom_objects":VIDEO_TASK,
    "perform_textbox_detection_by_id":VIDEO_TASK,
    "perform_text_recognition_by_id":VIDEO_TASK,
    "perform_face_detection":VIDEO_TASK,
    "perform_face_indexing":VIDEO_TASK,
    "alexnet_index_by_id":VIDEO_TASK,
    "alexnet_query_by_image":QUERY_TASK,
    "export_video_by_id": VIDEO_TASK,
    "delete_video_by_id": VIDEO_TASK,
    "import_video_by_id": VIDEO_TASK,
    "backup_video_to_s3": S3_TASK,
    "push_video_to_vdn_s3": S3_TASK,
    "import_video_from_s3": S3_TASK,
    "perform_clustering": CLUSTER_TASK,
    "assign_open_images_text_tags_by_id": VIDEO_TASK,
    "train_yolo_detector": TRAIN_TASK,
}

# List of tasks which can be called manually
MANUAL_VIDEO_TASKS = ['inception_index_by_id',
                      'inception_index_regions_by_id',
                      'vgg_index_by_id',
                      'perform_ssd_detection_by_id',
                      'perform_textbox_detection_by_id',
                      'perform_face_detection',
                      'perform_face_indexing',
                      'perform_yolo_detection_by_id',
                      'assign_open_images_text_tags_by_id',
                      'sync_bucket_video_by_id'
                      ]

OCR_VIDEO_TASKS = ['perform_textbox_detection_by_id',]


POST_OPERATION_TASKS = {
    "extract_frames_by_id":[
        {'task_name':'perform_ssd_detection_by_id','arguments':{}},
        {'task_name':'inception_index_by_id','arguments':{}},
        {'task_name':'vgg_index_by_id','arguments':{}},
        {'task_name':'perform_face_detection','arguments':{}},
        {'task_name':'sync_bucket_video_by_id','arguments':{'dirname':'frames'}},
        {'task_name':'sync_bucket_video_by_id','arguments':{'dirname':'segments'}},
    ],
    'perform_ssd_detection_by_id':[
        {'task_name':'inception_index_regions_by_id','arguments':{'region_type':'D','object_name__startswith':'SSD_', 'w__gte':50,'h__gte':50}},
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'detections'}},
    ],
    'perform_textbox_detection_by_id':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'detections'}},
        {'task_name': 'perform_text_recognition_by_id', 'arguments': {}},
    ],
    'inception_index_by_id':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'indexes'}},
    ],
    'vgg_index_by_id':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'indexes'}},
    ],
    'perform_face_detection':[
        {'task_name': 'perform_face_indexing', 'arguments': {}},
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'detections'}},
    ],
    'perform_face_indexing':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'indexes'}},
    ],
    'inception_index_regions_by_id':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'indexes'}},
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'detections'}},
    ],
    'import_vdn_file':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {}},
    ],
    'import_vdn_s3':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {}},
    ],
    'train_yolo_detector':[
    ],
    'import_vdn_detector_file':[
    ],
    'detect_custom_objects':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'detections'}},
    ]
}
# execute_index_subquery

VISUAL_INDEXES = {
    'inception':
        {
            'indexer_task':"inception_index_by_id",
            'indexer_queue':Q_INDEXER,
            'retriever_queue':Q_RETRIEVER,
            'detection_specific':False
        },
    'facenet':
        {
            'indexer_task': "perform_face_detection_indexing_by_id",
            'indexer_queue': Q_FACE_DETECTOR,
            'retriever_queue': Q_FACE_RETRIEVER,
            'detection_specific': True
        },
    'vgg':
        {
            'indexer_task': "vgg_index_by_id",
            'indexer_queue': Q_INDEXER,
            'retriever_queue': Q_RETRIEVER,
            'detection_specific': False
        },
    }


if 'ALEX_ENABLE' in os.environ:
    POST_OPERATION_TASKS['extract_frames_by_id'].append(
        {'task_name':'alexnet_index_by_id','arguments':{}}
    )
    VISUAL_INDEXES['alexnet'] = {
         'indexer_task': "alexnet_index_by_id",
         'indexer_queue': Q_INDEXER,
         'retriever_queue': Q_RETRIEVER,
         'detection_specific': False
    }


