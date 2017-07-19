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
Q_VGG = 'qvgg'

QUEUES = [Q_EXTRACTOR,Q_INDEXER,Q_DETECTOR,Q_RETRIEVER,Q_FACE_RETRIEVER,Q_FACE_DETECTOR,Q_CLUSTER,Q_TRAINER,Q_OCR,Q_VGG]
INDEXER_TASKS = {'perform_indexing',}

TASK_NAMES_TO_QUEUE = {
    "segment_video":Q_EXTRACTOR,
    "decode_video":Q_EXTRACTOR,
    "extract_frames":Q_EXTRACTOR,
    "perform_ssd_detection_by_id":Q_DETECTOR,
    "detect_custom_objects":Q_DETECTOR,
    "crop_regions_by_id":Q_EXTRACTOR,
    "perform_face_detection":Q_FACE_DETECTOR,
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
    "segment_video": VIDEO_TASK,
    "decode_video": VIDEO_TASK,
    "extract_frames":VIDEO_TASK,
    "import_vdn_file":VIDEO_TASK,
    "import_vdn_detector_file":IMPORT_TASK,
    "import_vdn_s3":VIDEO_TASK,
    "perform_ssd_detection_by_id":VIDEO_TASK,
    "detect_custom_objects":VIDEO_TASK,
    "perform_textbox_detection_by_id":VIDEO_TASK,
    "perform_text_recognition_by_id":VIDEO_TASK,
    "perform_face_detection":VIDEO_TASK,
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
MANUAL_VIDEO_TASKS = ['perform_indexing',
                      'perform_ssd_detection_by_id',
                      'perform_textbox_detection_by_id',
                      'perform_face_detection',
                      'assign_open_images_text_tags_by_id',
                      'sync_bucket_video_by_id'
                      ]

OCR_VIDEO_TASKS = ['perform_textbox_detection_by_id',]


POST_OPERATION_TASKS = {
    "extract_frames":[
        {'task_name':'perform_ssd_detection_by_id','arguments':{}},
        {'task_name': 'perform_indexing',
         'arguments': {'index': 'inception', 'target': 'frames',}
         },
        {'task_name':'perform_face_detection','arguments':{}},
        {'task_name':'sync_bucket_video_by_id','arguments':{'dirname':'frames'}},
        {'task_name':'sync_bucket_video_by_id','arguments':{'dirname':'segments'}},
    ],
    "segment_video":[
        {'task_name':'perform_indexing',
            'arguments':{'index':'inception','target': 'frames'}
         },
        {'task_name':'perform_face_detection','arguments':{}},
        {'task_name':'sync_bucket_video_by_id','arguments':{'dirname':'frames'}},
        {'task_name':'sync_bucket_video_by_id','arguments':{'dirname':'segments'}},
    ],
    'perform_ssd_detection_by_id':[
        {'task_name':'crop_regions_by_id',
         'arguments':{
            'filters':{'event_id':'__parent__'},
            'next_tasks':[
                {'task_name':'perform_indexing',
                    'arguments':{
                        'index':'inception',
                        'target':'regions',
                        'filters':{'event_id':'__grand_parent__','w__gte':50,'h__gte':50}
                    }
                 },
            ]
        }},
    ],
    'crop_regions_by_id':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'regions'}},
    ],
    'perform_textbox_detection_by_id':[
        {'task_name': 'crop_regions_by_id', 'arguments': {
            'selector': 'object_name__startswith',
            'prefix': 'CTPN_TEXTBOX',
            'next_tasks': [
                {'task_name': 'perform_text_recognition_by_id',
                 'arguments': {}
                 }]
        }},
    ],
    'perform_indexing':[
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'indexes'}},
    ],
    'perform_face_detection':[
        {'task_name': 'perform_indexing',
         'arguments': {'index': 'facenet',
                       'target': 'regions',
                       'filter':{'object_name__startswith':'MTCNN_face'}}
         },
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'regions'}},
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
        {'task_name': 'sync_bucket_video_by_id', 'arguments': {'dirname': 'regions'}},
    ]
}
# execute_index_subquery

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


if 'VGG_ENABLE' in os.environ:
    VISUAL_INDEXES['vgg']= {
            'indexer_task': "perform_indexing",
            'indexer_queue': Q_VGG,
            'retriever_queue': Q_VGG,
            'detection_specific': False
        }
    POST_OPERATION_TASKS['extract_frames'].append(
        {
            'task_name':'perform_indexing',
            'arguments':{'index':'vgg','target': 'frames'}
         },
    )
    POST_OPERATION_TASKS['segment_video'].append(
        {
            'task_name':'perform_indexing',
            'arguments':{'index':'vgg','target': 'frames'}
         },
    )
    POST_OPERATION_TASKS['perform_ssd_detection_by_id'][0]['arguments']['next_tasks'].append({
        'task_name': 'perform_indexing',
         'arguments': {
             'index': 'vgg',
             'target': 'regions',
             'filters': {'event_id': '__grand_parent__', 'w__gte': 50, 'h__gte': 50}
         }}
    )

