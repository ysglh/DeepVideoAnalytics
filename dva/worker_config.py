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
    "detect_custom_objects":Q_DETECTOR,
    "crop_regions_by_id":Q_EXTRACTOR,
    "perform_face_detection":Q_FACE_DETECTOR,
    "export_video_by_id":Q_EXTRACTOR,
    "import_vdn_file":Q_EXTRACTOR,
    "import_vdn_s3":Q_EXTRACTOR,
    "delete_video_by_id":Q_EXTRACTOR,
    "backup_video_to_s3":Q_EXTRACTOR,
    "sync_bucket":Q_EXTRACTOR,
    "import_vdn_detector_file":Q_EXTRACTOR,
    "push_video_to_vdn_s3":Q_EXTRACTOR,
    "import_video_by_id":Q_EXTRACTOR,
    "import_video_from_s3":Q_EXTRACTOR,
    "perform_clustering": Q_CLUSTER,
    "train_yolo_detector": Q_TRAINER,
}

VIDEO_TASK = 'video'
QUERY_TASK = 'query'
S3_TASK = 's3task'
CLUSTER_TASK = 'cluster'
TRAIN_TASK = 'trainining'
IMPORT_TASK = 'import'
DEFAULT_SEGMENTS_BATCH_SIZE = 10  # how many video segments should we process at a time?
DEFAULT_FRAMES_BATCH_SIZE = 500  # How many frames/images in a dataset should we process at a time?

TASK_NAMES_TO_TYPE = {
    "segment_video": VIDEO_TASK,
    "decode_video": VIDEO_TASK,
    "extract_frames":VIDEO_TASK,
    "import_vdn_file":VIDEO_TASK,
    "import_vdn_detector_file":IMPORT_TASK,
    "import_vdn_s3":VIDEO_TASK,
    "detect_custom_objects":VIDEO_TASK,
    "export_video_by_id": VIDEO_TASK,
    "delete_video_by_id": VIDEO_TASK,
    "import_video_by_id": VIDEO_TASK,
    "backup_video_to_s3": S3_TASK,
    "push_video_to_vdn_s3": S3_TASK,
    "import_video_from_s3": S3_TASK,
    "perform_clustering": CLUSTER_TASK,
    "train_yolo_detector": TRAIN_TASK,
}

# List of tasks which can be called manually
MANUAL_VIDEO_TASKS = ['perform_indexing',
                      'perform_detection',
                      'perform_analysis',
                      'sync_bucket'
                      ]

OCR_VIDEO_TASKS = ['perform_textbox_detection_by_id',]

DEFAULT_PROCESSING_PLAN =[
    {'operation': 'perform_detection', 'arguments': {
        'filters':'__parent__',
        'detector':'coco',
        'next_tasks':[
            {'operation': 'crop_regions_by_id',
             'arguments': {
                 'filters': {'event_id': '__parent_event__'},
                 'next_tasks': [
                     {'operation': 'perform_indexing',
                      'arguments': {
                          'index': 'inception',
                          'target': 'regions',
                          'filters': {'event_id': '__grand_parent_event__', 'w__gte': 50, 'h__gte': 50}
                      }
                      },
                 ]
             }},
        ]}
     },
    {'operation': 'perform_detection', 'arguments': {
        'filters':'__parent__',
        'detector':'face',
        'next_tasks':[
            {'operation': 'crop_regions_by_id',
             'arguments': {
                 'resize':[182,182],
                 'filters': {'event_id': '__parent_event__'},
                 'next_tasks': [
                     {'operation': 'perform_indexing',
                      'arguments': {
                          'index': 'facenet',
                          'target': 'regions',
                          'filters': {'event_id': '__grand_parent_event__'}
                      }
                      },
                 ]
             }},
        ]}
     },
    {'operation': 'perform_indexing', 'arguments':
        {'index': 'inception',
         'target': 'frames',
         'filters':'__parent__'
     }},
]


SYNC_TASKS = {
    "extract_frames":[
        {'operation':'sync_bucket','arguments':{'dirname':'frames'}},
    ],
    "segment_video":[
        {'operation':'sync_bucket','arguments':{'dirname':'segments'}},
    ],
    "decode_video":[
        {'operation': 'sync_bucket', 'arguments': {'dirname': 'frames'}},
    ],
    'perform_detection':[
    ],
    'crop_regions_by_id':[
        {'operation': 'sync_bucket', 'arguments': {'dirname': 'regions'}},
    ],
    'perform_indexing':[
        {'operation': 'sync_bucket', 'arguments': {'dirname': 'indexes'}},
    ],
    'perform_face_detection':[
        {'operation': 'sync_bucket', 'arguments': {'dirname': 'regions'}},
    ],
    'import_vdn_file':[
        {'operation': 'sync_bucket', 'arguments': {}},
    ],
    'import_vdn_s3':[
        {'operation': 'sync_bucket', 'arguments': {}},
    ],
    'train_yolo_detector':[
    ],
    'import_vdn_detector_file':[
    ],
    'detect_custom_objects':[
        {'operation': 'sync_bucket', 'arguments': {'dirname': 'regions'}},
    ]
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


ANALYZERS = {
    'tag':
        {
            'task':"perform_analysis",
            'queue':Q_DETECTOR,
        },
    'ocr':
        {
            'task':"perform_analysis",
            'queue':Q_OCR,
        },
    }


if 'VGG_ENABLE' in os.environ:
    VISUAL_INDEXES['vgg']= {
            'indexer_task': "perform_indexing",
            'indexer_queue': Q_VGG,
            'retriever_queue': Q_VGG,
            'detection_specific': False
        }
    DEFAULT_PROCESSING_PLAN.append({'operation': 'perform_indexing', 'arguments': {'index': 'vgg', 'target': 'frames', 'filters': '__parent__'}})
    for k in DEFAULT_PROCESSING_PLAN:
        if k['operation'] == 'perform_detection' and k['arguments']['detector'] == 'coco':
            k['arguments']['next_tasks'][0]['arguments']['next_tasks'].append({
                'operation': 'perform_indexing',
                'arguments': {'index': 'vgg',
                              'target': 'regions',
                              'filters': {'event_id': '__grand_parent_event__',
                                          'w__gte': 50,
                                          'h__gte': 50
                                           }
                              }
            })

