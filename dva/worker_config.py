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
DEFAULT_SEGMENTS_BATCH_SIZE = 10  # how many video segments should we process at a time?
DEFAULT_FRAMES_BATCH_SIZE = 500  # How many frames/images in a dataset should we process at a time?
DEFAULT_RESCALE = 0
DEFAULT_RATE = 30

TASK_NAMES_TO_TYPE = {
    "perform_video_segmentation": VIDEO_TASK,
    "perform_video_decode": VIDEO_TASK,
    "perform_dataset_extraction":VIDEO_TASK,
    "perform_detector_import":IMPORT_TASK,
    "perform_deletion": VIDEO_TASK,
    "perform_clustering": CLUSTER_TASK,
    "perform_detector_training": TRAIN_TASK,
}

DEFAULT_PROCESSING_PLAN =[
    {'operation': 'perform_detection', 'arguments': {
        'filters':'__parent__',
        'detector':'coco',
        'next_tasks':[
            {'operation': 'perform_transformation',
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
            {'operation': 'perform_transformation',
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
    "perform_dataset_extraction":[
        {'operation':'perform_sync','arguments':{'dirname':'frames'}},
    ],
    "perform_video_segmentation":[
        {'operation':'perform_sync','arguments':{'dirname':'segments'}},
    ],
    "perform_video_decode":[
        {'operation': 'perform_sync', 'arguments': {'dirname': 'frames'}},
    ],
    'perform_detection':[
    ],
    'perform_transformation':[
        {'operation': 'perform_sync', 'arguments': {'dirname': 'regions'}},
    ],
    'perform_indexing':[
        {'operation': 'perform_sync', 'arguments': {'dirname': 'indexes'}},
    ],
    'perform_import':[
        {'operation': 'perform_sync', 'arguments': {}},
    ],
    'perform_detector_training':[
    ],
    'perform_detector_import':[
    ],
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
    'custom':
        {
            'task':"perform_detection",
            'queue':Q_DETECTOR,
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

