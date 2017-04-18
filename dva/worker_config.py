import os

Q_EXTRACTOR = 'qextract'
Q_INDEXER = 'qindexer'
Q_DETECTOR = 'qdetector'
Q_RETRIEVER = 'qretriever'
Q_FACE_RETRIEVER = 'qfaceretriever'
Q_FACE_DETECTOR = 'qfacedetector'
Q_CLUSTER = 'qclusterer'

QUEUES = [Q_EXTRACTOR,Q_INDEXER,Q_DETECTOR,Q_RETRIEVER,Q_FACE_RETRIEVER,Q_FACE_DETECTOR,Q_CLUSTER]

TASK_NAMES_TO_QUEUE = {
    "inception_index_by_id":Q_INDEXER,
    "inception_index_ssd_detection_by_id":Q_INDEXER,
    "inception_query_by_image":Q_RETRIEVER,
    "facenet_query_by_image":Q_FACE_RETRIEVER,
    "extract_frames_by_id":Q_EXTRACTOR,
    "perform_ssd_detection_by_id":Q_DETECTOR,
    "perform_yolo_detection_by_id":Q_DETECTOR,
    "perform_face_detection_indexing_by_id":Q_FACE_DETECTOR,
    "alexnet_index_by_id":Q_INDEXER,
    "alexnet_query_by_image":Q_RETRIEVER,
    "export_video_by_id":Q_EXTRACTOR,
    "backup_video_to_s3":Q_EXTRACTOR,
    "import_video_by_id":Q_EXTRACTOR,
    "import_video_from_s3":Q_EXTRACTOR,
    "perform_clustering": Q_CLUSTER,
}



VIDEO_TASK = 'video'
QUERY_TASK = 'query'
S3_TASK = 's3task'
CLUSTER_TASK = 'cluster'

TASK_NAMES_TO_TYPE = {
    "inception_index_by_id":VIDEO_TASK,
    "inception_index_ssd_detection_by_id":VIDEO_TASK,
    "inception_query_by_image":QUERY_TASK,
    "facenet_query_by_image":QUERY_TASK,
    "extract_frames_by_id":VIDEO_TASK,
    "perform_ssd_detection_by_id":VIDEO_TASK,
    "perform_yolo_detection_by_id":VIDEO_TASK,
    "perform_face_detection_indexing_by_id":VIDEO_TASK,
    "alexnet_index_by_id":VIDEO_TASK,
    "alexnet_query_by_image":QUERY_TASK,
    "export_video_by_id": VIDEO_TASK,
    "import_video_by_id": VIDEO_TASK,
    "backup_video_to_s3": S3_TASK,
    "import_video_from_s3": S3_TASK,
    "perform_clustering": CLUSTER_TASK,
}

# List of tasks which can be called manually
MANUAL_VIDEO_TASKS = ['inception_index_by_id',
                      'inception_index_ssd_detection_by_id',
                      'perform_ssd_detection_by_id',
                      'perform_face_detection_indexing_by_id',
                      'perform_yolo_detection_by_id'
                      ]



POST_OPERATION_TASKS = {
    "extract_frames_by_id":['perform_ssd_detection_by_id','inception_index_by_id','perform_face_detection_indexing_by_id'],
    'perform_ssd_detection_by_id':['inception_index_ssd_detection_by_id',]
}

VISUAL_INDEXES = {
    'inception':
        {
            'indexer_task':"inception_index_by_id",
            'indexer_queue':Q_INDEXER,
            'retriever_task':"inception_query_by_image",
            'retriever_queue':Q_RETRIEVER,
            'detection_specific':False
        },
    'facenet':
        {
            'indexer_task': "perform_face_detection_indexing_by_id",
            'indexer_queue': Q_FACE_DETECTOR,
            'retriever_task':"facenet_query_by_image",
            'retriever_queue': Q_FACE_RETRIEVER,
            'detection_specific': True
        },
    }


if 'ALEX_ENABLE' in os.environ:
    POST_OPERATION_TASKS['extract_frames_by_id'].append('alexnet_index_by_id')
    VISUAL_INDEXES['alexnet'] = {
         'indexer_task': "alexnet_index_by_id",
         'indexer_queue': Q_INDEXER,
         'retriever_queue': Q_RETRIEVER,
         'detection_specific': False
    }


if 'YOLO_ENABLE' in os.environ:
    POST_OPERATION_TASKS['extract_frames_by_id'].append('perform_yolo_detection_by_id')