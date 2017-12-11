DEFAULT_SEGMENTS_BATCH_SIZE = 10  # how many video segments should we process at a time?
DEFAULT_FRAMES_BATCH_SIZE = 500  # How many frames/images in a dataset should we process at a time?
DEFAULT_RATE = 30 # Default video decoding 1 frame per 30 frames AND all i-frames


DEFAULT_PROCESSING_PLAN_VIDEO =[
    {'operation': 'perform_detection', 'arguments': {
        'filters':'__parent__',
        'detector':'coco',
        'next_tasks':[
            {'operation': 'perform_indexing',
             'arguments': {
                 'index': 'inception',
                 'target': 'regions',
                 'filters': {'event_id': '__parent_event__', 'w__gte': 50, 'h__gte': 50}
             }
             },
        ]}
     },
    {'operation': 'perform_detection', 'arguments': {
        'filters':'__parent__',
        'detector':'face',
        'next_tasks':[
            {'operation': 'perform_indexing',
             'arguments': {
                 'index': 'facenet',
                 'target': 'regions',
                 'filters': {'event_id': '__parent_event__'}
             }},
        ]}
     },
    {'operation': 'perform_indexing', 'arguments':
        {'index': 'inception',
         'target': 'frames',
         'filters':'__parent__'
     }},
    {'operation': 'perform_analysis', 'arguments':
        {'analyzer': 'tagger',
         'target': 'frames',
         'filters':'__parent__'
     }}
]

DEFAULT_PROCESSING_PLAN_DATASET = [
    {'operation': 'perform_detection', 'arguments': {
        'frames_batch_size': DEFAULT_FRAMES_BATCH_SIZE,
        'detector':'coco',
        'next_tasks':[
            {'operation': 'perform_indexing',
             'arguments': {
                 'index': 'inception',
                 'target': 'regions',
                 'filters': {'event_id': '__parent_event__', 'w__gte': 50, 'h__gte': 50}
             }
             },
        ]}
     },
    {'operation': 'perform_detection', 'arguments': {
        'detector':'face',
        'frames_batch_size': DEFAULT_FRAMES_BATCH_SIZE,
        'next_tasks':[
            {'operation': 'perform_indexing',
             'arguments': {
                 'index': 'facenet',
                 'target': 'regions',
                 'filters': {'event_id': '__parent_event__'}
             }
             },
        ]}
     },
    {'operation': 'perform_indexing', 'arguments':{
         'index': 'inception',
         'frames_batch_size': DEFAULT_FRAMES_BATCH_SIZE,
         'target': 'frames',
     }},
    {'operation': 'perform_analysis', 'arguments':
        {'analyzer': 'tagger',
         'target': 'frames',
         'frames_batch_size': DEFAULT_FRAMES_BATCH_SIZE,
         }}
]


DEFAULT_PROCESSING_PLAN_FRAMELIST = [
    {'operation': 'perform_detection', 'arguments': {
        'filters': {'event_id': '__parent_event__'},
        'detector':'coco',
        'next_tasks':[
            {'operation': 'perform_indexing',
             'arguments': {
                 'index': 'inception',
                 'target': 'regions',
                 'filters': {'event_id': '__parent_event__', 'w__gte': 50, 'h__gte': 50}
             }
             },
        ]}
     },
    {'operation': 'perform_detection', 'arguments': {
        'detector':'face',
        'filters': {'event_id': '__parent_event__'},
        'next_tasks':[
            {'operation': 'perform_indexing',
             'arguments': {
                 'index': 'facenet',
                 'target': 'regions',
                 'filters': {'event_id': '__parent_event__'}
             }
             },
        ]}
     },
    {'operation': 'perform_indexing', 'arguments':{
         'index': 'inception',
         'filters': {'event_id': '__parent_event__'},
         'target': 'frames',
     }},
    {'operation': 'perform_analysis', 'arguments':
        {'analyzer': 'tagger',
         'target': 'frames',
         'filters': {'event_id': '__parent_event__'},
         }}
]
