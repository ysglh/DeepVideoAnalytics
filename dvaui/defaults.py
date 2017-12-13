DEFAULT_SEGMENTS_BATCH_SIZE = 10  # how many video segments should we process at a time?
DEFAULT_FRAMES_BATCH_SIZE = 500  # How many frames/images in a dataset should we process at a time?
DEFAULT_RATE = 30  # Default video decoding 1 frame per 30 frames AND all i-frames

DEFAULT_PROCESSING_PLAN_VIDEO = [
    {'operation': 'perform_detection', 'arguments': {
        'filters': '__parent__',
        'detector': 'coco',
        'next_tasks': [
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
        'filters': '__parent__',
        'detector': 'face',
        'next_tasks': [
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
         'filters': '__parent__'
         }},
    {'operation': 'perform_analysis', 'arguments':
        {'analyzer': 'tagger',
         'target': 'frames',
         'filters': '__parent__'
         }}
]

DEFAULT_PROCESSING_PLAN_DATASET = [
    {'operation': 'perform_detection', 'arguments': {
        'frames_batch_size': DEFAULT_FRAMES_BATCH_SIZE,
        'detector': 'coco',
        'next_tasks': [
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
        'detector': 'face',
        'frames_batch_size': DEFAULT_FRAMES_BATCH_SIZE,
        'next_tasks': [
            {'operation': 'perform_indexing',
             'arguments': {
                 'index': 'facenet',
                 'target': 'regions',
                 'filters': {'event_id': '__parent_event__'}
             }
             },
        ]}
     },
    {'operation': 'perform_indexing', 'arguments': {
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
        'detector': 'coco',
        'next_tasks': [
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
        'detector': 'face',
        'filters': {'event_id': '__parent_event__'},
        'next_tasks': [
            {'operation': 'perform_indexing',
             'arguments': {
                 'index': 'facenet',
                 'target': 'regions',
                 'filters': {'event_id': '__parent_event__'}
             }
             },
        ]}
     },
    {'operation': 'perform_indexing', 'arguments': {
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

DEFAULT_MODELS = [
    {
        "model_type": "detector",
        "name": "coco",
        "detector_type": "T",
        "url": "https://www.dropbox.com/s/nzz26b2p4wxygg3/coco_mobilenet.pb",
        "filename": "coco_mobilenet.pb",
        "algorithm": "mobilenet_ssd",
        "mode": "T",
        "class_index_to_string": {
            "1": "person",
            "2": "bicycle",
            "3": "car",
            "4": "motorcycle",
            "5": "airplane",
            "6": "bus",
            "7": "train",
            "8": "truck",
            "9": "boat",
            "10": "traffic light",
            "11": "fire hydrant",
            "13": "stop sign",
            "14": "parking meter",
            "15": "bench",
            "16": "bird",
            "17": "cat",
            "18": "dog",
            "19": "horse",
            "20": "sheep",
            "21": "cow",
            "22": "elephant",
            "23": "bear",
            "24": "zebra",
            "25": "giraffe",
            "27": "backpack",
            "28": "umbrella",
            "31": "handbag",
            "32": "tie",
            "33": "suitcase",
            "34": "frisbee",
            "35": "skis",
            "36": "snowboard",
            "37": "sports ball",
            "38": "kite",
            "39": "baseball bat",
            "40": "baseball glove",
            "41": "skateboard",
            "42": "surfboard",
            "43": "tennis racket",
            "44": "bottle",
            "46": "wine glass",
            "47": "cup",
            "48": "fork",
            "49": "knife",
            "50": "spoon",
            "51": "bowl",
            "52": "banana",
            "53": "apple",
            "54": "sandwich",
            "55": "orange",
            "56": "broccoli",
            "57": "carrot",
            "58": "hot dog",
            "59": "pizza",
            "60": "donut",
            "61": "cake",
            "62": "chair",
            "63": "couch",
            "64": "potted plant",
            "65": "bed",
            "67": "dining table",
            "70": "toilet",
            "72": "tv",
            "73": "laptop",
            "74": "mouse",
            "75": "remote",
            "76": "keyboard",
            "77": "cell phone",
            "78": "microwave",
            "79": "oven",
            "80": "toaster",
            "81": "sink",
            "82": "refrigerator",
            "84": "book",
            "85": "clock",
            "86": "vase",
            "87": "scissors",
            "88": "teddy bear",
            "89": "hair drier",
            "90": "toothbrush"
        }
    },
    {
        "model_type": "detector",
        "name": "textbox",
        "url": "https://www.dropbox.com/s/ghxowdua65r82d6/checkpoint",
        "filename": "checkpoint",
        "additional_files": [
            {
                "url": "https://www.dropbox.com/s/v9w4mhcks7a719k/VGGnet_fast_rcnn_iter_50000.ckpt.data-00000-of-00001",
                "filename": "VGGnet_fast_rcnn_iter_50000.ckpt.data-00000-of-00001"
            },
            {
                "url": "https://www.dropbox.com/s/2d0licy8npype6r/VGGnet_fast_rcnn_iter_50000.ckpt.index",
                "filename": "VGGnet_fast_rcnn_iter_50000.ckpt.index"
            },
            {
                "url": "https://www.dropbox.com/s/vwmz2ra9jovlxjd/VGGnet_fast_rcnn_iter_50000.ckpt.meta",
                "filename": "VGGnet_fast_rcnn_iter_50000.ckpt.meta"
            }
        ],
        "algorithm": "CTPN",
        "mode": "T"
    },
    {
        "model_type": "detector",
        "name": "face",
        "url": None,
        "filename": None,
        "algorithm": "mtcnn_facenet",
        "mode": "T"
    },
    {
        "model_type": "indexer",
        "name": "inception",
        "url": "https://www.dropbox.com/s/fc7li2vwn8lvsyu/network.pb",
        "filename": "network.pb",
        "shasum": "48b026cf77dfbd5d9841cca3ee550ef0ee5a0751",
        "mode": "T"
    },
    {
        "model_type": "indexer",
        "name": "facenet",
        "url": "https://www.dropbox.com/s/jytpgw8et09ede9/facenet.pb",
        "filename": "facenet.pb",
        "shasum": "9f99caccbc75dcee8cb0a55a0551d7c5cb8a6836",
        "mode": "T"
    },
    {
        "model_type": "analyzer",
        "name": "crnn",
        "url": "https://www.dropbox.com/s/l0vo83hmvv2aipn/crnn.pth",
        "filename": "crnn.pth",
        "mode": "P"
    },
    {
        "model_type": "analyzer",
        "name": "tagger",
        "url": "https://www.dropbox.com/s/umo0xtljm9an90x/open_images.ckpt",
        "filename": "open_images.ckpt",
        "additional_files": [
            {
                "url": "https://www.dropbox.com/s/f93axdtlb3ltj40/open_images.ckpt.meta",
                "filename": "open_images.ckpt.meta"
            },
            {
                "url": "https://www.dropbox.com/s/2sd5nzpyhsj10w5/open_images_labelmap.txt",
                "filename": "open_images_labelmap.txt"
            },
            {
                "url": "https://www.dropbox.com/s/yboqv4leem6oy01/open_images_dict.csv",
                "filename": "open_images_dict.csv"
            }
        ],
        "mode": "T"
    }
]

DEFAULT_VDN_RESPONSE = [
    {
        "name": "Visual Data Network",
        "url": "http://www.visualdata.network/",
        "last_response_datasets": [
            {
                "description": "LFW_subset",
                "created": "2017-08-25T16:39:54.258687Z",
                "url": "http://www.visualdata.network/vdn/api/vdn_datasets/1/",
                "path": "https://www.dropbox.com/s/6nn84z4yzy47vuh/LFW.dva_export.zip",
                "requester_pays": False,
                "parent_url": "",
                "organization": {
                    "url": "http://www.visualdata.network/vdn/api/organizations/1/",
                    "description": "Default organization",
                    "user": "http://www.visualdata.network/api/users/1/",
                    "name": "Akshay Bhat",
                    "created": "2017-08-25T16:39:54.247384Z"
                },
                "name": "LFW_subset"
            },
            {
                "description": "MSCOCO_Sample_500",
                "created": "2017-08-25T16:39:54.276958Z",
                "url": "http://www.visualdata.network/vdn/api/vdn_datasets/2/",
                "path": "https://www.dropbox.com/s/qhzl9ig7yhems6j/MSCOCO_Sample.dva_export.zip",
                "requester_pays": False,
                "parent_url": "",
                "organization": {
                    "url": "http://www.visualdata.network/vdn/api/organizations/1/",
                    "description": "Default organization",
                    "user": "http://www.visualdata.network/api/users/1/",
                    "name": "Akshay Bhat",
                    "created": "2017-08-25T16:39:54.247384Z"
                },
                "name": "MSCOCO_Sample_500"
            },
            {
                "description": "Paris",
                "created": "2017-08-25T16:39:54.280653Z",
                "url": "http://www.visualdata.network/vdn/api/vdn_datasets/3/",
                "path": "https://www.dropbox.com/s/a7qf1f3j8vp4nuh/Paris.dva_export.zip",
                "requester_pays": False,
                "parent_url": "",
                "organization": {
                    "url": "http://www.visualdata.network/vdn/api/organizations/1/",
                    "description": "Default organization",
                    "user": "http://www.visualdata.network/api/users/1/",
                    "name": "Akshay Bhat",
                    "created": "2017-08-25T16:39:54.247384Z"
                },
                "name": "Paris"
            },
            {
                "description": "MSCOCO train ~14GB",
                "created": "2017-08-25T16:39:54.284367Z",
                "url": "http://www.visualdata.network/vdn/api/vdn_datasets/4/",
                "requester_pays": True,
                "parent_url": "",
                "organization": {
                    "url": "http://www.visualdata.network/vdn/api/organizations/1/",
                    "description": "Default organization",
                    "user": "http://www.visualdata.network/api/users/1/",
                    "name": "Akshay Bhat",
                    "created": "2017-08-25T16:39:54.247384Z"
                },
                "path": "s3://visualdatanetwork/coco_train.dva_export.zip",
                "name": "MSCOCO train ~14GB"
            },
            {
                "description": "aws_test_dir",
                "created": "2017-08-25T16:39:54.288147Z",
                "url": "http://www.visualdata.network/vdn/api/vdn_datasets/5/",
                "requester_pays": True,
                "parent_url": "",
                "organization": {
                    "url": "http://www.visualdata.network/vdn/api/organizations/1/",
                    "description": "Default organization",
                    "user": "http://www.visualdata.network/api/users/1/",
                    "name": "Akshay Bhat",
                    "created": "2017-08-25T16:39:54.247384Z"
                },
                "path": "s3://visualdatanetwork/007/",
                "name": "aws_test_dir"
            }
        ],
        "last_response_detectors": [
            {
                "description": "License plate",
                "created": "2017-08-25T16:39:54.292027Z",
                "url": "http://www.visualdata.network/vdn/api/vdn_detectors/1/",
                "path": "https://www.dropbox.com/s/ztsl59pxgzvd14k/1.dva_detector.zip",
                "requester_pays": False,
                "organization": {
                    "url": "http://www.visualdata.network/vdn/api/organizations/1/",
                    "description": "Default organization",
                    "user": "http://www.visualdata.network/api/users/1/",
                    "name": "Akshay Bhat",
                    "created": "2017-08-25T16:39:54.247384Z"
                },
                "name": "License plate"
            }
        ]
    }
]
