EXAMPLES = {
    0: {
        "process_type": "V",
        "tasks": [
            {
                "operation": "perform_import",
                "arguments": {
                    "source": "S3",
                    "key": "007/video/1.mp4",
                    "bucket": "visualdatanetwork",
                    "map": [
                        {
                            "operation": "perform_video_segmentation",
                            "arguments": {
                                "map": [
                                    {
                                        "operation": "perform_video_decode",
                                        "arguments": {
                                            "segments_batch_size": 10,
                                            "rate": 30,
                                            "rescale": 0,
                                            "map": [
                                                {"operation": "perform_indexing", "arguments":
                                                    {"index": "inception",
                                                     "target": "frames",
                                                     "filters": "__parent__"
                                                     }
                                                 },
                                                {"operation": "perform_detection", "arguments": {
                                                    "filters": "__parent__",
                                                    "detector": "coco",
                                                    "map": [
                                                        {"operation": "perform_indexing",
                                                         "arguments": {
                                                             "index": "inception",
                                                             "target": "regions",
                                                             "filters": {"event_id": "__parent_event__", "w__gte": 50,
                                                                         "h__gte": 50}
                                                         }
                                                         }
                                                    ]
                                                }
                                                 },
                                                {"operation": "perform_detection", "arguments": {
                                                    "filters": "__parent__",
                                                    "detector": "face",
                                                    "map": [
                                                        {"operation": "perform_indexing",
                                                         "arguments": {
                                                             "index": "facenet",
                                                             "target": "regions",
                                                             "filters": {"event_id": "__parent_event__"}}
                                                         }

                                                    ]
                                                }
                                                 }
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
}
