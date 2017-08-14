

EXAMPLES = {
    0:{
  "process_type": "V",
  "tasks": [
    {
      "operation": "import_video_from_s3",
      "arguments": {
        "key": "007/video/1.mp4",
        "bucket": "visualdatanetwork",
        "next_tasks": [
          {
            "operation": "segment_video",
            "arguments": {
              "next_tasks": [
                {
                  "operation": "decode_video",
                  "arguments": {
                    "rate": 30,
                    "rescale": 0,
                    "next_tasks": [
                        {"operation": "perform_indexing", "arguments":
                          {"index": "inception",
                          "target": "frames",
                          "filters":"__parent__"
                          }
                        },
                          {"operation": "perform_detection", "arguments": {
                          "filters":"__parent__",
                          "detector":"coco",
                          "next_tasks":[
                              {"operation": "crop_regions_by_id",
                               "arguments": {
                                   "filters": {"event_id": "__parent_event__"},
                                   "next_tasks": [
                                       {"operation": "perform_indexing",
                                        "arguments": {
                                            "index": "inception",
                                            "target": "regions",
                                            "filters": {"event_id": "__grand_parent_event__", "w__gte": 50, "h__gte": 50}
                                        }
                                        }
                                  ]
                               }
                              }
                          ]
                          }
                          },
                        {"operation": "perform_detection", "arguments": {
                        "filters":"__parent__",
                        "detector":"face",
                        "next_tasks":[
                            {"operation": "crop_regions_by_id",
                             "arguments": {
                                 "resize":[182,182],
                                 "filters": {"event_id": "__parent_event__"},
                                 "next_tasks": [
                                     {"operation": "perform_indexing",
                                      "arguments": {
                                          "index": "facenet",
                                          "target": "regions",
                                          "filters": {"event_id": "__grand_parent_event__"}}
                                    }]
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
  ]
}
}