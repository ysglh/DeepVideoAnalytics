

EXAMPLES = {
    0:{
  "process_type":"V",
  "tasks":[
      {
        "operation":"assign_label",
        "label":"test_1",
        "target":"frames",
        "filters":{"video_id":1,"w_gte":100}
      },
      {
        "operation":"assign_label",
        "label":"test_1",
        "target":"frames",
        "filters":{"video_id":1,"w_lt":100}
      }
    ]
    }
}