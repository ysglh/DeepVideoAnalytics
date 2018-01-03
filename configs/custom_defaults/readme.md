# Custom defaults

Mount a docker volume with your version of models, processing flow and initial process in this directory.

- On linux replace/edit trained_models.json to edit model download urls.

- To change which tasks are run when video/dataset/framelist is uploaded/imported/download change video/framelist/dataset_processing.json

- You can specify a startup DVAPQL process by setting INIT_PROCESS=/root/DVA/configs/custom_defaults/init_process.json
  and replacing/editing init_process.json