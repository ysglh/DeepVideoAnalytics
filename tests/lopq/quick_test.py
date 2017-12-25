# from dvaapp.models import Retriever, TrainedModel,TEvent,DVAPQL
# from dvaapp import processing
# spec = {
#     'process_type':DVAPQL.PROCESS,
#     'create':[{
#         'MODEL':'Retriever',
#         'spec':{
#             'algorithm':Retriever.LOPQ,
#             'arguments':{'components': 32, 'm': 8, 'v': 8, 'sub': 128},
#             'source_filters':{'indexer_shasum': TrainedModel.objects.get(name="inception",model_type=TrainedModel.INDEXER).shasum}
#         },
#         'tasks':[
#             {
#                 'operation':'perform_retriever_creation',
#                 'arguments':{'retriever_pk': '__pk__'}
#             }
#         ]
#     },]
# }
# p = processing.DVAPQLProcess()
# p.create_from_json(j=spec,user=None)
# p.launch()
# p.wait()
