import django
import os, sys, glob
sys.path.append("../server/")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from django.core.files.uploadedfile import SimpleUploadedFile
from dvaui.view_shared import handle_uploaded_file
from dvaapp.models import Video, TEvent, DVAPQL, Retriever, TrainedModel
from django.conf import settings
from dvaapp.processing import DVAPQLProcess
from dvaapp.tasks import perform_dataset_extraction, perform_indexing, perform_export, perform_import, \
    perform_retriever_creation, perform_detection, \
    perform_video_segmentation, perform_transformation

if __name__ == '__main__':
    for fname in glob.glob('ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name)
    if settings.DEV_ENV:
        for fname in glob.glob('ci/*.zip'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
            handle_uploaded_file(f, name)
    for i, v in enumerate(Video.objects.all()):
        perform_import(TEvent.objects.get(video=v, operation='perform_import').pk)
        if v.dataset:
            arguments = {'sync': True}
            perform_dataset_extraction(TEvent.objects.create(video=v, arguments=arguments).pk)
        else:
            arguments = {'sync': True}
            perform_video_segmentation(TEvent.objects.create(video=v, arguments=arguments).pk)
        arguments = {'index': 'inception', 'target': 'frames'}
        perform_indexing(TEvent.objects.create(video=v, arguments=arguments).pk)
        if i == 1:  # save travis time by just running detection on first video
            # face_mtcnn
            arguments = {'detector': 'face'}
            dt = TEvent.objects.create(video=v, arguments=arguments)
            perform_detection(dt.pk)
            print "done perform_detection"
            arguments = {'filters': {'event_id': dt.pk}, }
            perform_transformation(TEvent.objects.create(video=v, arguments=arguments).pk)
            print "done perform_transformation"
            # coco_mobilenet
            arguments = {'detector': 'coco'}
            dt = TEvent.objects.create(video=v, arguments=arguments)
            perform_detection(dt.pk)
            print "done perform_detection"
            arguments = {'filters': {'event_id': dt.pk}, }
            perform_transformation(TEvent.objects.create(video=v, arguments=arguments).pk)
            print "done perform_transformation"
            # inception on crops from detector
            arguments = {'index': 'inception', 'target': 'regions',
                         'filters': {'event_id': dt.pk, 'w__gte': 50, 'h__gte': 50}}
            perform_indexing(TEvent.objects.create(video=v, arguments=arguments).pk)
            print "done perform_indexing"
            # assign_open_images_text_tags_by_id(TEvent.objects.create(video=v).pk)
        temp = TEvent.objects.create(video=v, arguments={'destination': "FILE"})
        perform_export(temp.pk)
        temp.refresh_from_db()
        fname = temp.arguments['file_name']
        f = SimpleUploadedFile(fname, file("{}/exports/{}".format(settings.MEDIA_ROOT, fname)).read(),
                               content_type="application/zip")
        print fname
        vimported = handle_uploaded_file(f, fname)
        perform_import(TEvent.objects.get(video=vimported, operation='perform_import').pk)
        # dc = Retriever()
        # args = {}
        # args['components'] = 32
        # args['m'] = 8
        # args['v'] = 8
        # args['sub'] = 64
        # dc.algorithm = Retriever.LOPQ
        # dc.source_filters = {'indexer_shasum': TrainedModel.objects.get(name="inception",model_type=TrainedModel.INDEXER).shasum}
        # dc.arguments = args
        # dc.save()
        # clustering_task = TEvent()
        # clustering_task.arguments = {'retriever_pk': dc.pk}
        # clustering_task.operation = 'perform_retriever_creation'
        # clustering_task.save()
        # perform_retriever_creation(clustering_task.pk)
        # query_dict = {
        #     'process_type': DVAPQL.QUERY,
        #     'image_data_b64': base64.encodestring(file('tests/queries/query.png').read()),
        #     'tasks': [
        #         {
        #             'operation': 'perform_indexing',
        #             'arguments': {
        #                 'index': 'inception',
        #                 'target': 'query',
        #                 'map': [
        #                     {'operation': 'perform_retrieval',
        #                      'arguments': {'count': 20, 'retriever_pk': Retriever.objects.get(name='inception').pk}
        #                      }
        #                 ]
        #             }
        #
        #         }
        #
        #     ]
        # }
        # launch_workers_and_scheduler_from_environment()
        # qp = DVAPQLProcess()
        # qp.create_from_json(query_dict)
        # qp.launch()
        # qp.wait()
