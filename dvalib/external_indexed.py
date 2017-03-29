from .indexer import InceptionIndexer
import os,shutil,json, boto3, subprocess, time
import numpy as np
import logging
from dvalib import approximate


class ExternalIndexed(object):

    def __init__(self,path):
        self.name = ""
        self.indexer = None
        self.bucket_name = ""
        self.path = path
        self.image_filenames = []
        self.feature_filenames = []

    def prepare(self,input_path):
        self.input_path = input_path
        try:
            shutil.rmtree(self.path)
        except:
            pass
        os.mkdir(self.path)
        for dname in ['images','detections','indexes','approximate']:
            try:
                os.mkdir('{}/{}'.format(self.path,dname))
            except:
                pass

    def push_to_s3(self):
        s3 = boto3.resource('s3')
        s3.create_bucket(Bucket=self.bucket_name)
        time.sleep(30) # wait for it to create the bucket
        upload = subprocess.Popen(args=["aws","s3","cp",".","s3://{}/".format(self.bucket_name),'--recursive'],cwd=self.path)
        upload.communicate()
        upload.wait()
        bucket_request_payment = s3.BucketRequestPayment(self.bucket_name)
        response = bucket_request_payment.put(RequestPaymentConfiguration={'Payer': 'Requester'})
        bucket_policy = s3.BucketPolicy(self.bucket_name)
        response = bucket_policy.put(Policy=json.dumps({"Version": "2012-10-17", "Statement": [
            {"Sid": "AddPerm", "Effect": "Allow", "Principal": "*", "Action": "s3:GetObject",
             "Resource": "arn:aws:s3:::{}/*".format(self.bucket_name)}]}))


class ProductsIndex(ExternalIndexed):

    def __init__(self,path):
        super(ProductsIndex, self).__init__(path=path)
        self.indexer = InceptionIndexer()
        self.name = "products"
        self.bucket_name = "aub3dvaproducts"
        self.data = None

    def prepare(self,input_path):
        super(ProductsIndex, self).prepare(input_path)
        features = []
        for k,v,s in os.walk(input_path):
            if s:
                for fname in s:
                    if fname.endswith('.jpg'):
                        infile = "{}/{}".format(k,fname)
                        outfile = "{}/images/{}".format(self.path,infile.split('gtin-')[1].replace('/',''))
                        shutil.copyfile(infile,outfile)
                        self.image_filenames.append(outfile.split('/')[-1])
                    else:
                        print "skipping {}".format(fname)
        for i,image_path in enumerate(self.image_filenames):
            features.append(self.indexer.apply("{}/images/{}".format(self.path,image_path)))
            if i % 100 == 0:
                feat_fname = "{}_{}.npy".format(self.indexer.name, i)
                with open("{}/indexes/{}".format(self.path, feat_fname), 'w') as feats:
                    np.save(feats, np.array(features))
                    self.feature_filenames.append(feat_fname)
                features = []
        feat_fname = "{}_{}.npy".format(self.indexer.name,i)
        with open("{}/indexes/{}".format(self.path,feat_fname), 'w') as feats:
            np.save(feats, np.array(features))
            self.feature_filenames.append(feat_fname)
        with open('{}/metadata.json'.format(self.path), 'w') as metadata:
            json.dump({'images':self.image_filenames,'features':self.feature_filenames}, metadata)

    def load_metadata(self):
        with open('{}/metadata.json'.format(self.path)) as metadata:
            temp = json.load(metadata)
        self.image_filenames = temp['images']
        self.feature_filenames = temp['features']

    def build_approximate(self):
        data = []
        self.load_metadata()
        for fname in self.feature_filenames:
            fname = "{}/indexes/{}".format(self.path,fname)
            vectors = np.load(fname)
            print fname
            data.append(vectors)
        data = np.concatenate(data).squeeze()
        logging.info("performing fit on {}".format(data.shape))
        lmdb_path = "{}/approximate/{}_lmdb".format(self.path, self.indexer.name)
        model_path = "{}/approximate/{}_model".format(self.path, self.indexer.name)
        approximate_model = approximate.ApproximateIndexer(self.indexer.name, model_path, lmdb_path)
        approximate_model.prepare(data)

class VisualGenomeIndex(ExternalIndexed):

    def __init__(self,path):
        super(VisualGenomeIndex, self).__init__(path=path)
        self.indexer = InceptionIndexer()
        self.name = "products"
        self.bucket_name = "aub3dvavisualgenome"

    def prepare(self,input_path):
        raise NotImplementedError

    def load_metadata(self):
        raise NotImplementedError

    def load_approximate(self):
        raise NotImplementedError

    def build_approximate(self):
        raise NotImplementedError


