from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import os
import warnings

from ..dataset import ImageDataset


class Tango2000(ImageDataset):
    """Tango2000.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    dataset_dir = 'tango2000'
    dataset_url = '' #TBD
    camid_counter = 0
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        
        gallery_paths = []
        gallery_pids = set()
        query_paths = []
        query_pids = set()
        for root,dirs,files in os.walk(self.data_dir):
            for idx,f in enumerate(files):
                if (idx >= len(files) - 2) and (len(files) > 2):
                    query_paths.append(osp.join(root,f))
                    query_pids.add(root.split('/')[-1])
                    continue
                gallery_paths.append(osp.join(root,f))
                gallery_pids.add(root.split('/')[-1])

        # allow alternative directory structure
        train = self.process_dir(query_paths + gallery_paths)
        query = self.process_dir(query_paths)
        gallery = self.process_dir(gallery_paths)
        
        self.validate(query_pids, gallery_pids)
        
        super(Tango2000, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, img_paths):
        data = []
        for img_path in img_paths:
            pid = int(osp.dirname(img_path).split('/')[-1])
            camid = self.camid_counter
            self.camid_counter += 1
            data.append((img_path, pid, camid))
        return data
    
    def validate(self, query_pids, gallery_pids):
        query_pids = list(query_pids)
        gallery_pids = list(gallery_pids)
        print([i for i in query_pids if i not in gallery_pids])
