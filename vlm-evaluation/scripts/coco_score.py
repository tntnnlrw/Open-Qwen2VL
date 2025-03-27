from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import pdb
import copy
from collections import defaultdict
import numpy as np
import pdb
import math
import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import copy

# def precook(s, n=4, out=False):
#     """
#     Takes a string as input and returns an object that can be given to
#     either cook_refs or cook_test. This is optional: cook_refs and cook_test
#     can take string arguments as well.
#     :param s: string : sentence to be converted into ngrams
#     :param n: int    : number of ngrams for which representation is calculated
#     :return: term frequency vector for occuring ngrams
#     """
#     words = s.split()
#     counts = defaultdict(int)
#     for k in xrange(1,n+1):
#         for i in xrange(len(words)-k+1):
#             ngram = tuple(words[i:i+k])
#             counts[ngram] += 1
#     return counts

# def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
#     '''Takes a list of reference sentences for a single segment
#     and returns an object that encapsulates everything that BLEU
#     needs to know about them.
#     :param refs: list of string : reference sentences for some image
#     :param n: int : number of ngrams for which (ngram) representation is calculated
#     :return: result (list of dict)
#     '''
#     return [precook(ref, n) for ref in refs]

# def cook_test(test, n=4):
#     '''Takes a test sentence and returns an object that
#     encapsulates everything that BLEU needs to know about it.
#     :param test: list of string : hypothesis sentence for some image
#     :param n: int : number of ngrams for which (ngram) representation is calculated
#     :return: result (dict)
#     '''
#     return precook(test, n, True)

# class CiderScorer(object):
#     """CIDEr scorer.
#     """

#     def copy(self):
#         ''' copy the refs.'''
#         new = CiderScorer(n=self.n)
#         new.ctest = copy.copy(self.ctest)
#         new.crefs = copy.copy(self.crefs)
#         return new

#     def __init__(self, test=None, refs=None, n=4, sigma=6.0):
#         ''' singular instance '''
#         self.n = n
#         self.sigma = sigma
#         self.crefs = []
#         self.ctest = []
#         self.document_frequency = defaultdict(float)
#         self.cook_append(test, refs)
#         self.ref_len = None

#     def cook_append(self, test, refs):
#         '''called by constructor and __iadd__ to avoid creating new instances.'''

#         if refs is not None:
#             self.crefs.append(cook_refs(refs))
#             if test is not None:
#                 self.ctest.append(cook_test(test)) ## N.B.: -1
#             else:
#                 self.ctest.append(None) # lens of crefs and ctest have to match

#     def size(self):
#         assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
#         return len(self.crefs)

#     def __iadd__(self, other):
#         '''add an instance (e.g., from another sentence).'''

#         if type(other) is tuple:
#             ## avoid creating new CiderScorer instances
#             self.cook_append(other[0], other[1])
#         else:
#             self.ctest.extend(other.ctest)
#             self.crefs.extend(other.crefs)

#         return self
#     def compute_doc_freq(self):
#         '''
#         Compute term frequency for reference data.
#         This will be used to compute idf (inverse document frequency later)
#         The term frequency is stored in the object
#         :return: None
#         '''
#         for refs in self.crefs:
#             # refs, k ref captions of one image
#             for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
#                 self.document_frequency[ngram] += 1
#             # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

#     def compute_cider(self):
#         def counts2vec(cnts):
#             """
#             Function maps counts of ngram to vector of tfidf weights.
#             The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
#             The n-th entry of array denotes length of n-grams.
#             :param cnts:
#             :return: vec (array of dict), norm (array of float), length (int)
#             """
#             vec = [defaultdict(float) for _ in range(self.n)]
#             length = 0
#             norm = [0.0 for _ in range(self.n)]
#             for (ngram,term_freq) in cnts.iteritems():
#                 # give word count 1 if it doesn't appear in reference corpus
#                 df = np.log(max(1.0, self.document_frequency[ngram]))
#                 # ngram index
#                 n = len(ngram)-1
#                 # tf (term_freq) * idf (precomputed idf) for n-grams
#                 vec[n][ngram] = float(term_freq)*(self.ref_len - df)
#                 # compute norm for the vector.  the norm will be used for computing similarity
#                 norm[n] += pow(vec[n][ngram], 2)

#                 if n == 1:
#                     length += term_freq
#             norm = [np.sqrt(n) for n in norm]
#             return vec, norm, length

#         def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
#             '''
#             Compute the cosine similarity of two vectors.
#             :param vec_hyp: array of dictionary for vector corresponding to hypothesis
#             :param vec_ref: array of dictionary for vector corresponding to reference
#             :param norm_hyp: array of float for vector corresponding to hypothesis
#             :param norm_ref: array of float for vector corresponding to reference
#             :param length_hyp: int containing length of hypothesis
#             :param length_ref: int containing length of reference
#             :return: array of score for each n-grams cosine similarity
#             '''
#             delta = float(length_hyp - length_ref)
#             # measure consine similarity
#             val = np.array([0.0 for _ in range(self.n)])
#             for n in range(self.n):
#                 # ngram
#                 for (ngram,count) in vec_hyp[n].iteritems():
#                     # vrama91 : added clipping
#                     val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

#                 if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
#                     val[n] /= (norm_hyp[n]*norm_ref[n])

#                 assert(not math.isnan(val[n]))
#                 # vrama91: added a length based gaussian penalty
#                 val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
#             return val

#         # compute log reference length
#         self.ref_len = np.log(float(len(self.crefs)))

#         scores = []
#         for test, refs in zip(self.ctest, self.crefs):
#             # compute vector for test captions
#             vec, norm, length = counts2vec(test)
#             # compute vector for ref captions
#             score = np.array([0.0 for _ in range(self.n)])
#             for ref in refs:
#                 vec_ref, norm_ref, length_ref = counts2vec(ref)
#                 score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
#             # change by vrama91 - mean of ngram scores, instead of sum
#             score_avg = np.mean(score)
#             # divide by number of references
#             score_avg /= len(refs)
#             # multiply score by 10
#             score_avg *= 10.0
#             # append score of an image to the score list
#             scores.append(score_avg)
#         return scores

#     def compute_score(self, option=None, verbose=0):
#         # compute idf
#         self.compute_doc_freq()
#         # assert to check document frequency
#         assert(len(self.ctest) >= max(self.document_frequency.values()))
#         # compute cider score
#         score = self.compute_cider()
#         # debug
#         # print score
#         return np.mean(np.array(score)), np.array(score)

# class Cider:
#     """
#     Main Class to compute the CIDEr metric 

#     """
#     def __init__(self, test=None, refs=None, n=4, sigma=6.0):
#         # set cider to sum over 1 to 4-grams
#         self._n = n
#         # set the standard deviation parameter for gaussian penalty
#         self._sigma = sigma

#     def compute_score(self, gts, res):
#         """
#         Main function to compute CIDEr score
#         :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
#                 ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
#         :return: cider (float) : computed CIDEr score for the corpus 
#         """

#         assert(gts.keys() == res.keys())
#         imgIds = gts.keys()

#         cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

#         for id in imgIds:
#             hypo = res[id]
#             ref = gts[id]

#             # Sanity check.
#             assert(type(hypo) is list)
#             assert(len(hypo) == 1)
#             assert(type(ref) is list)
#             assert(len(ref) > 0)

#             cider_scorer += (hypo[0], ref)

#         (score, scores) = cider_scorer.compute_score()

#         return score, scores

#     def method(self):
#         return "CIDEr"

# class COCO:
#     def __init__(self, annotation_file=None):
#         """
#         Constructor of Microsoft COCO helper class for reading and visualizing annotations.
#         :param annotation_file (str): location of annotation file
#         :param image_folder (str): location to the folder that hosts images.
#         :return:
#         """
#         # load dataset
#         self.dataset = {}
#         self.anns = []
#         self.imgToAnns = {}
#         self.catToImgs = {}
#         self.imgs = []
#         self.cats = []
#         if not annotation_file == None:
#             print('loading annotations into memory...')
#             time_t = datetime.datetime.utcnow()
#             dataset = json.load(open(annotation_file, 'r'))
#             print(datetime.datetime.utcnow() - time_t)
#             self.dataset = dataset
#             self.createIndex()

#     def createIndex(self):
#         # create index
#         print('creating index...')
#         imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
#         anns =      {ann['id']:       [] for ann in self.dataset['annotations']}
#         for ann in self.dataset['annotations']:
#             imgToAnns[ann['image_id']] += [ann]
#             anns[ann['id']] = ann

#         imgs      = {im['id']: {} for im in self.dataset['images']}
#         for img in self.dataset['images']:
#             imgs[img['id']] = img

#         cats = []
#         catToImgs = []
#         if self.dataset['type'] == 'instances':
#             cats = {cat['id']: [] for cat in self.dataset['categories']}
#             for cat in self.dataset['categories']:
#                 cats[cat['id']] = cat
#             catToImgs = {cat['id']: [] for cat in self.dataset['categories']}
#             for ann in self.dataset['annotations']:
#                 catToImgs[ann['category_id']] += [ann['image_id']]

#         print('index created!')

#         # create class members
#         self.anns = anns
#         self.imgToAnns = imgToAnns
#         self.catToImgs = catToImgs
#         self.imgs = imgs
#         self.cats = cats

#     def info(self):
#         """
#         Print information about the annotation file.
#         :return:
#         """
#         for key, value in self.datset['info'].items():
#             print('%s: %s'%(key, value))

#     def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
#         """
#         Get ann ids that satisfy given filter conditions. default skips that filter
#         :param imgIds  (int array)     : get anns for given imgs
#                catIds  (int array)     : get anns for given cats
#                areaRng (float array)   : get anns for given area range (e.g. [0 inf])
#                iscrowd (boolean)       : get anns for given crowd label (False or True)
#         :return: ids (int array)       : integer array of ann ids
#         """
#         imgIds = imgIds if type(imgIds) == list else [imgIds]
#         catIds = catIds if type(catIds) == list else [catIds]

#         if len(imgIds) == len(catIds) == len(areaRng) == 0:
#             anns = self.dataset['annotations']
#         else:
#             if not len(imgIds) == 0:
#                 anns = sum([self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns],[])
#             else:
#                 anns = self.dataset['annotations']
#             anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
#             anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
#         if self.dataset['type'] == 'instances':
#             if not iscrowd == None:
#                 ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
#             else:
#                 ids = [ann['id'] for ann in anns]
#         else:
#             ids = [ann['id'] for ann in anns]
#         return ids

#     def getCatIds(self, catNms=[], supNms=[], catIds=[]):
#         """
#         filtering parameters. default skips that filter.
#         :param catNms (str array)  : get cats for given cat names
#         :param supNms (str array)  : get cats for given supercategory names
#         :param catIds (int array)  : get cats for given cat ids
#         :return: ids (int array)   : integer array of cat ids
#         """
#         catNms = catNms if type(catNms) == list else [catNms]
#         supNms = supNms if type(supNms) == list else [supNms]
#         catIds = catIds if type(catIds) == list else [catIds]

#         if len(catNms) == len(supNms) == len(catIds) == 0:
#             cats = self.dataset['categories']
#         else:
#             cats = self.dataset['categories']
#             cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
#             cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
#             cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
#         ids = [cat['id'] for cat in cats]
#         return ids

#     def getImgIds(self, imgIds=[], catIds=[]):
#         '''
#         Get img ids that satisfy given filter conditions.
#         :param imgIds (int array) : get imgs for given ids
#         :param catIds (int array) : get imgs with all given cats
#         :return: ids (int array)  : integer array of img ids
#         '''
#         imgIds = imgIds if type(imgIds) == list else [imgIds]
#         catIds = catIds if type(catIds) == list else [catIds]

#         if len(imgIds) == len(catIds) == 0:
#             ids = self.imgs.keys()
#         else:
#             ids = set(imgIds)
#             for catId in catIds:
#                 if len(ids) == 0:
#                     ids = set(self.catToImgs[catId])
#                 else:
#                     ids &= set(self.catToImgs[catId])
#         return list(ids)

#     def loadAnns(self, ids=[]):
#         """
#         Load anns with the specified ids.
#         :param ids (int array)       : integer ids specifying anns
#         :return: anns (object array) : loaded ann objects
#         """
#         if type(ids) == list:
#             return [self.anns[id] for id in ids]
#         elif type(ids) == int:
#             return [self.anns[ids]]

#     def loadCats(self, ids=[]):
#         """
#         Load cats with the specified ids.
#         :param ids (int array)       : integer ids specifying cats
#         :return: cats (object array) : loaded cat objects
#         """
#         if type(ids) == list:
#             return [self.cats[id] for id in ids]
#         elif type(ids) == int:
#             return [self.cats[ids]]

#     def loadImgs(self, ids=[]):
#         """
#         Load anns with the specified ids.
#         :param ids (int array)       : integer ids specifying img
#         :return: imgs (object array) : loaded img objects
#         """
#         if type(ids) == list:
#             return [self.imgs[id] for id in ids]
#         elif type(ids) == int:
#             return [self.imgs[ids]]

#     def showAnns(self, anns):
#         """
#         Display the specified annotations.
#         :param anns (array of object): annotations to display
#         :return: None
#         """
#         if len(anns) == 0:
#             return 0
#         if self.dataset['type'] == 'instances':
#             ax = plt.gca()
#             polygons = []
#             color = []
#             for ann in anns:
#                 c = np.random.random((1, 3)).tolist()[0]
#                 if type(ann['segmentation']) == list:
#                     # polygon
#                     for seg in ann['segmentation']:
#                         poly = np.array(seg).reshape((len(seg)/2, 2))
#                         polygons.append(Polygon(poly, True,alpha=0.4))
#                         color.append(c)
#                 else:
#                     # mask
#                     mask = COCO.decodeMask(ann['segmentation'])
#                     img = np.ones( (mask.shape[0], mask.shape[1], 3) )
#                     if ann['iscrowd'] == 1:
#                         color_mask = np.array([2.0,166.0,101.0])/255
#                     if ann['iscrowd'] == 0:
#                         color_mask = np.random.random((1, 3)).tolist()[0]
#                     for i in range(3):
#                         img[:,:,i] = color_mask[i]
#                     ax.imshow(np.dstack( (img, mask*0.5) ))
#             p = PatchCollection(polygons, facecolors=color, edgecolors=(0,0,0,1), linewidths=3, alpha=0.4)
#             ax.add_collection(p)
#         if self.dataset['type'] == 'captions':
#             for ann in anns:
#                 print(ann['caption'])

#     def loadRes(self, resFile):
#         """
#         Load result file and return a result api object.
#         :param   resFile (str)     : file name of result file
#         :return: res (obj)         : result api object
#         """
#         res = COCO()
#         res.dataset['images'] = [img for img in self.dataset['images']]
#         res.dataset['info'] = copy.deepcopy(self.dataset['info'])
#         res.dataset['type'] = copy.deepcopy(self.dataset['type'])
#         res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])

#         print('Loading and preparing results...     ')
#         time_t = datetime.datetime.utcnow()
#         anns    = json.load(open(resFile))
#         assert type(anns) == list, 'results in not an array of objects'
#         annsImgIds = [ann['image_id'] for ann in anns]
#         assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
#                'Results do not correspond to current coco set'
#         if 'caption' in anns[0]:
#             imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
#             res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
#             for id, ann in enumerate(anns):
#                 ann['id'] = id
#         elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 bb = ann['bbox']
#                 x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
#                 ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
#                 ann['area'] = bb[2]*bb[3]
#                 ann['id'] = id
#                 ann['iscrowd'] = 0
#         elif 'segmentation' in anns[0]:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 ann['area']=sum(ann['segmentation']['counts'][2:-1:2])
#                 ann['bbox'] = []
#                 ann['id'] = id
#                 ann['iscrowd'] = 0
#         print('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

#         res.dataset['annotations'] = anns
#         res.createIndex()
#         return res


#     @staticmethod
#     def decodeMask(R):
#         """
#         Decode binary mask M encoded via run-length encoding.
#         :param   R (object RLE)    : run-length encoding of binary mask
#         :return: M (bool 2D array) : decoded binary mask
#         """
#         N = len(R['counts'])
#         M = np.zeros( (R['size'][0]*R['size'][1], ))
#         n = 0
#         val = 1
#         for pos in range(N):
#             val = not val
#             for c in range(R['counts'][pos]):
#                 R['counts'][pos]
#                 M[n] = val
#                 n += 1
#         return M.reshape((R['size']), order='F')

#     @staticmethod
#     def encodeMask(M):
#         """
#         Encode binary mask M using run-length encoding.
#         :param   M (bool 2D array)  : binary mask to encode
#         :return: R (object RLE)     : run-length encoding of binary mask
#         """
#         [h, w] = M.shape
#         M = M.flatten(order='F')
#         N = len(M)
#         counts_list = []
#         pos = 0
#         # counts
#         counts_list.append(1)
#         diffs = np.logical_xor(M[0:N-1], M[1:N])
#         for diff in diffs:
#             if diff:
#                 pos +=1
#                 counts_list.append(1)
#             else:
#                 counts_list[pos] += 1
#         # if array starts from 1. start with 0 counts for 0
#         if M[0] == 1:
#             counts_list = [0] + counts_list
#         return {'size':      [h, w],
#                'counts':    counts_list ,
#                }

#     @staticmethod
#     def segToMask( S, h, w ):
#          """
#          Convert polygon segmentation to binary mask.
#          :param   S (float array)   : polygon segmentation mask
#          :param   h (int)           : target mask height
#          :param   w (int)           : target mask width
#          :return: M (bool 2D array) : binary mask
#          """
#          M = np.zeros((h,w), dtype=np.bool)
#          for s in S:
#              N = len(s)
#              rr, cc = polygon(np.array(s[1:N:2]), np.array(s[0:N:2])) # (y, x)
#              M[rr, cc] = 1
#          return M


# class COCOEvalCap:
#     def __init__(self, coco, cocoRes):
#         self.evalImgs = []
#         self.eval = {}
#         self.imgToEval = {}
#         self.coco = coco
#         self.cocoRes = cocoRes
#         self.params = {'image_id': coco.getImgIds()}

#     def evaluate(self):
#         imgIds = self.params['image_id']
#         # imgIds = self.coco.getImgIds()
#         gts = {}
#         res = {}
#         for imgId in imgIds:
#             gts[imgId] = self.coco.imgToAnns[imgId]
#             res[imgId] = self.cocoRes.imgToAnns[imgId]

#         # =================================================
#         # Set up scorers
#         # =================================================
#         print('tokenization...')
#         tokenizer = PTBTokenizer()
#         gts  = tokenizer.tokenize(gts)
#         res = tokenizer.tokenize(res)

#         # =================================================
#         # Set up scorers
#         # =================================================
#         print('setting up scorers...')
#         scorers = [
#             # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#             # (Meteor(),"METEOR"),
#             # (Rouge(), "ROUGE_L"),
#             (Cider(), "CIDEr"),
#             # (Spice(), "SPICE")
#         ]

#         # =================================================
#         # Compute scores
#         # =================================================
#         for scorer, method in scorers:
#             print('computing %s score...'%(scorer.method()))
#             score, scores = scorer.compute_score(gts, res)
#             if type(method) == list:
#                 for sc, scs, m in zip(score, scores, method):
#                     self.setEval(sc, m)
#                     self.setImgToEvalImgs(scs, gts.keys(), m)
#                     print("%s: %0.3f"%(m, sc))
#             else:
#                 self.setEval(score, method)
#                 self.setImgToEvalImgs(scores, gts.keys(), method)
#                 print("%s: %0.3f"%(method, score))
#         self.setEvalImgs()

#     def setEval(self, score, method):
#         self.eval[method] = score

#     def setImgToEvalImgs(self, scores, imgIds, method):
#         for imgId, score in zip(imgIds, scores):
#             if not imgId in self.imgToEval:
#                 self.imgToEval[imgId] = {}
#                 self.imgToEval[imgId]["image_id"] = imgId
#             self.imgToEval[imgId][method] = score

#     def setEvalImgs(self):
#         self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

def compute_cider(
    result_path,
    annotations_path,
):
    # create coco object and coco_result object
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    return coco_eval.eval


def postprocess_captioning_generation(predictions):
    return predictions.split("Output", 1)[0]

import sys
compute_cider(sys.argv[1], sys.argv[2])