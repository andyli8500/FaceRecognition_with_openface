import os
import sys
import numpy as np
import cPickle

import openface as of
import cv2
import time

cur_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cur_path, './util/')))
from lsh import LSH_hog, LSH_sift

rootDir = '/home/jiaxi/Documents/openface/openface'
facePredictor = os.path.join(rootDir, 'models/dlib', 'shape_predictor_68_face_landmarks.dat')
align = of.AlignDlib(facePredictor)

netDir = os.path.join(rootDir, 'models/openface', 'nn4.small2.v1.t7')
net = of.TorchNeuralNet(netDir, imgDim=96, cuda=True)

SETNAME = 'lfw_raw'


def embed_dist(x1, x2):
    diff = x1 - x2
    dist = np.dot(diff.T, diff)
    return dist


class LocalMatcher(object):
    def __init__(self, setname):
        self.trinet_index = self.load_triN('conf/{:s}_triN.pkl'.format(setname))

    def load_triN(self, pin):
        with open(pin) as fh:
            data = cPickle.load(fh)
        return data

    def match_triN(self, img):
        """Put the aligned face and net generated vector here"""
        # img = cv2.imread(img_dst)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bb = align.getAllFaceBoundingBoxes(img)
        bbs = bb if bb is not None else []
        person = []
        for bb in bbs:
            identities = []
            landmarks = align.findLandmarks(img, bb)
            alignedFace = align.align(96, img, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=of.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue
            rep = net.forward(alignedFace)
            phash = LSH_sift(rep)
            if phash in self.trinet_index:
                name = self.trinet_index[phash]['imgpath'].split('/')[-2].lower()

                pname = self.trinet_index[phash]['imgpath'].split('/')[-2].replace('_', ' ')
                #   identities.append((self.trinet_index[phash]['imgpath'], pname, 0.0, '0.0', urldat[name]))

                print('found in db!')

            for k, v in self.trinet_index.items():
                # name = v['imgpath'].split('/')[-2].lower()
                pname = v['imgpath'].split('/')[-2].replace('_', ' ')
                identities.append((v['imgpath'], pname, embed_dist(np.array(v['vec']), rep),
                                   '{:.3f}'.format(embed_dist(np.array(v['vec']), rep))))

            sorted_list = sorted(identities, key=lambda d: d[2])
            # print sorted_list[:20]
            # return sorted_list[0][1]
            if len(sorted_list) is 0:
                continue

            out_name = [[sorted_list[0][1], sorted_list[1][1], sorted_list[2][1]], bb]
            person.append(out_name)
        return person

    def search(self, dst_thum, debug=True):
        return self.match_triN(dst_thum)


def get_global_vars():
    index_alg = LocalMatcher(SETNAME)
    return index_alg


def main():
    # image = 'index.jpeg'
    # img = cv2.imread(image)
    #
    index_alg = get_global_vars()
    # person = index_alg.search(img)
    # for pp in person:
    #     name = pp[0]
    #     bb = pp[1]
    #
    # # bbs = align.getAllFaceBoundingBoxes(img)
    #
    #     b1 = (bb.left(), bb.bottom())
    #     tr = (bb.right(), bb.top())
    #     cv2.rectangle(img, b1, tr, color=(153, 255, 204), thickness=2)
    #     cv2.putText(img, name[0], (bb.left(), bb.bottom() + 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
    #     cv2.putText(img, name[1], (bb.left(), bb.bottom() + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
    #     cv2.putText(img, name[2], (bb.left(), bb.bottom() + 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
    #
    # cv2.imshow('img', img)
    # cv2.imwrite('index_out.jpeg', img)
    # cv2.waitKey(0)
    # img = cv2.imread(image)

    video = 'Jennifer_Aniston_0.mp4'

    cap = cv2.VideoCapture(video)

    idx = 0
    frames = []
    while(cap.isOpened()):
        print 'Processing:', idx
        ret, frame = cap.read()
        if frame is None:
            break

        frames.append(frame)

        idx += 1
    cap.release()

    for i, frame in enumerate(frames):
        person = index_alg.search(frame)
        st = time.time() * 1000
        if len(person) is not 0:
            for pp in person:
                name = pp[0]
                bb = pp[1]
                b1 = (bb.left(), bb.bottom())
                tr = (bb.right(), bb.top())
                cv2.rectangle(frame, b1, tr, color=(153, 255, 204), thickness=2)
                cv2.putText(frame, name[0], (bb.left(), bb.bottom()+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 128, 0))
                cv2.putText(frame, name[1], (bb.left(), bb.bottom()+30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 128, 0))
                cv2.putText(frame, name[2], (bb.left(), bb.bottom()+45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 128, 0))
        t = time.time() * 1000 - st
        print 'frame', i, ':', t, 's.'
        # cv2.imshow('frame', frame)
        cv2.imwrite('output/{}.jpg'.format(i), frame)
        # cv2.waitKey(1)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
