#!/usr/bin/env python3
# This version uses 70³ 3-channel SHN shnData as input/src data.
################################################################################
# ----- IMPORT --------------------------------------------------------------- #
################################################################################

import sys, os, time
import numpy as np
import tensorflow as tf
import re
from collections import defaultdict
import random
import mrcfile
import argparse
from datetime import date, datetime
import warnings
import operator
from scipy import ndimage                   # rescaling of 3D volumes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1: no info, 2 no warning
warnings.filterwarnings("ignore")

# These global variables are ugly, but they work
weightArray = None
debugFlag   = None
augFlag     = None



################################################################################
# ----- HELPER FUNCTIONS ----------------------------------------------------- #
################################################################################

def saveMRC(selectedData, voxelSize, cellVec, cellA_XYZ, filename, zip=True):
    with mrcfile.new(filename, overwrite=True, compression='gzip' if zip else None) as outmrc:
        outmrc.set_data(selectedData)
        outmrc.set_volume()
        outmrc.voxel_size = (voxelSize[0], voxelSize[1], voxelSize[2])
        outmrc.header.nxstart, outmrc.header.nystart, outmrc.header.nzstart = cellVec[0], cellVec[1], cellVec[2]
        outmrc.header.cella.x, outmrc.header.cella.y, outmrc.header.cella.z = cellA_XYZ



################################################################################
# ----- CNN MODEL ------------------------------------------------------------ #
################################################################################

def unet_model_fn(features, labels, mode):
    # on-device augmentation of features and labels while training
    if mode == tf.estimator.ModeKeys.TRAIN:
        if augFlag:
            # ROTATION IN 90° STEPS
            # ---------------------
            # Axis rotation around BC-Plane (=A-Axis):
            #  > a,b,c ---> a,c,-b ---> a,-b,-c ---> a,-c,b
            #
            ############################################################################################
            #  Empty     # Rotation along ZX-Plane (=Y-Axis)                 # Rotation along ZY-Plane #
            #  Example   ###############################################################################
            #  Cube      #   Z, Y, X  #   X, Y,-Z  #  -Z, Y,-X  #  -X, Y, Z  #   Y,-Z, X  #  -Y, Z, X  #
            ############################################################################################
            #    *----*  #    *----*  #    *----*  #    *----*  #    *----*  #    *----*  #    *----*  #
            # Z /|   /|  #   /%%%%/|  #   /|   /|  #   /|   /|  #   /|   /|  #   /|   /|  #   /|%%%/|  #
            # ^*----* |Y #  *----* |  #  *----*%|  #  *----* |  #  *----* |  #  *----* |  #  *----*%|  #
            #  | *--|-*  #  | *--|-*  #  | *--|%*  #  | *--|-*  #  |%*--|-*  #  |%%%%|-*  #  | *--|-*  #
            #  |/   |/   #  |/   |/   #  |/   |/   #  |/%%%|/   #  |/   |/   #  |%%%%|/   #  |/   |/   #
            #  *----*>X  #  *----*    #  *----*    #  *----*    #  *----*    #  *----*    #  *----*    #
            ############################################################################################
            #
            # Rotation along XY-Plane (=Z-Axis)
            # ---------------------------------
            #          0°    90°   180°   270°
            # Top    ZYX -> ZXy -> Zyx -> ZxY
            # Right  XYz -> Xzy -> XyZ -> XZY
            # Bottom zYx -> zxy -> zyX -> zXY
            # Left   xYZ -> xZy -> xyz -> xzY
            # Front  YzX -> YXZ -> YZx -> Yxz
            # Back   yZX -> yXz -> yzx -> yxZ

            def fnRotTop(inT):    return inT # identity
            def fn5RotRight(inT): return tf.transpose(tf.reverse(inT, [1]), [0, 3, 2, 1, 4]) # rotate ZX
            def fn5RotLeft(inT):  return tf.transpose(tf.reverse(inT, [3]), [0, 3, 2, 1, 4])
            def fn5RotFront(inT): return tf.transpose(tf.reverse(inT, [1]), [0, 2, 1, 3, 4]) # rotate ZY
            def fn5RotBack(inT):  return tf.transpose(tf.reverse(inT, [2]), [0, 2, 1, 3, 4])
            def fn4RotRight(inT): return tf.transpose(tf.reverse(inT, [1]), [0, 3, 2, 1]) # rotate ZX
            def fn4RotLeft(inT):  return tf.transpose(tf.reverse(inT, [3]), [0, 3, 2, 1])
            def fn4RotFront(inT): return tf.transpose(tf.reverse(inT, [1]), [0, 2, 1, 3]) # rotate ZY
            def fn4RotBack(inT):  return tf.transpose(tf.reverse(inT, [2]), [0, 2, 1, 3])
            def fnRotBottom(inT): return tf.reverse(inT, [1, 3]) # inverse ZX

            faceSelector = tf.random_uniform([1], minval=0, maxval=6, dtype=tf.int32)

            faceRotatedX, faceRotatedY, faceRotatedM = tf.case({
                tf.equal(faceSelector[0], 0): lambda: (   fnRotTop(features["x"]),    fnRotTop(labels["y"]),    fnRotTop(labels["m"])),
                tf.equal(faceSelector[0], 1): lambda: (fn4RotRight(features["x"]), fn5RotRight(labels["y"]), fn4RotRight(labels["m"])),
                tf.equal(faceSelector[0], 2): lambda: ( fn4RotLeft(features["x"]),  fn5RotLeft(labels["y"]),  fn4RotLeft(labels["m"])),
                tf.equal(faceSelector[0], 3): lambda: (fn4RotFront(features["x"]), fn5RotFront(labels["y"]), fn4RotFront(labels["m"])),
                tf.equal(faceSelector[0], 4): lambda: ( fn4RotBack(features["x"]),  fn5RotBack(labels["y"]),  fn4RotBack(labels["m"])),
                tf.equal(faceSelector[0], 5): lambda: (fnRotBottom(features["x"]), fnRotBottom(labels["y"]), fnRotBottom(labels["m"])),
            }, exclusive=False)

            def fnZ000(inT): return inT  # identity
            def fn5Z090(inT): return tf.transpose(tf.reverse(inT, [2]), [0, 1, 3, 2, 4])
            def fn5Z270(inT): return tf.transpose(tf.reverse(inT, [3]), [0, 1, 3, 2, 4])
            def fn4Z090(inT): return tf.transpose(tf.reverse(inT, [2]), [0, 1, 3, 2])
            def fn4Z270(inT): return tf.transpose(tf.reverse(inT, [3]), [0, 1, 3, 2])
            def fnZ180(inT): return tf.reverse(inT, [2, 3])

            zAxisRot = tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32)

            rotatedInX, rotatedInY, rotatedInM = tf.case({
                tf.equal(zAxisRot[0], 0): lambda: ( fnZ000(faceRotatedX),  fnZ000(faceRotatedY),  fnZ000(faceRotatedM)),
                tf.equal(zAxisRot[0], 1): lambda: (fn4Z090(faceRotatedX), fn5Z090(faceRotatedY), fn4Z090(faceRotatedM)),
                tf.equal(zAxisRot[0], 2): lambda: ( fnZ180(faceRotatedX),  fnZ180(faceRotatedY),  fnZ180(faceRotatedM)),
                tf.equal(zAxisRot[0], 3): lambda: (fn4Z270(faceRotatedX), fn5Z270(faceRotatedY), fn4Z270(faceRotatedM)),
            }, exclusive=False)

            # TRANSLATION & CROPPING
            # ---------------------
            cropLen = 30
            cropOff = 0

            # generate cropping vectors
            # Separate a single uniform distribution into 3 vector components
            # X = randomN % 30
            # Y = randomN // 30 % 30
            # Z = randomN // 30**2
            #randomV = tf.tile(tf.random_uniform([1], minval=0, maxval=cropLen**3, dtype=tf.int32), [3])
            #randomD = tf.floordiv(randomV, tf.constant([      1, cropLen, cropLen**2]))
            #randomM = tf.floormod(randomD, tf.constant([cropLen, cropLen,          1]))

            cropX = tf.concat([tf.zeros([1,], dtype=tf.int32), tf.random_uniform([3], minval=cropOff, maxval=cropLen, dtype=tf.int32)], axis=0)
            #cropX = tf.concat([tf.zeros([1,], dtype=tf.int32), randomM], axis=0)
            cropY = tf.concat([tf.add(cropX, [0,10,10,10]), tf.zeros([1, ], dtype=tf.int32)], axis=0)

            # TO THE CROPPING BLOCK! OFF WITH HIS HEAD!
            in_x = tf.slice(rotatedInX, cropX, (rotatedInX.shape[0], 40, 40, 40))
            in_y = tf.slice(rotatedInY, cropY, (rotatedInY.shape[0], 20, 20, 20, 4))
            in_m = tf.slice(rotatedInM, cropY[0:4], (rotatedInM.shape[0], 20, 20, 20))

        else: # simple center crop
            in_x = features["x"][-1, 15:55, 15:55, 15:55]
            in_y = labels["y"][-1, 25:45, 25:45, 25:45, : ]
            in_m = labels["m"][-1, 25:45, 25:45, 25:45]

    elif mode == tf.estimator.ModeKeys.EVAL:
        # simple center crop
        in_x = features["x"][-1, 15:55, 15:55, 15:55]
        in_y = tf.reshape(labels["y"][-1, 25:45, 25:45, 25:45, :], [-1, 20, 20, 20, 4])
        in_m = tf.reshape(labels["m"][-1, 25:45, 25:45, 25:45], [-1, 20, 20, 20])

    else: # predict
        in_x = features["x"]
        """if labels is not None:
            in_y = labels["y"]
            in_m = labels["m"]
        else:"""
        in_y = None
        in_m = None

    """print("DBG_X: {}".format(in_x.shape))
    if in_y is not None:
        print("DBG_Y: {}".format(in_y.shape))
    if in_m is not None:
        print("DBG_M: {}".format(in_m.shape))"""

    input_layer = tf.reshape(in_x, [-1, 40, 40, 40, 1])

    ##### LEVEL 1

    # > 40x40x40 | 1
    # Conv1_1
    conv1_1 = tf.layers.conv3d(
		inputs=input_layer,
	    filters=32,
	    kernel_size=[3, 3, 3],
	    padding="valid",
	    activation=tf.nn.relu)

    # > 38x38x38 | 32
    # Conv1_2
    conv1_2 = tf.layers.conv3d(
	    inputs=conv1_1,
	    filters=64,
	    kernel_size=[3, 3, 3],
	    padding="valid",
	    activation=tf.nn.relu)

    # > 36x36x36 | 64
    # Pool1
    pool1 = tf.layers.max_pooling3d(inputs=conv1_2, pool_size=2, strides=2)


    ##### LEVEL 2

    # > 18x18x18 | 64
    # Conv2_1
    conv2_1 = tf.layers.conv3d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3, 3],
        padding="valid",
        activation=tf.nn.relu)

    # > 16x16x16 | 128
    # Pool2
    pool2 = tf.layers.max_pooling3d(inputs=conv2_1, pool_size=2, strides=2)

    ##### LEVEL 3

    # TODO:
    # - Check if fully connected OR convolution are better
    """
    # > 8x8x8 | 128 => 65536
    pool2_flat = tf.reshape(pool2, [-1, (8 ** 3) * 128])

    dense3 = tf.layers.dense(inputs=pool2_flat, units=(8 ** 3) * 16, activation=tf.nn.relu)

    # > 32768 => 8x8x8 | 64
    dense3_cube = tf.reshape(dense3, [-1, 8, 8, 8, 16])
    """

    # test: convolution

    # > 8x8x8 | 128 => 65536
    conv3 = tf.layers.conv3d(inputs=pool2,
                             filters=256,
                             kernel_size=[3, 3, 3],
                             padding="same",
                             activation=tf.nn.relu)

    # upconvoltion!
    # > 8x8x8 | 256
    uconv3 = tf.layers.conv3d_transpose(inputs=conv3,
                                        filters=256,
                                        kernel_size=2,
                                        strides=2,
                                        padding="valid",
                                        activation=tf.nn.relu,
                                        data_format='channels_last',
                                        use_bias=None)

    ###### LEVEL 4 (upconv)

    #print(str(uconv3.get_shape()))
    # > 16x16x16 | 128 + 16x16x16 | 256
    ccat4 = tf.concat([conv2_1, uconv3], axis=4)
    #print(str(ccat4.get_shape()) + ":" + str(ccat4.dtype))

    # > 16x16x16 | 384
    conv4_1 = tf.layers.conv3d(inputs=ccat4,
                               filters=256,
                               kernel_size=[3, 3, 3],
                               padding="valid",
                               activation=tf.nn.relu)

    # > 14x14x14 | 256
    conv4_2 = tf.layers.conv3d(inputs=conv4_1,
                               filters=128,
                               kernel_size=[3, 3, 3],
                               padding="valid",
                               activation=tf.nn.relu)

    # upconvoltion!
    # > 12x12x12 | 128
    uconv4 = tf.layers.conv3d_transpose(inputs=conv4_2,
                                        filters=128,
                                        kernel_size=[2, 2, 2],
                                        strides=2,
                                        padding="valid",
                                        activation=tf.nn.relu,
                                        data_format='channels_last',
                                        use_bias=None)

    ###### LEVEL 5 (upconv)
    #print(str(uconv4.get_shape()))

    # crop conv1_2 to central 24x24x24
    crop5 = tf.slice(conv1_2, [0, 6, 6, 6, 0], [-1, 24, 24, 24, 64])
    #print(str(crop5.get_shape()))

    # concatenate
    # > 24x24x24 | 64 + 24x24x24 | 128
    ccat5 = tf.concat([crop5, uconv4], axis=4)
    #print(str(ccat5.get_shape()))

    # > 24x24x24 | 196
    conv5_1 = tf.layers.conv3d(inputs=ccat5,
                               filters=128,
                               kernel_size=[3, 3, 3],
                               padding="valid",
                               activation=tf.nn.relu)

    # > 22x22x22 | 128
    #conv5_2
    logits = tf.layers.conv3d(inputs=conv5_1,
                              filters=4,
                              kernel_size=[3, 3, 3],
                              padding="valid")
                              #activation=tf.nn.relu)
                                   #activation=tf.nn.sigmoid)

    # output: 20x20x20 | 4
    predictions = {
        "classes": tf.argmax(input=logits, axis=-1), # predicted classes (PREDICT/EVAL)
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor", axis=-1)
    }

    ##### PREDICTION/EVALUATION
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    ## generate weight mask:
    # 1) Combine labels and one-hot predictions to single multi-hot-mask
    # 2) Multiply multi-hot-mask with weight array
    # 3) Get the maximum channel value
    # -> If something was (mis-)-labeled it now has the maximum weight of what it was (mis)-labeled as.
    # E.g. sheet labeled as empty -> sheet weight. (~90)
    # empty recognized as emopty -> empty weight (1)
    # 4) Multiply with boolean training mask.
    onehotPredictions = tf.one_hot(predictions["classes"], depth=4, on_value=1, off_value=0, dtype=tf.int32, axis=-1)
    modfWeights = tf.multiply(tf.reduce_max(tf.multiply(tf.maximum(in_y, onehotPredictions), [16,16,16,1]), axis=-1), tf.to_int32(in_m)) # 16,16,16 for 050818

    # calculate loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=in_y, logits=logits, weights=modfWeights)

    eval_metric_ops = {}

    # calculate accuracy
    argmaxLabels = tf.argmax(in_y, axis=-1)
    floatArgMaxSize = tf.to_float(tf.size(predictions["classes"]))
    with tf.name_scope('accuracy'):
        accuracyTotal = tf.summary.scalar('accuracy', tf.divide(tf.to_float(tf.count_nonzero(tf.equal(argmaxLabels, predictions["classes"]))), floatArgMaxSize))
        accuracyDict = {}
        for i, type in enumerate(['sheet', 'helix', 'nucls', 'empty']):
            argmaxOfType = tf.boolean_mask(tf.equal(argmaxLabels, i), in_m)
            predicOfType = tf.boolean_mask(tf.equal(predictions["classes"], i), in_m)

            count_tp = tf.to_float(tf.count_nonzero(tf.logical_and(               argmaxOfType,                predicOfType)))
            count_fp = tf.to_float(tf.count_nonzero(tf.logical_and(tf.logical_not(argmaxOfType),               predicOfType)))
            count_fn = tf.to_float(tf.count_nonzero(tf.logical_and(               argmaxOfType, tf.logical_not(predicOfType))))
            sum_tp_fp_fn = count_tp + count_fp + count_fn

            score_tp = count_tp / sum_tp_fp_fn
            score_fp = count_fp / sum_tp_fp_fn
            score_fn = count_fn / sum_tp_fp_fn

            accuracyDict[type] = tf.summary.scalar('{}_truePositive'.format(type),  score_tp)
            accuracyDict[type] = tf.summary.scalar('{}_falsePositive'.format(type), score_fp)
            accuracyDict[type] = tf.summary.scalar('{}_falseNegative'.format(type), score_fn)

            if mode == tf.estimator.ModeKeys.EVAL:
                expScore_tp = tf.expand_dims(score_tp, axis=0)
                expScore_fp = tf.expand_dims(score_fp, axis=0)
                expScore_fn = tf.expand_dims(score_fn, axis=0)
                maskedScore_tp = tf.boolean_mask(expScore_tp, tf.logical_not(tf.is_nan(expScore_tp)))
                maskedScore_fp = tf.boolean_mask(expScore_fp, tf.logical_not(tf.is_nan(expScore_fp)))
                maskedScore_fn = tf.boolean_mask(expScore_fn, tf.logical_not(tf.is_nan(expScore_fn)))
                eval_metric_ops = {**eval_metric_ops,
                    "accuracy/{}_truePositive".format(type):  tf.metrics.mean(maskedScore_tp),
                    "accuracy/{}_falsePositive".format(type): tf.metrics.mean(maskedScore_fp),
                    "accuracy/{}_falseNegative".format(type): tf.metrics.mean(maskedScore_fn),
                }

    ##### EVALUATE
    if mode == tf.estimator.ModeKeys.EVAL:
        in_y_label = tf.argmax(input=in_y, axis=-1)
        eval_metric_ops["accuracy/accuracy"] = tf.metrics.accuracy(labels=in_y_label, predictions=predictions["classes"], name='acc_op')
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    ##### TRAIN
    # Configure the Training Op (for TRAIN mode)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # ADAM is great (see http://ruder.io/optimizing-gradient-descent/ for an explanation of various optimizer types)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=0.1)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



################################################################################
# ----- ARGPARSE ------------------------------------------------------------- #
################################################################################

parser = argparse.ArgumentParser(description="HARUSPEX V2019-01-16")
parser.add_argument('-s', '--steps',    metavar='<STEPS>', default=30000, type=int, action='store', help="Set steps.")
parser.add_argument('-n', '--network',  default=os.path.join(os.path.dirname(sys.argv[0]), str(date.today())), metavar='<PATH>', action='store', help="Set UNET path.")
parser.add_argument('-p', '--datasrc',  default=None, metavar='<PATH>', action='store', help="Set data path.")
parser.add_argument('-o', '--output',   default="./", metavar='<PATH>', action='store', help="Output directory for evaluation/listing.")
parser.add_argument('-L', '--limit',    metavar='<LIMIT>', type=int, nargs='+', action='store', help="Limit the number of secondary structure views.")
parser.add_argument('-S', '--seed',     metavar='<SEED>',  type=int, action='store', help="Set integer seed for shuffling and init.")
parser.add_argument('-M', '--metrics',  default=False, const=True,  action='store_const', help="Enable full metric logging (TensorBoard).")
parser.add_argument('-O', '--original', default=False, const=True,  action='store_const', help="Disable input data augmentation.")
parser.add_argument('-R', '--reverse',  default=True,  const=False, action='store_const', help="Reverse sorting for secondary structure views.")
parser.add_argument('-Z', '--no-zoom',  default=False, const=True,  action='store_const', help="Do not rescale voxel size to [1.0; 1.2] Angstrom.")
parser.add_argument('-r', '--random',   default=False, const=True,  action='store_const', help="Randomly shuffle views before applying limit.")
parser.add_argument('-d', '--density',  default=False, const=True,  action='store_const', help="Multiply predictions with density before saving.")
parser.add_argument('-m', '--nomrc',    default=False, const=True,  action='store_const', help="Do not save mrc output during map-prediction.")
parser.add_argument('-V', '--vectors',  default=False, const=True,  action='store_const', help="Save slice vectors as .json file.")

parser.add_argument('command', type=str, help="Command.")
parser.add_argument('dataset', type=str, help="For prediction: Source file or list of mrc files. For training: Dataset directory or list.", default=None)

args = parser.parse_args() #(sys.argv[1:])

args.dataset = os.path.abspath(os.path.expanduser(args.dataset))
args.network = os.path.abspath(os.path.expanduser(args.network))
args.output  = os.path.abspath(os.path.expanduser(args.output))
if args.datasrc: args.datasrc = os.path.abspath(os.path.expanduser(args.datasrc))

if not args.datasrc:
    args.datasrc = args.dataset



################################################################################
# ----- MAIN ----------------------------------------------------------------- #
################################################################################

def main(argv):
    # enable debugging
    if args.metrics:
        global debugFlag
        debugFlag = True

    # enable data augmentation
    if not args.original:
        global augFlag
        augFlag = True

    # list source/marked data
    viewList = []

    # handle directories and list-files
    if args.command.lower() not in ['map-predict', 'list-predict']:
        if os.path.isdir(args.dataset):
            print("listing source/marked data views...")
            regexp = re.compile(r"^(\d+)-([0-9a-z]+)_view-(\d+)_([^_]*)_(marked).npz$") # for every marked there's a source

            for fileIt in os.listdir(args.dataset):
                mObj = regexp.match(fileIt)
                if mObj:
                    emdbid, pdbid, sdev, ident, type = mObj.groups()
                    viewList.append((emdbid, pdbid, int(sdev), ident))

        elif os.path.isfile(args.dataset):
            print("loading view list...")
            regexp = re.compile(r"^VIEW (\d+) ([0-9a-z]+) (\d+) ([^_]*)$")
            with open(args.dataset, 'r') as listf:
                lineList = [x.strip() for x in listf.read().splitlines() if x[0] != '#']
                regxList = [regexp.match(x).groups() for x in lineList if x]
                viewList = [(x[0], x[1], int(x[2]), x[3]) for x in regxList if x is not None]

        print(" -> {} views found".format(len(viewList)))
        if not viewList:
            exit(1)

        if args.seed:
            print("Using seed '{}'...".format(args.seed))
            random.seed(args.seed)

        if not args.random:
            print("Sorting data views...")
            viewList.sort(key=operator.itemgetter(2), reverse=args.reverse)
        else:
            print("Shuffling data views...")
            random.shuffle(viewList)

        print(" -> top    : {}".format(viewList[0]))
        print(" -> bottom : {}".format(viewList[-1]))

        if args.limit:
            offset, length = 0, -1
            if len(args.limit) == 1:
                length = int(args.limit[0])
            else:
                offset = int(args.limit[0])
                length = int(args.limit[1])
            print("cropping views to [{}; {}[...".format(offset, offset+length))
            viewList = viewList[offset:offset+length]
            print(" -> top    : {}".format(viewList[0]))
            print(" -> bottom : {}".format(viewList[-1]))

    if args.command.lower()[0:4] == 'list' and viewList:
        if args.command.lower() == 'list-save':
            os.makedirs(args.output, exist_ok=True)
            listPath = os.path.join(args.output, (os.path.basename(args.dataset) or os.path.basename(os.path.dirname(args.dataset))) + '.txt')
            print("writing listed files to '{}'...".format(listPath))
            with open(listPath, 'w') as listf:
                listf.write("# RUN {}".format(datetime.now().replace(microsecond=0)) + os.linesep)
                listf.write("# ARG {}".format(' '.join(sys.argv)) + os.linesep)
                listf.write("# FORMAT <TYPE> <EMDB> <PDB> <MILLI-MSD> <ID>" + os.linesep)
                for vPack in viewList:
                    listf.write("VIEW {} {} {} {}".format(*vPack) + os.linesep)

    else:
        if viewList:
            # reserve arrays and convert data to one-hot
            print("reserve data...")
            sourceArray = np.zeros((len(viewList), 70, 70, 70), dtype=np.float32)
            markedArray = np.zeros((len(viewList), 70, 70, 70, 4), dtype=np.int32)
            noMaskArray = np.zeros((len(viewList), 70, 70, 70), dtype=np.bool)

            print("load data...")
            for it, pack in enumerate(viewList):
                print("\r -> loading ({:5}/{}) - {} ... ".format(it+1, len(viewList), pack), end="")
                sys.stdout.flush()

                sourcePath = os.path.join(args.datasrc, "{}-{}_view-{}_{}_source.npz".format(*pack))
                markedPath = os.path.join(args.datasrc, "{}-{}_view-{}_{}_marked.npz".format(*pack))
                noMaskPath = os.path.join(args.datasrc, "{}-{}_view-{}_{}_mask.npz".format(*pack))

                sourceView = np.load(sourcePath)['map']
                markedView = np.load(markedPath)['map']
                noMaskView = np.load(noMaskPath)['map']

                if sourceView.shape != markedView.shape[0:3] or np.any(sourceView.shape < np.asarray([70,70,70])):
                    print(" -> Shape error! ({}, {})".format(str(sourceView.shape), str(markedView.shape)))
                    continue

                ## Turn channel labels to one-hot labels:
                # 1) Add new dimension to label array with base value 0.01 .
                # 2) Calculate argmax on last axis (prediction channels).
                # 3) Get element wise boolean truth for label values >= max value
                #  -> Whatever channel was the maximum is now True, all others False.
                #    (Except for equal channels where there are now multiple True values)
                #  -> "Empty" voxels were automatically labeled by having no channel above 0.1 .
                # 4) Remove double true values by first applying argmax (Returns first True index) and then using
                #    eye to convert the index data to one-hot values.
                # (Order of the channels is least-common to most-common -> less common have preference!)
                resizedView = np.zeros((70,70,70,4), dtype=np.float32)
                resizedView[:,:,:,3] = 0.01
                resizedView[:,:,:,0:3] = markedView
                argmaxView = np.argmax(resizedView, axis=-1)

                #print(argmaxView)
                #print("{}, {}, {}, {}".format(*[float(np.count_nonzero(np.equal(argmaxView, i)))/float(argmaxView.size) for i in range(0,4)]))

                onehotView = np.eye(4, dtype=np.int32)[argmaxView]

                #print(onehotView)

                sourceArray[it] = sourceView
                markedArray[it] = onehotView#[10:30, 10:30, 10:30]
                noMaskArray[it] = noMaskView

                """print('|' + "=" * 70 + '|')
                for y in range(0,70):
                    print("|", end="")
                    for x in range(0,70):
                        print("{}".format(argmaxView[0,y,x].item()), end="")
                    print("|")
                print('|' + "=" * 70 + '|')"""

            print("")
            # calculate weight vector
            print("calculate weight vector...")
            global weightArray
            sumVector = np.sum(markedArray, axis=(0,1,2,3))

            print(" -> sum vector:    [{:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}]".format(*tuple(sumVector)))

            maxSum = np.max(sumVector)
            weightArray = np.divide(maxSum, sumVector)
            print(" -> weight vector: [{:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}]".format(*tuple(weightArray)))

        # create network folder
        if not os.path.isdir(args.network):
            os.makedirs(args.network)

        # run config
        proto_config = tf.ConfigProto()
        proto_config.gpu_options.allow_growth = True

        nn_config = tf.estimator.RunConfig().replace(
            save_checkpoints_steps = 5000,  # Save checkpoints every 5000 steps.
            keep_checkpoint_max = 10  # Retain the 10 most recent checkpoints.
            #,session_config = proto_config
        )


        # Estimator
        annotator = tf.estimator.Estimator(model_fn=unet_model_fn, model_dir=args.network, config=nn_config)

        # Logging
        #tensors_to_log = {"probabilities": "softmax_tensor"}#, "accuracy": "t_accuracy"}
        #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

        # execute command
        if args.command.lower() == 'train':
            # Train the model
            train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x": sourceArray},
                                                                y = {"y": markedArray, "m": noMaskArray},
                                                                     #"w": weightArray},
                                                                batch_size = 100,
                                                                num_epochs = None,
                                                                shuffle    = True)
            annotator.train(input_fn = train_input_fn,
                            steps    = args.steps,
                            hooks    = []) # logging_hook

        elif args.command.lower() in ['eval', 'predict']:
            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": sourceArray},
                                                               y={"y": markedArray, "m": noMaskArray},
                                                               num_epochs=5, shuffle=False)

            if args.command.lower() == 'eval':
                eval_results = annotator.evaluate(input_fn=eval_input_fn)
                print(eval_results)

            else:
                print("Predicting...")
                pred_results = annotator.predict(input_fn=eval_input_fn)

                os.makedirs(args.output, exist_ok=True)
                print("Saving predictions to {}...".format(args.output))
                for pack, prediction in zip(viewList, pred_results):
                    fileName = "{}-{}_view-{}_{}_predict.npy".format(*pack)
                    outPath = os.path.join(args.output, fileName)
                    print(" -> " + fileName)
                    np.save(outPath, prediction)

        elif args.command.lower() in ['map-predict', 'list-predict']:
            mrcPool = []
            if args.command.lower() == 'map-predict':
                mrcPool = [args.dataset]
            else:
                with open(args.dataset, 'r') as lFd:
                    #FIXME: There is a better way to pass/construct this path
                    mrcPool = [os.path.join(args.datasrc, "EMD-{0}/map/emd_{0}.map".format(x.strip().split('-')[0])) for x in lFd.readlines() if x.strip()[0] != '#']

            mrcPool.sort()

            for mrcPath in mrcPool:
                if not os.path.isfile(mrcPath):
                    mrcPath = mrcPath + '.gz'
                print("Loading mrc file '{}'...".format(mrcPath))
                with mrcfile.open(mrcPath) as mrcFd:
                    # extract map and cell vectors
                    mrcData = np.asfarray(mrcFd.data)
                    voxelSize = np.asarray([mrcFd.voxel_size.x, mrcFd.voxel_size.y, mrcFd.voxel_size.z], dtype=np.float64)
                    cellVec = np.asarray([mrcFd.header.nxstart, mrcFd.header.nystart, mrcFd.header.nzstart], dtype=np.float64)
                    cellA_XYZ = (mrcFd.header.cella.x, mrcFd.header.cella.y, mrcFd.header.cella.z)

                    # Reorder axes (see colorEMMap.py for full explanation)
                    axisSort = np.subtract(3, [mrcFd.header.maps, mrcFd.header.mapr, mrcFd.header.mapc])
                    print(" -> AxisSort: " + str(axisSort))
                    mrcData = np.transpose(mrcData, axes=axisSort)

                    # We also have to adjust the cell vectors which follow XYZ convention:
                    vectorSort = np.subtract([mrcFd.header.mapc, mrcFd.header.mapr, mrcFd.header.maps], 1)
                    cellVec = np.asarray([cellVec[i] for i in vectorSort])
                    voxelSize = np.asarray([voxelSize[i] for i in vectorSort])
                    cellA_XYZ = np.asarray([cellA_XYZ[i] for i in vectorSort])
                    print(" -> VoxelSize: " + str(voxelSize))

                # rescale mrc if voxel size out of bounds (same as training data generation!)
                zoomFactor = np.asarray([1, 1, 1], dtype=np.float)
                oldVoxelSize, oldCellVec = voxelSize, cellVec
                rescaledFlag = False
                originalMrcData = mrcData
                if not args.no_zoom and (np.max(voxelSize) > 1.2 or np.min(voxelSize) < 1.0):
                    print(" -> VoxelSize {} out of [1.0; 1.2] bounds. Rescaling to 1.1 A/V.".format(voxelSize))
                    targetScale = np.asarray([1.1, 1.1, 1.1], dtype=np.float)
                    zoomFactor = np.divide(voxelSize, targetScale)
                    print(" ... previous shape, vector, voxel: {}, {}, {}".format(str(mrcData.shape), str(cellVec), str(voxelSize)))
                    mrcData = ndimage.zoom(originalMrcData, zoom=zoomFactor)
                    cellVec = np.multiply(cellVec, zoomFactor)
                    voxelSize = targetScale
                    print(" ... rescaled shape, vector, voxel: {}, {}, {}".format(str(mrcData.shape), str(cellVec), str(voxelSize)))
                    rescaledFlag = True

                print("Tiling mrc data...")
                gridShape = np.ceil(np.divide(np.subtract(mrcData.shape, 20), 20)).astype(np.int)
                gridSize = np.prod(gridShape)
                print(" -> Grid shape: {}".format(gridShape))
                sourceArray = np.zeros((np.prod(gridShape), 40, 40, 40), dtype=np.float32)

                for midx in np.ndindex((gridShape[0], gridShape[1], gridShape[2])):
                    idx = np.sum(np.multiply(midx, (gridShape[1]*gridShape[2], gridShape[2], 1)))

                    ftl = np.minimum(np.multiply(midx, 20), np.subtract(mrcData.shape, 40))
                    bbr = np.add(ftl, 40)
                    print("\r {} -> {:6}/{} -> {}".format(str(midx), idx, gridSize, str(ftl)), end="")
                    sourceArray[idx] = mrcData[ftl[0]:bbr[0], ftl[1]:bbr[1], ftl[2]:bbr[2]]

                print("")
                print("Predicting...")
                eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": sourceArray}, num_epochs=1, shuffle=False)
                pred_results = annotator.predict(input_fn=eval_input_fn)

                print("Rebuilding annotated mrc maps...")
                annotatedMap = np.zeros((*mrcData.shape, 4), dtype=np.float32)
                for i, pmap in enumerate(pred_results):
                    x = i % gridShape[2]
                    y = i // gridShape[2] % gridShape[1]
                    z = i // (gridShape[2] * gridShape[1])
                    ftl = np.minimum(np.add(np.multiply([z,y,x], 20), 10), np.subtract(mrcData.shape, 40)).astype(np.int)
                    bbr = np.add(ftl, 20)
                    print("\r {:6}/{} -> {}, {}, {}...     ".format(i, gridSize, z,y,x), end="")
                    annotatedMap[ftl[0]:bbr[0], ftl[1]:bbr[1], ftl[2]:bbr[2], :] = pmap["probabilities"]

                print("")

                # return to original scaling (same as training data generation!)
                if rescaledFlag:
                    print(" -> reverting rescaling for mrc/npz files.")
                    unzoomFactor = np.divide(1, zoomFactor)
                    print(" -> " + str(unzoomFactor))
                    unzoomedAnnotatedList = []
                    for i in range(0, annotatedMap.shape[-1]):
                        unzoomedAnnotatedList.append(ndimage.zoom(annotatedMap[:, :, :, i], zoom=unzoomFactor))
                    unzoomedAnnotatedMap = np.stack(unzoomedAnnotatedList, axis=-1) #[:, :, :, 0:3]
                    print(" ... rescaled shape: {}".format(str(unzoomedAnnotatedMap.shape)))
                else:
                    unzoomedAnnotatedMap = annotatedMap #[:, :, :, 0:3]

                # Multiply predictions with original density in order to produce coot/chimera viewable maps.
                # Network may amplify random noise outside protein due it's independent look at
                if args.density:
                    print(" -> Multiplying predictions with original density.")
                    unzoomedAnnotatedMap = np.multiply(unzoomedAnnotatedMap.astype(np.float32), np.expand_dims(originalMrcData, -1).astype(np.float32))

                # save predicted output
                prefix = os.path.splitext(os.path.basename(mrcPath))[0]

                if not args.nomrc:
                    print("Saving annotated mrc maps...")
                    for i, suffix in enumerate(['sheet', 'helix', 'npair']):
                        outMrcPath = os.path.join(args.output, "{}_{}.mrc".format(prefix, suffix))
                        print(" -> {} : '{}'".format(suffix, outMrcPath))
                        saveMRC(np.squeeze(unzoomedAnnotatedMap[:, :, :, i]), oldVoxelSize, oldCellVec, cellA_XYZ, outMrcPath, zip=False)

                npz4Path = os.path.join(args.output, prefix + '_predict4.npz')
                print("Saving 4 channel npz file to '{}'...".format(npz4Path))
                stampA = time.time()
                np.savez_compressed(npz4Path, map=unzoomedAnnotatedMap)
                stampB = time.time()
                print("Time: {} - {} = {}".format(stampB, stampA, stampB - stampA))


if __name__ == "__main__":
    tf.app.run()


