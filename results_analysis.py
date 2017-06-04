import numpy as np
np.set_printoptions(linewidth=200)
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


pred = np.load('test_results.npy')
truth = np.load('truth.npy')

print pred.shape, truth.shape


prob = softmax(pred[:,:2])
# print prob.sum(axis = 1)
print pred


print 'Detection accuracy: ', np.sum(np.argmax(prob, axis = 1) == truth[:,0].astype(np.bool))/float(prob.shape[0])

print 'Gender accuracy: ', np.sum(np.argmax(softmax(pred[:,68:70]), axis = 1) == truth[:,-1].astype(np.bool))/float(prob.shape[0])

# detection
fpr, tpr, thresholds = roc_curve(truth[:,0], prob[:,1])
precision, recall, th = precision_recall_curve(truth[:,0], prob[:,1])

# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# # plt.savefig('detection_pr_re.eps', format='eps', dpi=1000)
# plt.show()


# plt.plot(fpr, tpr)
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# # plt.savefig('detection_roc.eps', format='eps', dpi=1000)
# plt.show()

