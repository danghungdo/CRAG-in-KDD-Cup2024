Task 1
# gemma 27b
{'score': -0.16814486326681444, 'exact_accuracy': 0.1334072431633407, 'accuracy': 0.1334072431633407, 'hallucination': 0.30155210643015523, 'missing': 0.5650406504065041, 'n_miss': 1529, 'n_correct': 361, 'n_correct_exact': 361, 'total': 2706}

# llama 70b
{'score': 0.20066518847006654, 'exact_accuracy': 0.1334072431633407, 'accuracy': 0.3178122690317812, 'hallucination': 0.1171470805617147, 'missing': 0.5650406504065041, 'n_miss': 1529, 'n_correct': 860, 'n_correct_exact': 361, 'total': 2706}

# bge router
'score': 0.18218773096821872, 'exact_accuracy': 0.1337767923133777, 'accuracy': 0.30339985218034, 'hallucination': 0.12121212121212122, 'missing': 0.5753880266075388, 'n_miss': 1557, 'n_correct': 821, 'n_correct_exact': 362, 'total': 2706

Task 2

{'score': -0.0003695491500369519, 'exact_accuracy': 0.27309682187730966, 'accuracy': 0.27309682187730966, 'hallucination': 0.2734663710273466, 'missing': 0.4534368070953437, 'n_miss': 1227, 'n_correct': 739, 'n_correct_exact': 739, 'total': 2706}


domain 96%
dynamnic 69%

Task 1: gemma 27b
{'score': -0.1807095343680709, 'exact_accuracy': 0.1334072431633407, 'accuracy': 0.1334072431633407, 'hallucination': 0.31411677753141165, 'missing': 0.5524759793052476, 'n_miss': 1495, 'n_correct': 361, 'n_correct_exact': 361, 'total': 2706}

Task 1: llama 70b
{'score': 0.18588322246858824, 'exact_accuracy': 0.1334072431633407, 'accuracy': 0.31670362158167037, 'hallucination': 0.13082039911308205, 'missing': 0.5524759793052476, 'n_miss': 1495, 'n_correct': 857, 'n_correct_exact': 361, 'total': 2706}

Task 2:
 {'score': 0.3337028824833703, 'exact_accuracy': 0.27161862527716185, 'accuracy': 0.43902439024390244, 'hallucination': 0.10532150776053215, 'missing': 0.4556541019955654, 'n_miss': 1233, 'n_correct': 1188, 'n_correct_exact': 735, 'total': 2706}

finetuned version
 {'score': 0.33813747228381374, 'exact_accuracy': 0.27605321507760533, 'accuracy': 0.44198078344419806, 'hallucination': 0.10384331116038432, 'missing': 0.4541759053954176, 'n_miss': 1229, 'n_correct': 1196, 'n_correct_exact': 747, 'total': 2706}

 task 2: bge router
 {'score': 0.33665927568366594, 'exact_accuracy': 0.2845528455284553, 'accuracy': 0.44382852919438287, 'hallucination': 0.10716925351071692, 'missing': 0.4490022172949002, 'n_miss': 1215, 'n_correct': 1201, 'n_correct_exact': 770, 'total': 2706}

bge + pe
 {'score': 0.34737620103473765, 'exact_accuracy': 0.29637841832963785, 'accuracy': 0.4933481152993348, 'hallucination': 0.14597191426459719, 'missing': 0.360679970436068, 'n_miss': 976, 'n_correct': 1335, 'n_correct_exact': 802, 'total': 2706}

 bge + pe + fpd
 {'score': 0.287509238728751, 'exact_accuracy': 0.29859571322985956, 'accuracy': 0.5173688100517368, 'hallucination': 0.22985957132298596, 'missing': 0.25277161862527714, 'n_miss': 684, 'n_correct': 1400, 'n_correct_exact': 808, 'total': 2706}


domain adapter

0.9878048780487805
              precision    recall  f1-score   support

     finance       1.00      1.00      1.00       661
       movie       0.99      0.98      0.98       611
       music       0.98      0.99      0.99       373
        open       0.97      0.98      0.97       542
      sports       1.00      1.00      1.00       519

    accuracy                           0.99      2706
   macro avg       0.99      0.99      0.99      2706
weighted avg       0.99      0.99      0.99      2706

[[658   1   0   2   0]
 [  1 597   1  12   0]
 [  0   1 370   2   0]
 [  1   2   7 531   1]
 [  0   1   0   1 517]]

old dynamic adapter

 0.6906873614190687
               precision    recall  f1-score   support

fast-changing       0.71      0.31      0.44       353
    real-time       0.83      0.67      0.74       267
slow-changing       0.44      0.88      0.59       583
       static       0.91      0.71      0.80      1503

     accuracy                           0.69      2706
    macro avg       0.72      0.64      0.64      2706
 weighted avg       0.77      0.69      0.70      2706

[[ 111   25  169   48]
 [  20  180   60    7]
 [  12    7  514   50]
 [  14    5  420 1064]]

new dynamic adapter

 0.8983739837398373
               precision    recall  f1-score   support

fast-changing       0.86      0.86      0.86       353
    real-time       0.89      0.97      0.92       267
slow-changing       0.79      0.86      0.83       583
       static       0.96      0.91      0.93      1503

     accuracy                           0.90      2706
    macro avg       0.87      0.90      0.89      2706
 weighted avg       0.90      0.90      0.90      2706

[[ 302   16   26    9]
 [   7  258    1    1]
 [  21    7  504   51]
 [  20   10  106 1367]]

simple Q split 1
 {'score': 0.2089136490250696, 'exact_accuracy': 0.16434540389972144, 'accuracy': 0.28969359331476324, 'hallucination': 0.0807799442896936, 'missing': 0.6295264623955432, 'n_miss': 226, 'n_correct': 104, 'n_correct_exact': 59, 'total': 359}

 {'score': 0.21448467966573825, 'exact_accuracy': 0.1671309192200557, 'accuracy': 0.2924791086350975, 'hallucination': 0.07799442896935933, 'missing': 0.6295264623955432, 'n_miss': 226, 'n_correct': 105, 'n_correct_exact': 60, 'total': 359}

