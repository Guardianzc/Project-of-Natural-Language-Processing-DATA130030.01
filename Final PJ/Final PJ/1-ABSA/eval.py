# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import codecs
from typing import List
from sklearn import metrics


def calc_acc_f1(golds: List[str], predicts: List[str]) -> tuple:
    return metrics.accuracy_score(golds, predicts), \
           metrics.f1_score(golds, predicts, labels=['positive', 'negative', 'neutral'], average='macro')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("run this script: "
                        "python eval.py $golds_file_path $predict_file_path")
    golds_file = sys.argv[1]
    predicts_file = sys.argv[2]
    golds = []
    with codecs.open(golds_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            golds.append(line.strip().split('\t')[-1])
    predicts = []
    with codecs.open(predicts_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            predicts.append(line.strip())
    acc, f1_score = calc_acc_f1(golds, predicts)
    print("acc: {}, f1: {}".format(acc, f1_score))
