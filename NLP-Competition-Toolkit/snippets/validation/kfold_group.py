# title: Group K-Fold dan leave-one-source-out
# when: Penulis, video, thread, query, dokumen, atau klaster duplikat tidak melintasi test.
# avoid: Test ternyata acak: skor terlalu pesimistis. Laporkan random dan grup.
# tags: group query dokumen sumber
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

def folds_group(groups, n=5, seed=42):
    return list(GroupKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(groups)), groups=groups))

def folds_leave_source(sources):
    return list(LeaveOneGroupOut().split(np.zeros(len(sources)), groups=sources))
