# title: Class weight dan bobot loss
# when: Setelah threshold, bila kelas minor tetap diabaikan.
# avoid: Log loss: bobot mengubah probabilitas.
# tags: class_weight sample_weight bobot
import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

def weights_for(y):
    """Kembalikan (sample_weight, bobot per kelas untuk loss PyTorch)."""
    y = np.asarray(y)
    return compute_sample_weight("balanced", y), compute_class_weight("balanced", classes=np.unique(y), y=y).astype("float32")
