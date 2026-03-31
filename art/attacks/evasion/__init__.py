"""
Module providing evasion attacks under a common interface.
"""
# pylint: disable=C0413

from art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from art.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from art.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from art.attacks.evasion.dpatch import DPatch
from art.attacks.evasion.dpatch_robust import RobustDPatch
