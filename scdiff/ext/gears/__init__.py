"""GEARS module.

This module is adapted from https://github.com/snap-stanford/GEARS

@article{roohani2023predicting,
  title={Predicting transcriptional outcomes of novel multigene perturbations with GEARS},
  author={Roohani, Yusuf and Huang, Kexin and Leskovec, Jure},
  journal={Nature Biotechnology},
  pages={1--9},
  year={2023},
  publisher={Nature Publishing Group US New York}
}

"""
__all__ = ["GEARS", "PertData"]


from .gears import GEARS
from .pertdata import PertData
