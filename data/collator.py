from typing import List, Dict

import torch

from model.data_collator import DataCollator

class DrugCollator(DataCollator):
    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}
        """
        data_dict = super().__call__(examples)

        data_dict["drug"] = torch.stack([example["drug"] for example in examples], dim=0)
        data_dict["target"] = torch.stack([example["target"] for example in examples], dim=0)

        return data_dict